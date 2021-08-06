# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import tqdm
from tqdm import tqdm
import torch
from xmodaler.functional import dict_to_cuda
from .defaults import DefaultTrainer
from xmodaler.config import kfg
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY

__all__ = ['SingleStreamRetrievalTrainer']

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


@ENGINE_REGISTRY.register()
class SingleStreamRetrievalTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(SingleStreamRetrievalTrainer, self).__init__(cfg)

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        score_matrix, gt_iidxes = inference(cfg, model, test_data_loader)
        comm.synchronize()

        if comm.get_world_size() > 1:
            all_score = concat_all_gather(score_matrix)
            comm.synchronize()
            all_gt_iidxes = concat_all_gather(gt_iidxes)
            comm.synchronize()

            if not comm.is_main_process():
                # NOTE: only use rank0 to compute final scores
                return 'ignore'

        else:
            all_score = score_matrix
            all_gt_iidxes = gt_iidxes

        all_gt_iidxes = tuple(all_gt_iidxes.view(-1).cpu().tolist())
        eval_res = itm_eval(all_score, all_gt_iidxes)
        return eval_res


@torch.no_grad()
def inference(cfg, model, test_data_loader):
    model.eval()

    if comm.is_main_process:
        pbar = tqdm(total=len(test_data_loader))
    else:
        pbar = NoOp()

    total_txt_num = len(test_data_loader)
    score_matrix = None
    gt_iidxes = (torch.zeros(total_txt_num, dtype=torch.long) - 1).cuda()

    for i, mini_batches in enumerate(test_data_loader):
        comm.synchronize()

        assert len(mini_batches) == 1, "input batch size > 1"
        mini_batches = mini_batches[0]

        if score_matrix is None:
            # init score_matrix
            total_img_num = int(mini_batches[0]['total_img_num'])
            score_matrix = torch.zeros(total_txt_num, total_img_num, dtype=torch.float32).cuda()

        j = 0
        for batch in mini_batches:
            dict_to_cuda(batch)
            scores = model(batch)[kfg.OUTPUT]
            bs = scores.size(0)
            score_matrix.data[i, j:j+bs] = scores.data.squeeze(1)
            j += bs
        assert j == score_matrix.size(1)
        gt_iidxes[i] = batch['matched_imgfeatidx']
        pbar.update(1)

    model.train()
    pbar.close()
    gt_iidxes = gt_iidxes.unsqueeze(1)
    return score_matrix, gt_iidxes


@torch.no_grad()
def itm_eval(score_matrix, t2gtiidxes):
    # image retrieval
    total_txt_num = len(t2gtiidxes)
    _, rank_txt = score_matrix.topk(10, dim=1)
    gt_img_j = torch.LongTensor(t2gtiidxes).to(rank_txt.device).unsqueeze(1).expand_as(rank_txt)
    rank = (rank_txt == gt_img_j).nonzero()

    rank = rank[:, 1:]
    if rank.numel():
        ir_r1 = (rank < 1).sum().item() / total_txt_num
        ir_r5 = (rank < 5).sum().item() / total_txt_num
        ir_r10 = (rank < 10).sum().item() / total_txt_num
    else:
        ir_r1, ir_r5, ir_r10 = 0, 0, 0

    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3

    eval_log = {
        'img_r1': ir_r1,
        'img_r5': ir_r5,
        'img_r10': ir_r10,
        'img_r_mean': ir_mean
    }
    return eval_log