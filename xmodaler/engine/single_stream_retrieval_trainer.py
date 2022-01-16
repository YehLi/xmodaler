# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import time
import math 
import copy
from tqdm import tqdm
import torch
from xmodaler.functional import dict_to_cuda, expand_tensor, clip_t_inputs, clip_v_inputs
from .defaults import DefaultTrainer
from xmodaler.config import kfg
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY

__all__ = ['SingleStreamRetrievalTrainer', 'SingleStreamRetrievalTrainerHardNegatives']

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


@ENGINE_REGISTRY.register()
class SingleStreamRetrievalTrainerHardNegatives(SingleStreamRetrievalTrainer):
    def __init__(self, cfg):
        super(SingleStreamRetrievalTrainerHardNegatives, self).__init__(cfg)
        self.num_hard_sample = cfg.DATALOADER.NEGATIVE_SIZE
        assert self.num_hard_sample > 0

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)
        data_time = time.perf_counter() - start

        data = comm.unwrap_model(self.model).preprocess_batch(data)
        
        # clip visual & text inputs for faster forward
        clipped_data = self.clip_inputs(data)
        data.update(clipped_data)

        # evaluation for hard negatives minding
        with torch.no_grad():
            tmp_data = copy.deepcopy(data)
            hard_data = self.hard_negative_mining(tmp_data)
            data.update(hard_data)

        # forward with hard
        outputs_dict = self.model(data)

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
        losses = sum(losses_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(losses_dict, data_time)
        self.optimizer.step()

    def clip_inputs(self, data):
        v_feats = data[kfg.ATT_FEATS]
        v_loc = data[kfg.ATT_FEATS_LOC]
        v_masks = data[kfg.ATT_MASKS]
        v_feats, v_loc, v_masks = clip_v_inputs(v_feats, v_loc, v_masks)

        u_tokens_ids = data[kfg.U_TOKENS_IDS]
        u_tokens_type = data[kfg.U_TOKENS_TYPE]
        tokens_masks = data[kfg.TOKENS_MASKS]
        u_tokens_ids, u_tokens_type, tokens_masks = clip_t_inputs(u_tokens_ids, u_tokens_type, tokens_masks)

        return {
            kfg.ATT_FEATS: v_feats,
            kfg.ATT_FEATS_LOC: v_loc,
            kfg.ATT_MASKS: v_masks,
            kfg.U_TOKENS_IDS: u_tokens_ids,
            kfg.U_TOKENS_TYPE: u_tokens_type,
            kfg.TOKENS_MASKS: tokens_masks,
        }
        
    @torch.no_grad()
    def hard_negative_mining(self, data):
        self.model.eval()
        batch_size = data[kfg.ATT_FEATS].size(0)
        device = data[kfg.ATT_FEATS].device

        # extract origin inputs
        v_feats = data[kfg.ATT_FEATS]
        v_masks = data[kfg.ATT_MASKS]
        v_loc = data[kfg.ATT_FEATS_LOC]
        u_tokens_ids = data[kfg.U_TOKENS_IDS]
        tokens_masks = data[kfg.TOKENS_MASKS]
        u_tokens_type = data[kfg.U_TOKENS_TYPE]

        # expand visual input
        (v_feats2, v_masks2, v_loc2) = [
            expand_tensor(x, batch_size, dim=1) \
            for x in (v_feats, v_masks, v_loc)
        ]
        # expand text input
        (u_tokens_ids2, tokens_masks2, u_tokens_type2) = [
            expand_tensor(x, batch_size, dim=0) \
            for x in (u_tokens_ids, tokens_masks, u_tokens_type)
        ]

        # calculate scores by batches
        total_num = u_tokens_ids2.shape[0]
        scores = torch.zeros([total_num, 1], device=device)

        bs = 1024
        bn = math.ceil(total_num / bs)
        for i in range(bn):
            st = i*bs
            ed = (i+1)*bs
            ed = total_num if ed > total_num else ed

            tmp_data = {
                kfg.ATT_FEATS: v_feats2[st:ed],
                kfg.ATT_FEATS_LOC: v_loc2[st:ed],
                kfg.ATT_MASKS: v_masks2[st:ed],
                kfg.U_TOKENS_IDS: u_tokens_ids2[st:ed],
                kfg.U_TOKENS_TYPE: u_tokens_type2[st:ed],
                kfg.TOKENS_MASKS: tokens_masks2[st:ed],
            }
            data.update(tmp_data)

            scores_batch = self.model(data)[kfg.OUTPUT]
            scores[st:ed] = scores_batch

        scores = scores.view(batch_size, batch_size)
        # clear diagonals
        I = torch.eye(scores.size(0), device=device) > .5
        scores = scores.masked_fill_(I, -99999.0)

        num_options = self.num_hard_sample + 1

        _, hardest_indexes = torch.topk(scores, dim=-1, k=self.num_hard_sample)
        hardest_indexes = hardest_indexes.view(-1)
        row_indexes = expand_tensor(torch.arange(batch_size, device=scores.device), self.num_hard_sample, dim=1)
        selected_indexes = row_indexes * batch_size + hardest_indexes
        # select hardest sent
        u_tokens_ids_hard = u_tokens_ids2[selected_indexes].view(batch_size, self.num_hard_sample, -1)
        u_tokens_type_hard = u_tokens_type2[selected_indexes].view(batch_size, self.num_hard_sample, -1)
        tokens_masks_hard = tokens_masks2[selected_indexes].view(batch_size, self.num_hard_sample, -1)
        
        # Conacat to original positive sample (1 pos + self.num_hard_sample neg)
        v_feats = expand_tensor(v_feats, num_options, dim=1)
        v_masks = expand_tensor(v_masks, num_options, dim=1)
        v_loc = expand_tensor(v_loc, num_options, dim=1)

        u_tokens_ids = torch.cat([u_tokens_ids.unsqueeze(1), u_tokens_ids_hard], dim=1).view([-1] + list(u_tokens_ids.shape[1:]))
        u_tokens_type = torch.cat([u_tokens_type.unsqueeze(1), u_tokens_type_hard], dim=1).view([-1] + list(u_tokens_type.shape[1:]))
        tokens_masks = torch.cat([tokens_masks.unsqueeze(1), tokens_masks_hard], dim=1).view([-1] + list(tokens_masks.shape[1:]))

        self.model.train()
        # return the hard batches
        return {
            kfg.ATT_FEATS: v_feats,
            kfg.ATT_FEATS_LOC: v_loc,
            kfg.ATT_MASKS: v_masks,
            kfg.U_TOKENS_IDS: u_tokens_ids,
            kfg.U_TOKENS_TYPE: u_tokens_type,
            kfg.TOKENS_MASKS: tokens_masks,
            kfg.SAMPLE_PER_SAMPLE: num_options
        }