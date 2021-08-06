# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
import torch.distributed as dist
import random

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.utils.distributed import any_broadcast
from ..predictor import build_v_predictor, build_predictor_with_name
from .transformer_enc_dec import TransformerEncoderDecoder
from .build import META_ARCH_REGISTRY

from ..embedding import build_embeddings
from ..encoder import build_encoder
from ..predictor import build_predictor

__all__ = ["UniterPretrain", "UniterForMMUnderstanding", "UniterRetrieval"]

@META_ARCH_REGISTRY.register()
class UniterForMMUnderstanding(TransformerEncoderDecoder):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        max_seq_len,
        token_embed,
        visual_embed,
        encoder,
        decoder,
        predictor,
        greedy_decoder,
        beam_searcher,
        v_predictor,

        itm_predictor,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            token_embed=token_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher,
            v_predictor=v_predictor
        )
        self.itm_predictor = itm_predictor

    @classmethod
    def from_config(cls, cfg):
        ret = {
            # basic config
            "token_embed": build_embeddings(cfg, cfg.MODEL.TOKEN_EMBED.NAME),
            "visual_embed": build_embeddings(cfg, cfg.MODEL.VISUAL_EMBED.NAME),
            "encoder": build_encoder(cfg),
            "decoder": None,
            "predictor": build_predictor(cfg),

            "greedy_decoder": None,
            "beam_searcher": None,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "v_predictor": None,

            # uniter pretrain config, in order to load the pretrained pooler
            "itm_predictor": build_predictor_with_name(cfg, 'BertIsMatchedPredictor')
        }

        return ret

    def bind_or_init_weights(self):
        self.predictor.pooler = self.itm_predictor.pooler

    def get_extended_attention_mask(self, batched_inputs):
        if kfg.TOKENS_MASKS not in batched_inputs:
            batched_inputs[kfg.TOKENS_MASKS] = torch.ones((batched_inputs[kfg.ATT_MASKS].size(0), self.max_seq_len)).cuda()

        tmasks = batched_inputs[kfg.TOKENS_MASKS]
        tmasks = tmasks.to(dtype=next(self.parameters()).dtype)
        ext_u_tmasks = tmasks.unsqueeze(1).unsqueeze(2)
        ext_u_tmasks = (1.0 - ext_u_tmasks) * -10000.0
        
        vmasks = batched_inputs[kfg.ATT_MASKS]
        vmasks = vmasks.to(dtype=next(self.parameters()).dtype)
        vmasks = vmasks.unsqueeze(1).unsqueeze(2)
        ext_vmasks = (1.0 - vmasks) * -10000.0

        return {
            kfg.TOKENS_MASKS: tmasks,
            kfg.EXT_U_TOKENS_MASKS: ext_u_tmasks,
            kfg.ATT_MASKS: vmasks,
            kfg.EXT_ATT_MASKS: ext_vmasks
        }


@META_ARCH_REGISTRY.register()
class UniterRetrieval(UniterForMMUnderstanding):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        max_seq_len,
        token_embed,
        visual_embed,
        encoder,
        decoder,
        predictor,
        greedy_decoder,
        beam_searcher,
        v_predictor,

        itm_predictor,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            token_embed=token_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher,
            v_predictor=v_predictor,
            itm_predictor=itm_predictor
        )

    def bind_or_init_weights(self):
        self.predictor.pooler = self.itm_predictor.pooler

        self.predictor.cls.weight.data = self.itm_predictor.is_match_cls.weight.data[:1, :]
        self.predictor.cls.bias.data = self.itm_predictor.is_match_cls.bias.data[:1]


@META_ARCH_REGISTRY.register()
class UniterPretrain(TransformerEncoderDecoder):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        max_seq_len,
        token_embed,
        visual_embed,
        encoder,
        decoder,
        predictor,
        greedy_decoder,
        beam_searcher,
        v_predictor,
        v_regressor,
        itm_predictor,
        tasks,
        mix_ratio
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            token_embed=token_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher,
            v_predictor=v_predictor
        )
        self.v_regressor = v_regressor
        self.itm_predictor = itm_predictor
        self.v_regressor.weight = self.visual_embed.embeddings.weight

        # prepare for random sample pretraining task
        self.sampling_pool = []
        for name, r in zip(tasks, mix_ratio):
            self.sampling_pool.extend([name]*r)

        try:
            self.world_size = torch.distributed.get_world_size()
            self.distributed = True
        except:
            self.distributed = False
            self.world_size = 1

    @classmethod
    def from_config(cls, cfg):
        assert cfg.MODEL.BERT.V_TARGET_SIZE > 0
        assert len(cfg.MODEL.PRETRAIN_TASKS) == len(cfg.MODEL.PRETRAIN_TASKS_MIX_RATIO)

        ret = {
            # basic config
            "token_embed": build_embeddings(cfg, cfg.MODEL.TOKEN_EMBED.NAME),
            "visual_embed": build_embeddings(cfg, cfg.MODEL.VISUAL_EMBED.NAME),
            "encoder": build_encoder(cfg),
            "decoder": None,
            "predictor": build_predictor(cfg),

            "greedy_decoder": None,
            "beam_searcher": None,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,

            # uniter pretrain config
            "v_predictor": build_v_predictor(cfg),
            "v_regressor": build_predictor_with_name(cfg, cfg.MODEL.V_REGRESSOR),
            "itm_predictor": build_predictor_with_name(cfg, cfg.MODEL.ITM_PREDICTOR),
            "tasks": tuple(cfg.MODEL.PRETRAIN_TASKS),
            'mix_ratio': tuple(cfg.MODEL.PRETRAIN_TASKS_MIX_RATIO)
        }

        return ret

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        super().add_config(cfg, tmp_cfg)
        cfg.MODEL.V_REGRESSOR = ''
        cfg.MODEL.ITM_PREDICTOR = ''
        cfg.MODEL.PRETRAIN_TASKS = ['itm', 'mlm', 'mrfr', 'mrc-kl']
        cfg.MODEL.PRETRAIN_TASKS_MIX_RATIO = [1, 1, 1, 1]

    def preprocess_inputs(self, inputs, task_name):
        if task_name == 'itm':
            inputs[kfg.ATT_FEATS] = inputs[kfg.ATT_FEATS_WO_MASK]
            inputs[kfg.U_TOKENS_IDS] = inputs[kfg.U_TOKENS_IDS_WO_MASK]
        elif task_name == 'mlm':
            inputs[kfg.ATT_FEATS] = inputs[kfg.ATT_FEATS_WO_MASK]
        elif task_name == 'mrfr':
            inputs[kfg.U_TOKENS_IDS] = inputs[kfg.U_TOKENS_IDS_WO_MASK]
            inputs[kfg.V_TARGET] = inputs[kfg.ATT_FEATS_WO_MASK]
        elif task_name == 'mrc-kl':
            inputs[kfg.U_TOKENS_IDS] = inputs[kfg.U_TOKENS_IDS_WO_MASK]
        else:
            raise NotImplementedError

        if task_name != 'itm':
            # mask out neg vl pair
            itm_neg_label = inputs[kfg.ITM_NEG_LABEL]

            image_label = inputs[kfg.V_TARGET_LABELS] * (itm_neg_label == 0).long().unsqueeze(1)
            inputs[kfg.V_TARGET_LABELS][image_label == 0] = -1
            masked_lm_labels = inputs[kfg.U_TARGET_IDS] * (itm_neg_label == 0).long().unsqueeze(1)
            inputs[kfg.U_TARGET_IDS][masked_lm_labels == 0] = -1

        return inputs

    def _forward(self, batched_inputs):
        inputs = batched_inputs
        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        # random select a task
        task_name = random.choice(self.sampling_pool)
        if self.distributed:
            # make sure all process is training same task
            dist.barrier()
            task_name = any_broadcast(task_name, 0, n_gpu=self.world_size)
            dist.barrier()

        # Preprocess
        inputs = self.preprocess_inputs(inputs, task_name)

        # Forward embeddings
        ve_out = self.visual_embed(batched_inputs)
        inputs.update(ve_out)

        te_out = self.token_embed(batched_inputs)
        inputs.update(te_out)

        # Forward encoder
        encoder_out = self.encoder(inputs)
        inputs.update(encoder_out)

        # Forward Head
        if task_name == 'itm':
            scores = self.itm_predictor(inputs)
            inputs.update(scores)

        elif task_name == 'mlm':
            tlogits = self.predictor(inputs)
            inputs.update(tlogits)
    
        elif task_name == 'mrfr':
            vregs = self.v_regressor(inputs)
            inputs.update(vregs)
        
        elif task_name == 'mrc-kl':
            vlogits = self.v_predictor(inputs)
            inputs.update(vlogits)

        return inputs

