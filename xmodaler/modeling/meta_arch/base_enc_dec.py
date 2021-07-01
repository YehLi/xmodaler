import numpy as np
import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..embedding import build_embeddings
from ..encoder import build_encoder, add_encoder_config
from ..decoder import build_decoder, add_decoder_config
from ..predictor import build_predictor, add_predictor_config


class BaseEncoderDecoder(nn.Module, metaclass=ABCMeta):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        token_embed,
        visual_embed,
        encoder,
        decoder,
        predictor
    ):
        super(BaseEncoderDecoder, self).__init__()
        self.token_embed = token_embed
        self.visual_embed = visual_embed
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.vocab_size = vocab_size

    @classmethod
    def from_config(cls, cfg):
        return {
            "token_embed": build_embeddings(cfg, cfg.MODEL.TOKEN_EMBED.NAME),
            "visual_embed": build_embeddings(cfg, cfg.MODEL.VISUAL_EMBED.NAME),
            "encoder": build_encoder(cfg),
            "decoder": build_decoder(cfg),
            "predictor": build_predictor(cfg),
            "vocab_size": cfg.MODEL.VOCAB_SIZE
        }

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        add_encoder_config(cfg, tmp_cfg)
        add_decoder_config(cfg, tmp_cfg)
        add_predictor_config(cfg, tmp_cfg)

    @abstractmethod
    def get_extended_attention_mask(self, att_mask):
        pass

    @abstractmethod
    def forward(self):
        pass

    def preprocess_batch(self, batched_inputs):
        batch_size = len(batched_inputs)
        repeat_num = batched_inputs[0][kfg.TOKENS_IDS].shape[0]
        
        tokens_arr = [x[kfg.TOKENS_IDS] for x in batched_inputs]
        tokens_ids = torch.cat(tokens_arr, dim=0).cuda()

        target_arr = [x[kfg.TARGET_IDS] for x in batched_inputs]
        target_ids = torch.cat(target_arr, dim=0).cuda()

        feats_dim = batched_inputs[0][kfg.ATT_FEATS].shape[1]
        feats_num = [x[kfg.ATT_FEATS].shape[0] for x in batched_inputs]
        max_feats_num = max(feats_num)
        
        att_feats = np.zeros((batch_size, max_feats_num, feats_dim), dtype=np.float32)
        att_masks = np.zeros((batch_size, max_feats_num), dtype=np.float32)
        for i, num in enumerate(feats_num):
            att_feats[i, 0:num] = batched_inputs[i][kfg.ATT_FEATS]
            att_masks[i, 0:num] = 1
        att_feats = torch.from_numpy(att_feats).cuda()
        att_masks = torch.from_numpy(att_masks).cuda()

        att_feats = att_feats.view(batch_size, 1, max_feats_num, feats_dim).expand(batch_size, repeat_num, max_feats_num, feats_dim)
        att_feats = att_feats.reshape(-1, max_feats_num, feats_dim)
        att_masks = att_masks.view(batch_size, 1, max_feats_num).expand(batch_size, repeat_num, max_feats_num)
        att_masks = att_masks.reshape(-1, max_feats_num)

        return {
            kfg.TOKENS_IDS: tokens_ids,
            kfg.TARGET_IDS: target_ids,
            kfg.ATT_FEATS: att_feats,
            kfg.ATT_MASKS: att_masks
        }

    def preprocess_batch_test(self, batched_inputs):
        batch_size = len(batched_inputs)

        feats_dim = batched_inputs[0][kfg.ATT_FEATS].shape[1]
        feats_num = [x[kfg.ATT_FEATS].shape[0] for x in batched_inputs]
        max_feats_num = max(feats_num)
        
        att_feats = np.zeros((batch_size, max_feats_num, feats_dim), dtype=np.float32)
        att_masks = np.zeros((batch_size, max_feats_num), dtype=np.float32)
        for i, num in enumerate(feats_num):
            att_feats[i, 0:num] = batched_inputs[i][kfg.ATT_FEATS]
            att_masks[i, 0:num] = 1
        att_feats = torch.from_numpy(att_feats).cuda()
        att_masks = torch.from_numpy(att_masks).cuda()

        return {
            kfg.ATT_FEATS: att_feats,
            kfg.ATT_MASKS: att_masks
        }


