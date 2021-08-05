# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import copy
import numpy as np
import weakref
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from xmodaler.functional import pad_tensor, dict_to_cuda
from ..embedding import build_embeddings
from ..encoder import build_encoder, add_encoder_config
from ..decoder import build_decoder, add_decoder_config
from ..predictor import build_predictor, add_predictor_config
from ..decode_strategy import build_beam_searcher, build_greedy_decoder

class BaseEncoderDecoder(nn.Module, metaclass=ABCMeta):
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
        beam_searcher
    ):
        super(BaseEncoderDecoder, self).__init__()
        self.token_embed = token_embed
        self.visual_embed = visual_embed
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.greedy_decoder = greedy_decoder
        self.beam_searcher = beam_searcher
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    @classmethod
    def from_config(cls, cfg):
        return {
            "token_embed": build_embeddings(cfg, cfg.MODEL.TOKEN_EMBED.NAME),
            "visual_embed": build_embeddings(cfg, cfg.MODEL.VISUAL_EMBED.NAME),
            "encoder": build_encoder(cfg),
            "decoder": build_decoder(cfg),
            "predictor": build_predictor(cfg),
            "greedy_decoder": build_greedy_decoder(cfg),
            "beam_searcher": build_beam_searcher(cfg),
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN
        }

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        add_encoder_config(cfg, tmp_cfg)
        add_decoder_config(cfg, tmp_cfg)
        add_predictor_config(cfg, tmp_cfg)

    @abstractmethod
    def get_extended_attention_mask(self, batched_inputs):
        pass

    def forward(self, batched_inputs, use_beam_search=None, output_sents=False):
        if use_beam_search is None:
            return self._forward(batched_inputs)
        elif use_beam_search == False or self.beam_searcher.beam_size == 1:
            return self.greedy_decode(batched_inputs, output_sents)
        else:
            return self.decode_beam_search(batched_inputs, output_sents)

    @abstractmethod
    def _forward(self, batched_inputs):
        pass

    def preprocess_batch(self, batched_inputs):
        vfeats = [x[kfg.ATT_FEATS] for x in batched_inputs]
        vfeats, vmasks = pad_tensor(vfeats, padding_value=0, use_mask=True)
        ret = { kfg.ATT_FEATS: vfeats, kfg.ATT_MASKS: vmasks }

        if kfg.ATT_FEATS_WO_MASK in batched_inputs[0]:
            vfeats_wo_mask = [x[kfg.ATT_FEATS_WO_MASK] for x in batched_inputs]
            vfeats_wo_mask = pad_tensor(vfeats_wo_mask, padding_value=0, use_mask=False)
            ret.update( { kfg.ATT_FEATS_WO_MASK: vfeats_wo_mask } )

        if kfg.RELATION in batched_inputs[0]:
            relation = [x[kfg.RELATION] for x in batched_inputs]
            relation = pad_tensor(relation, padding_value=0, use_mask=False) # GCN-LSTM, only support 36 features
            ret.update( { kfg.RELATION: relation } )

        if kfg.ATTRIBUTE in batched_inputs[0]:
            attributes = [x[kfg.ATTRIBUTE] for x in batched_inputs]
            attributes = pad_tensor(attributes, padding_value=0, use_mask=False) # LSTM-A
            ret.update( { kfg.ATTRIBUTE: attributes } )

        if kfg.GLOBAL_FEATS in batched_inputs[0]:
            gv_feats = [x[kfg.GLOBAL_FEATS] for x in batched_inputs]
            gv_feats = pad_tensor(gv_feats, padding_value=0, use_mask=False) 
            ret.update( { kfg.GLOBAL_FEATS: gv_feats } )

        if kfg.U_TOKENS_IDS in batched_inputs[0]:
            u_tokens_ids = [x[kfg.U_TOKENS_IDS] for x in batched_inputs]
            u_tokens_ids, tmasks = pad_tensor(u_tokens_ids, padding_value=0, use_mask=True)
            ret.update( { kfg.U_TOKENS_IDS: u_tokens_ids, kfg.TOKENS_MASKS: tmasks} )

        if kfg.U_TOKENS_IDS_WO_MASK in batched_inputs[0]:
            u_tokens_ids_wo_mask = [x[kfg.U_TOKENS_IDS_WO_MASK] for x in batched_inputs]
            u_tokens_ids_wo_mask = pad_tensor(u_tokens_ids_wo_mask, padding_value=0, use_mask=False)
            ret.update( { kfg.U_TOKENS_IDS_WO_MASK: u_tokens_ids_wo_mask } )

        if kfg.G_TOKENS_IDS in batched_inputs[0]:
            g_tokens_ids = [x[kfg.G_TOKENS_IDS] for x in batched_inputs]
            g_tokens_ids, tmasks = pad_tensor(g_tokens_ids, padding_value=0, use_mask=True)
            ret.update( { kfg.G_TOKENS_IDS: g_tokens_ids, kfg.TOKENS_MASKS: tmasks} )

        if kfg.U_TARGET_IDS in batched_inputs[0]:
            u_target_ids = [x[kfg.U_TARGET_IDS] for x in batched_inputs]
            u_target_ids = pad_tensor(u_target_ids, padding_value=-1, use_mask=False)
            ret.update({ kfg.U_TARGET_IDS: u_target_ids })

        if kfg.G_TARGET_IDS in batched_inputs[0]:
            g_target_ids = [x[kfg.G_TARGET_IDS] for x in batched_inputs]
            g_target_ids = pad_tensor(g_target_ids, padding_value=-1, use_mask=False)
            ret.update({ kfg.G_TARGET_IDS: g_target_ids })

        if kfg.ATT_FEATS_LOC in batched_inputs[0]:
            vfeats_loc = [x[kfg.ATT_FEATS_LOC] for x in batched_inputs]
            vfeats_loc = pad_tensor(vfeats_loc, padding_value=0, use_mask=False)
            ret.update({ kfg.ATT_FEATS_LOC: vfeats_loc })

        if kfg.U_TOKENS_TYPE in batched_inputs[0]:
            u_tokens_type = [x[kfg.U_TOKENS_TYPE] for x in batched_inputs]
            u_tokens_type = pad_tensor(u_tokens_type, padding_value=0, use_mask=False)
            ret.update({ kfg.U_TOKENS_TYPE: u_tokens_type })

        if kfg.G_TOKENS_TYPE in batched_inputs[0]:
            g_tokens_type = [x[kfg.G_TOKENS_TYPE] for x in batched_inputs]
            g_tokens_type = pad_tensor(g_tokens_type, padding_value=1, use_mask=False)
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })

        if kfg.V_TARGET in batched_inputs[0]:
            v_target = [x[kfg.V_TARGET] for x in batched_inputs]
            v_target = pad_tensor(v_target, padding_value=0, use_mask=False)
            ret.update({ kfg.V_TARGET: v_target })

        if kfg.V_TARGET_LABELS in batched_inputs[0]:
            v_target_labels = [x[kfg.V_TARGET_LABELS] for x in batched_inputs]
            v_target_labels = pad_tensor(v_target_labels, padding_value=-1, use_mask=False)
            ret.update({ kfg.V_TARGET_LABELS: v_target_labels })

        if kfg.ITM_NEG_LABEL in batched_inputs[0]:
            itm_neg_labels = [x[kfg.ITM_NEG_LABEL] for x in batched_inputs]
            itm_neg_labels = torch.stack(itm_neg_labels, dim=0)
            ret.update({ kfg.ITM_NEG_LABEL: itm_neg_labels })

        if kfg.SEQ_PER_SAMPLE in batched_inputs[0]:
            batch_size, max_feats_num, feats_dim = vfeats.size()[0:3]
            repeat_num = batched_inputs[0][kfg.SEQ_PER_SAMPLE].item()

            vfeats = vfeats.unsqueeze(1).expand(batch_size, repeat_num, max_feats_num, feats_dim)
            vfeats = vfeats.reshape(-1, max_feats_num, feats_dim)
            vmasks = vmasks.unsqueeze(1).expand(batch_size, repeat_num, max_feats_num)
            vmasks = vmasks.reshape(-1, max_feats_num)
            ret.update({ kfg.ATT_FEATS: vfeats, kfg.ATT_MASKS: vmasks })

            if kfg.ATT_FEATS_LOC in batched_inputs[0]:
                vfeats_loc_dim = vfeats_loc.size(-1)
                vfeats_loc = vfeats_loc.unsqueeze(1).expand(batch_size, repeat_num, max_feats_num, vfeats_loc_dim)
                vfeats_loc = vfeats_loc.reshape(-1, max_feats_num, vfeats_loc_dim)
                ret.update({ kfg.ATT_FEATS_LOC: vfeats_loc })

            if kfg.RELATION in batched_inputs[0]:
                relation = relation.unsqueeze(1).expand(batch_size, repeat_num, max_feats_num, max_feats_num)
                relation = relation.reshape(-1, max_feats_num, max_feats_num)
                ret.update({ kfg.RELATION: relation })

            if kfg.ATTRIBUTE in batched_inputs[0]:
                attribute_dim = attributes.size(-1)
                attributes = attributes.unsqueeze(1).expand(batch_size, repeat_num, attribute_dim)
                attributes = attributes.reshape(-1, attribute_dim)
                ret.update({ kfg.ATTRIBUTE: attributes })

            if kfg.GLOBAL_FEATS in batched_inputs[0]:
                gv_feat_dim = gv_feats.size(-1)
                gv_feats = gv_feats.unsqueeze(1).expand(batch_size, repeat_num, gv_feat_dim)
                gv_feats = gv_feats.reshape(-1, gv_feat_dim)
                ret.update({ kfg.GLOBAL_FEATS: gv_feats })

        dict_to_cuda(ret)
        if kfg.IDS in batched_inputs[0]:
            ids = [x[kfg.IDS]  for x in batched_inputs ]
            if kfg.SEQ_PER_SAMPLE in batched_inputs[0]:
                ids = np.repeat(np.expand_dims(ids, axis=1), repeat_num, axis=1).flatten()
            ret.update({ kfg.IDS: ids })
        return ret

    def greedy_decode(self, batched_inputs, output_sents=False):
        return self.greedy_decoder(
            batched_inputs, 
            output_sents,
            model=weakref.proxy(self)
        )

    def decode_beam_search(self, batched_inputs, output_sents=False):
        return self.beam_searcher(
            batched_inputs, 
            output_sents,
            model=weakref.proxy(self)
        )
