# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import pad_tensor, dict_to_cuda
from ..predictor import build_v_predictor
from .base_enc_dec import BaseEncoderDecoder
from .build import META_ARCH_REGISTRY

__all__ = ["TransformerEncoderDecoder"]

@META_ARCH_REGISTRY.register()
class TransformerEncoderDecoder(BaseEncoderDecoder):
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
            beam_searcher=beam_searcher
        )
        self.v_predictor = v_predictor

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        if cfg.MODEL.BERT.V_TARGET_SIZE > 0:
            v_predictor = build_v_predictor(cfg)
        else:
            v_predictor = None
        
        ret.update({ "v_predictor": v_predictor })
        return ret

    def get_extended_attention_mask(self, batched_inputs):
        if kfg.TOKENS_MASKS not in batched_inputs:
            batched_inputs[kfg.TOKENS_MASKS] = torch.ones((batched_inputs[kfg.ATT_MASKS].size(0), self.max_seq_len)).cuda()

        tmasks = batched_inputs[kfg.TOKENS_MASKS]
        seq_length = tmasks.size(-1)
        tmasks = tmasks.to(dtype=next(self.parameters()).dtype)
        ext_u_tmasks = tmasks.unsqueeze(1).unsqueeze(2)
        ext_u_tmasks = (1.0 - ext_u_tmasks) * -10000.0

        ext_g_tmasks = torch.tril(torch.ones(
            (seq_length, seq_length), dtype=tmasks.dtype, device=tmasks.device))
        ext_g_tmasks = ext_g_tmasks.unsqueeze(0).expand(
            (tmasks.size(0), seq_length, seq_length))
        ext_g_tmasks = ext_g_tmasks * tmasks.unsqueeze(1)
        ext_g_tmasks = ext_g_tmasks.to(dtype=next(self.parameters()).dtype)
        ext_g_tmasks = ext_g_tmasks.unsqueeze(1)
        ext_g_tmasks = (1.0 - ext_g_tmasks) * -10000.0

        vmasks = batched_inputs[kfg.ATT_MASKS]
        vmasks = vmasks.to(dtype=next(self.parameters()).dtype)
        vmasks = vmasks.unsqueeze(1).unsqueeze(2)
        ext_vmasks = (1.0 - vmasks) * -10000.0

        return {
            kfg.TOKENS_MASKS: tmasks,
            kfg.EXT_U_TOKENS_MASKS: ext_u_tmasks,
            kfg.EXT_G_TOKENS_MASKS: ext_g_tmasks,
            kfg.ATT_MASKS: vmasks,
            kfg.EXT_ATT_MASKS: ext_vmasks
        }

    def _forward(self, batched_inputs):        
        inputs = batched_inputs
        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = self.visual_embed(batched_inputs)
        inputs.update(ve_out)

        if self.encoder is not None:
            encoder_out_v = self.encoder(inputs, mode='v')
            inputs.update(encoder_out_v)

        if self.decoder is not None:
            inputs = self.decoder.preprocess(inputs)

        te_out = self.token_embed(batched_inputs)
        inputs.update(te_out)
        
        if self.encoder is not None:
            encoder_out_t = self.encoder(inputs, mode='t')
            inputs.update(encoder_out_t)
        
        if self.decoder is not None:
            decoder_out = self.decoder(inputs)
            inputs.update(decoder_out)

        if self.predictor is not None:
            tlogits = self.predictor(inputs)
            inputs.update(tlogits)

        if self.v_predictor is not None:
            vlogits = self.v_predictor(inputs)
            inputs.update(vlogits)
        return inputs