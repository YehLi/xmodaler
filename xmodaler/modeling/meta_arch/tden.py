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
from ..predictor import build_v_predictor, build_predictor_with_name, add_predictor_config
from ..predictor import MultiModalSimilarity
from .transformer_enc_dec import TransformerEncoderDecoder
from .build import META_ARCH_REGISTRY

__all__ = ["TDENBiTransformer", "TDENPretrain"]

@META_ARCH_REGISTRY.register()
class TDENBiTransformer(TransformerEncoderDecoder):
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
            beam_searcher=beam_searcher,
            v_predictor=v_predictor
        )

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({ "v_predictor": None })
        return ret

    def get_extended_attention_mask(self, batched_inputs):
        if kfg.TOKENS_MASKS not in batched_inputs:
            batched_inputs[kfg.TOKENS_MASKS] = torch.ones((batched_inputs[kfg.ATT_MASKS].size(0), self.max_seq_len)).cuda()

        tmasks = batched_inputs[kfg.TOKENS_MASKS]
        tmasks = tmasks.to(dtype=next(self.parameters()).dtype)
        ext_u_tmasks = tmasks.unsqueeze(1).unsqueeze(2)
        ext_u_tmasks = (1.0 - ext_u_tmasks) * -10000.0
        ext_g_tmasks = ext_u_tmasks

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

@META_ARCH_REGISTRY.register()
class TDENRetrieval(TransformerEncoderDecoder):
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
            beam_searcher=beam_searcher,
            v_predictor=v_predictor
        )
        self.similarity_predictor = predictor
        self.predictor = None

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({ "v_predictor": None })
        return ret

    def _forward(self, batched_inputs):
        inputs = super()._forward(batched_inputs)
        scores = self.similarity_predictor(inputs)
        inputs.update(scores)
        return inputs


@META_ARCH_REGISTRY.register()
class TDENPretrain(TransformerEncoderDecoder):
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
        similarity_predictor
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
        self.similarity_predictor = similarity_predictor

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        if cfg.MODEL.BERT.V_TARGET_SIZE > 0:
            v_predictor = build_v_predictor(cfg)
        else:
            v_predictor = None

        similarity_predictor = build_predictor_with_name(cfg, "MultiModalSimilarity")
        
        ret.update({ "v_predictor": v_predictor, "similarity_predictor": similarity_predictor })
        return ret

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        super().add_config(cfg, tmp_cfg)
        MultiModalSimilarity.add_config(cfg)

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

        if inputs["tden_pretrain_similarity"] == True:
            scores = self.similarity_predictor(inputs)
            inputs.update(scores)
        
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

