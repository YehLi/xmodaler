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


__all__ = ["VLPBertPreTraining"]

@META_ARCH_REGISTRY.register()
class VLPBertPreTraining(BaseEncoderDecoder):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        token_embed,
        visual_embed,
        encoder,
        decoder,
        predictor,
        v_predictor,
    ):
        super().__init__(
            vocab_size=vocab_size,
            token_embed=token_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
        )
        self.v_predictor = v_predictor

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        if cfg.MODEL.BERT.V_TARGET_SIZE > 0:
            ret.update({ "v_predictor": build_v_predictor(cfg, ) })

        return ret

    def get_extended_attention_mask(self, batched_inputs):
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

    def forward(self, batched_inputs):
        batched_inputs = self.preprocess_batch(batched_inputs)
        
        masks = self.get_extended_attention_mask(batched_inputs)
        inputs = masks

        vfeats = batched_inputs[kfg.ATT_FEATS]
        vfeats_loc = batched_inputs[kfg.ATT_FEATS_LOC]
        vfeats = self.visual_embed(vfeats, vfeats_loc)
        inputs.update({ kfg.ATT_FEATS: vfeats })

        if kfg.U_TOKENS_TYPE in batched_inputs:
            u_tokens_ids = batched_inputs[kfg.U_TOKENS_IDS]
            u_tokens_type = batched_inputs[kfg.U_TOKENS_TYPE]
            u_token_embed = self.token_embed(u_tokens_ids, token_type_ids=u_tokens_type)
            inputs.update({ kfg.U_TOKEN_EMBED: u_token_embed })

        if kfg.G_TOKENS_TYPE in batched_inputs:
            g_tokens_ids = batched_inputs[kfg.G_TOKENS_IDS]
            g_tokens_type = batched_inputs[kfg.G_TOKENS_TYPE]
            g_token_embed = self.token_embed(g_tokens_ids, token_type_ids=g_tokens_type)
            inputs.update({ kfg.G_TOKEN_EMBED: g_token_embed })
        
        encoder_out = self.encoder(inputs)
        inputs.update(encoder_out)

        decoder_out = self.decoder(inputs)
        inputs.update(decoder_out)


    def preprocess_batch(self, batched_inputs):
        vfeats = [x[kfg.ATT_FEATS] for x in batched_inputs]
        vfeats, vmasks = pad_tensor(vfeats, padding_value=0, use_mask=True)
        ret = { kfg.ATT_FEATS: vfeats, kfg.ATT_MASKS: vmasks }

        if kfg.U_TOKENS_IDS in batched_inputs[0]:
            u_tokens_ids = [x[kfg.U_TOKENS_IDS] for x in batched_inputs]
            u_tokens_ids, tmasks = pad_tensor(u_tokens_ids, padding_value=0, use_mask=True)
            ret.update( { kfg.U_TOKENS_IDS: u_tokens_ids, kfg.TOKENS_MASKS: tmasks} )

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

        dict_to_cuda(ret)
        return ret
