import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .base_enc_dec import BaseEncoderDecoder
from .build import META_ARCH_REGISTRY

__all__ = ["RnnAttEncoderDecoder"]

@META_ARCH_REGISTRY.register()
class RnnAttEncoderDecoder(BaseEncoderDecoder):

    def get_extended_attention_mask(self, att_masks):
        if att_masks is not None:
            att_masks = att_masks.to(dtype=next(self.parameters()).dtype)
            ext_att_masks = (1.0 - att_masks) * -10000.0
        else:
            ext_att_masks = None
            
        return {
            kfg.ATT_MASKS: att_masks,
            kfg.EXT_ATT_MASKS: ext_att_masks
        }

    def forward(self, batched_inputs):
        batched_inputs = self.preprocess_batch(batched_inputs)
        tokens_ids = batched_inputs[kfg.TOKENS_IDS]
        att_feats = batched_inputs[kfg.ATT_FEATS]
        att_masks = batched_inputs[kfg.ATT_MASKS]

        att_feats = self.visual_embed(att_feats)
        inputs = self.get_extended_attention_mask(att_masks)
        inputs.update( { kfg.ATT_FEATS: att_feats } )
        
        encoder_out = self.encoder(inputs)
        inputs.update(encoder_out)
        inputs = self.decoder.preprocess(inputs)
        
        batch_size, seq_len = tokens_ids.shape
        outputs = Variable(torch.zeros(batch_size, seq_len, self.predictor.vocab_size).cuda())
        
        for t in range(seq_len):
            if t >= 1 and tokens_ids[:, t].max() == 0:
                break
            
            wt = tokens_ids[:, t].clone()
            token_embed = self.token_embed(wt)
            inputs.update({ kfg.TOKEN_EMBED: token_embed })
            decoder_out = self.decoder(inputs)
            inputs.update(decoder_out)

            logit = self.predictor(inputs)
            outputs[:, t] = logit
            
        return { 
            kfg.LOGITS: outputs,
            kfg.TARGET_IDS: batched_inputs[kfg.TARGET_IDS]
        }

    def decode(self, cfg, batched_inputs):
        batched_inputs = self.preprocess_batch_test(batched_inputs)
        att_feats = batched_inputs[kfg.ATT_FEATS]
        att_masks = batched_inputs[kfg.ATT_MASKS]

        att_feats = self.visual_embed(att_feats)
        inputs = self.get_extended_attention_mask(att_masks)
        inputs.update( { kfg.ATT_FEATS: att_feats } )

        encoder_out = self.encoder(inputs)
        inputs.update(encoder_out)
        inputs = self.decoder.preprocess(inputs)

        batch_size = att_feats.size(0)
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.MAX_SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.MAX_SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)

        for t in range(cfg.MODEL.MAX_SEQ_LEN):
            token_embed = self.token_embed(wt)
            inputs.update({ kfg.TOKEN_EMBED: token_embed })
            decoder_out = self.decoder(inputs)
            inputs.update(decoder_out)

            logit = self.predictor(inputs)
            logprobs_t = F.log_softmax(logit, dim=1)
            logP_t, wt = torch.max(logprobs_t, 1)

            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        return sents, logprobs









