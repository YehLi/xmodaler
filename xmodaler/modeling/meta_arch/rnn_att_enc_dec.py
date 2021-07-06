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

    def get_extended_attention_mask(self, batched_inputs):
        att_masks = batched_inputs[kfg.ATT_MASKS]
        if att_masks is not None:
            att_masks = att_masks.to(dtype=next(self.parameters()).dtype)
            ext_att_masks = (1.0 - att_masks) * -10000.0
        else:
            ext_att_masks = None
            
        return {
            kfg.ATT_MASKS: att_masks,
            kfg.EXT_ATT_MASKS: ext_att_masks
        }

    def _forward(self, batched_inputs):
        inputs = batched_inputs
        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = self.visual_embed(batched_inputs)
        inputs.update(ve_out)

        encoder_out_v = self.encoder(inputs, mode='v')
        inputs.update(encoder_out_v)
        inputs = self.decoder.preprocess(inputs)

        tokens_ids = batched_inputs[kfg.G_TOKENS_IDS]
        batch_size, seq_len = tokens_ids.shape
        ss_prob = batched_inputs[kfg.SS_PROB]
        outputs = Variable(torch.zeros(batch_size, seq_len, self.vocab_size).cuda())
        
        for t in range(seq_len):
            if t >= 1 and tokens_ids[:, t].max() == 0:
                break

            if self.training and t >= 1 and ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < ss_prob
                if mask.sum() == 0:
                    wt = tokens_ids[:, t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = tokens_ids[:, t].data.clone()
                    prob_prev = torch.exp(outputs[:, t-1].detach())
                    wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
            else:
                wt = tokens_ids[:, t].clone()
            inputs.update({ kfg.G_TOKENS_IDS: wt })
            
            te_out = self.token_embed(inputs)
            inputs.update(te_out)

            encoder_out_t = self.encoder(inputs, mode='t')
            inputs.update(encoder_out_t)

            decoder_out = self.decoder(inputs)
            inputs.update(decoder_out)

            logit = self.predictor(inputs)[kfg.G_LOGITS]
            outputs[:, t] = logit

        inputs.update({kfg.G_LOGITS: outputs})
        return inputs