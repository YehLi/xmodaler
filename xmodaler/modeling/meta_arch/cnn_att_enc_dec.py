import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .base_enc_dec import BaseEncoderDecoder
from .build import META_ARCH_REGISTRY

__all__ = ["CnnEncoderDecoder"]

@META_ARCH_REGISTRY.register()
class CnnEncoderDecoder(BaseEncoderDecoder):

    def get_extended_attention_mask(self, att_masks):
        if att_masks is not None:
            att_masks = att_masks.to(dtype=next(self.parameters()).dtype)
            ext_att_masks = (1.0 - att_masks) * -10000.0
        else:
            att_masks = None
            ext_att_masks = None
        return {
            kfg.ATT_MASKS: att_masks,
            kfg.EXT_ATT_MASKS: ext_att_masks
        }

    def get_decoder_masks(self, input):
        self.padding_idx = -1

        b_s, seq_len = input.shape[:2]

        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        return mask_self_attention, mask_queries

    def forward(self, batched_inputs):
        batched_inputs = self.preprocess_batch(batched_inputs)
        tokens_ids = batched_inputs[kfg.TOKENS_IDS]
        att_feats = batched_inputs[kfg.ATT_FEATS]
        att_masks = batched_inputs[kfg.ATT_MASKS]

        att_feats = self.visual_embed(att_feats)
        inputs = self.get_extended_attention_mask(att_masks)
        inputs.update( { kfg.ATT_FEATS: att_feats } )
        
        encoder_out = self.encoder(inputs)
        
        inputs = self.decoder.preprocess(inputs)
        inputs.update(encoder_out)
        mask_self_attention, mask_queries = self.get_decoder_masks(tokens_ids)
        inputs.update({kfg.SELF_ATT_MASKS: mask_self_attention,
                        kfg.SEQ_MASKS: mask_queries})
        token_embed = self.token_embed(tokens_ids)
        inputs.update({ kfg.TOKEN_EMBED: token_embed })

        decoder_out = self.decoder(inputs)

        # batch_size, num_seq, d_model = decoder_out.size()
        lm_inputs = {kfg.HIDDEN_STATES: [decoder_out]}

        outputs = self.predictor(lm_inputs)

        return { 
            kfg.LOGITS: outputs,
            kfg.TARGET_IDS: batched_inputs[kfg.TARGET_IDS]
        }
        

    def get_decoder_masks_test(self, input):
        self.padding_idx = -1

        b_s, seq_len = input.shape[:2]

        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        
        return mask_queries

    def decode(self, cfg, batched_inputs):
        batched_inputs = self.preprocess_batch_test(batched_inputs)
        att_feats = batched_inputs[kfg.ATT_FEATS]
        att_masks = batched_inputs[kfg.ATT_MASKS]

        att_feats = self.visual_embed(att_feats)
        inputs = self.get_extended_attention_mask(att_masks)
        inputs.update( { kfg.ATT_FEATS: att_feats } )

        encoder_out = self.encoder(inputs)
        inputs = self.decoder.preprocess(inputs)
        inputs.update(encoder_out)

        batch_size = att_feats.size(0)
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.MAX_SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.MAX_SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, 1, dtype=torch.long).cuda())
        unfinished = wt.eq(wt) 

        mask_self_attention_constant = torch.triu(torch.ones((cfg.MODEL.MAX_SEQ_LEN, cfg.MODEL.MAX_SEQ_LEN), dtype=torch.uint8, device=wt.device),
                                         diagonal=1)
        mask_self_attention_constant = mask_self_attention_constant.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        # mask_self_attention_constant = mask_self_attention_constant + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention_constant = mask_self_attention_constant
        mask_self_attention_constant = mask_self_attention_constant.gt(0)  # (b_s, 1, seq_len, seq_len)

        for t in range(cfg.MODEL.MAX_SEQ_LEN):
            mask_queries = self.get_decoder_masks_test(sents[:, :t+1])
            token_embed = self.token_embed(sents[:, :t+1])
            inputs.update({ kfg.TOKEN_EMBED: token_embed,
                            kfg.SELF_ATT_MASKS: mask_self_attention_constant[:, :, :t+1, :t+1],
                            kfg.SEQ_MASKS: mask_queries
            })
            decoder_out = self.decoder(inputs)

            lm_inputs = {kfg.HIDDEN_STATES: [decoder_out]}

            logit = self.predictor(lm_inputs)[:,-1,:]
            
            logprobs_t = F.log_softmax(logit, dim=1)
            logP_t, wt = torch.max(logprobs_t, 1)

            wt = wt.view(-1).long().unsqueeze(1)
            unfinished = unfinished * (wt > 0)

            wt = wt * unfinished.type_as(wt)

            sents[:,t] = wt.squeeze()
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        return sents, logprobs









