import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from .decode_strategy import DecodeStrategy
from .build import DECODE_STRATEGY_REGISTRY

@DECODE_STRATEGY_REGISTRY.register()
class GreedyDecoder(DecodeStrategy):

    def _forward(self, batched_inputs, model):
        is_sample = batched_inputs.get(kfg.DECODE_BY_SAMPLE, False)

        inputs = batched_inputs
        masks = model.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = model.visual_embed(batched_inputs)
        inputs.update(ve_out)

        encoder_out_v = model.encoder(inputs, mode='v')
        inputs.update(encoder_out_v)
        inputs = model.decoder.preprocess(inputs)

        batch_size = inputs[kfg.ATT_FEATS].size(0)
        sents = Variable(torch.zeros((batch_size, self.max_seq_len), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, self.max_seq_len).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        g_tokens_type = batched_inputs[kfg.G_TOKENS_TYPE]
        unfinished = wt.eq(wt)

        for t in range(self.max_seq_len):
            inputs.update({ kfg.G_TOKENS_IDS: wt, kfg.G_TOKENS_TYPE: g_tokens_type[:,t] })

            te_out = model.token_embed(inputs)
            inputs.update(te_out)

            encoder_out_t = model.encoder(inputs, mode='t')
            inputs.update(encoder_out_t)

            logit = model.predictor(inputs)[kfg.G_LOGITS]
            logprobs_t = F.log_softmax(logit, dim=1)

            if is_sample:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            else:
                logP_t, wt = torch.max(logprobs_t, 1)

            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break

        return {
            kfg.IDS: batched_inputs[kfg.IDS],
            kfg.G_SENTS_IDS: sents,
            kfg.G_LOGP: logprobs
        }