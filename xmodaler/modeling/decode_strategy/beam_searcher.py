import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import expand_tensor
from .decode_strategy import DecodeStrategy
from .build import DECODE_STRATEGY_REGISTRY

@DECODE_STRATEGY_REGISTRY.register()
class BeamSearcher(DecodeStrategy):

    def _select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def _expand_state(self, states, selected_beam, batch_size, beam_size, cur_beam_size):
        for i in range(len(states)):
            shape = list(states[i].shape)
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            states[i] = torch.gather(states[i].view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                beam.expand(*([batch_size, beam_size] + shape[1:])))
            states[i] = states[i].view(*([-1, ] + shape[1:]))


    def _forward(self, batched_inputs, model):
        batch_size = batched_inputs[kfg.ATT_FEATS].size(0)
        beam_size = self.beam_size
        log_probs = []
        selected_words = None
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        g_tokens_type = batched_inputs[kfg.G_TOKENS_TYPE]

        inputs = batched_inputs
        masks = model.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = model.visual_embed(batched_inputs)
        inputs.update(ve_out)

        encoder_out_v = model.encoder(inputs, mode='v')
        inputs.update(encoder_out_v)
        inputs = model.decoder.preprocess(inputs)
        
        outputs = []
        for t in range(self.max_seq_len):
            cur_beam_size = 1 if t == 0 else beam_size

            inputs.update({ kfg.G_TOKENS_IDS: wt, kfg.G_TOKENS_TYPE: g_tokens_type[:,t] })
            vt_out = model.token_embed(inputs)
            inputs.update(vt_out)

            encoder_out_t = model.encoder(inputs, mode='t')
            inputs.update(encoder_out_t)

            decoder_out = model.decoder(inputs)
            inputs.update(decoder_out)

            logit = model.predictor(inputs)[kfg.G_LOGITS]
            word_logprob = F.log_softmax(logit, dim=1)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self._select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='floor')
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # hidden states
            if kfg.G_HIDDEN_STATES in inputs:
                states = inputs[kfg.G_HIDDEN_STATES]
                self._expand_state(states, selected_beam, batch_size, beam_size, cur_beam_size)
                inputs.update({ kfg.G_HIDDEN_STATES: states })

            # cells
            if kfg.G_CELL_STATES in inputs:
                cells = inputs[kfg.G_CELL_STATES]
                self._expand_state(cells, selected_beam, batch_size, beam_size, cur_beam_size)
                inputs.update({ kfg.G_CELL_STATES: cells })

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                if kfg.ATT_FEATS in inputs:
                    att_feats = expand_tensor(inputs[kfg.ATT_FEATS], beam_size)
                    inputs.update({ kfg.ATT_FEATS: att_feats })

                if kfg.GLOBAL_FEATS in inputs:
                    gv_feat = expand_tensor(inputs[kfg.GLOBAL_FEATS], beam_size)
                    inputs.update({ kfg.GLOBAL_FEATS: gv_feat })

                if kfg.ATT_MASKS in inputs:
                    att_mask = expand_tensor(inputs[kfg.ATT_MASKS], beam_size)
                    inputs.update({ kfg.ATT_MASKS: att_mask })

                if kfg.EXT_ATT_MASKS in inputs:
                    ext_att_masks = expand_tensor(inputs[kfg.EXT_ATT_MASKS], beam_size)
                    inputs.update({ kfg.EXT_ATT_MASKS: ext_att_masks })

                if kfg.P_ATT_FEATS in inputs:
                    p_att_feats = expand_tensor(inputs[kfg.P_ATT_FEATS], beam_size)
                    inputs.update({ kfg.P_ATT_FEATS: p_att_feats })

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.max_seq_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.max_seq_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]
        
        return {
            kfg.IDS: batched_inputs[kfg.IDS],
            kfg.G_SENTS_IDS: outputs,
            kfg.G_LOGP: log_probs
        }