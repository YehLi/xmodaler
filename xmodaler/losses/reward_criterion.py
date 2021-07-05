import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class RewardCriterion(nn.Module):
    @configurable
    def __init__(self, eos_id):
        super(RewardCriterion, self).__init__()
        self.eos_id = eos_id

    @classmethod
    def from_config(cls, cfg):
        return {
            'eos_id': cfg.SCORER.EOS_ID
        }

    def forward(self, outputs_dict):
        seq = outputs_dict[kfg.G_SENTS_IDS]
        logP = outputs_dict[kfg.G_LOGP]
        rewards = outputs_dict[kfg.REWARDS]

        mask = (seq > 0) & (seq != self.eos_id)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)

        return { 'RewardCriterion': loss }