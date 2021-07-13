# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from torch.autograd import Variable
from xmodaler.config import kfg

__all__ = ["Decoder"]

class Decoder(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    def preprocess(self, batched_inputs):
        return batched_inputs

    def init_states(self, batch_size):
        return {
            kfg.G_HIDDEN_STATES: [Variable(torch.zeros(batch_size, self.hidden_size).cuda()) for _ in range(self.num_layers)],
            kfg.G_CELL_STATES: [Variable(torch.zeros(batch_size, self.hidden_size).cuda()) for _ in range(self.num_layers)]
        }
