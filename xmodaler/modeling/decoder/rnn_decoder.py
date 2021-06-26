from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from torch.autograd import Variable
from xmodaler.config import kfg

__all__ = ["RnnDecoder"]

class RnnDecoder(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    def init_states(self, batch_size):
        return {
            kfg.HIDDEN_STATES: Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),
            kfg.CELL_STATES: Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        }
