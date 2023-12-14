#model SimpleExample:
#    10 ?= ? * 2

import torch

from pyplus.pytorch.simple import TorchPlus

tp = TorchPlus()
def assign_values(self:TorchPlus):
    self.all_leaf_tensors[0]  = torch.FloatTensor([2])
    self.all_leaf_tensors[0].requires_grad=False
    self.all_leaf_tensors[1]  = torch.FloatTensor(1)
    self.all_leaf_tensors[1].requires_grad=True
    self.all_leaf_tensors[2]  = torch.FloatTensor([10])
    self.all_leaf_tensors[2].requires_grad=False
tp.assign_leaf_tensors=assign_values
def assign_process(self:TorchPlus):
    self._pred = self.all_leaf_tensors[1] * self.all_leaf_tensors[0]
    self._label = self.all_leaf_tensors[2]
tp.assign_process_process = assign_process


v=tp.train()
print(v)