#model SimpleExample:
#    21 ?= ? * $2

import torch

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()
def assign_values(self:TorchPlus):
    self.all_leaf_tensors[0]  = TorchTensorPlus(ttype=TTPType.INPUT)
    self.all_leaf_tensors[0].tensor = torch.FloatTensor([2])
    self.all_leaf_tensors[1]  = TorchTensorPlus(ttype=TTPType.PARAMETER)
    self.all_leaf_tensors[1].tensor = torch.FloatTensor(1)
    self.all_leaf_tensors[2]  = TorchTensorPlus(ttype=TTPType.DEFAULT)
    self.all_leaf_tensors[2].tensor = torch.FloatTensor([21])
tp.assign_leaf_tensors=assign_values
def assign_process(self:TorchPlus):
    #.......
    #
    #self._pred = self.all_leaf_tensors[1] * self.all_leaf_tensors[0]
    #self._label = self.all_leaf_tensors[2]
    self._pred = self.all_leaf_tensors[1].tensor * self.all_leaf_tensors[0].tensor
    self._label = self.all_leaf_tensors[2].tensor
tp.assign_process_process = assign_process


v=tp.train()
print(v)