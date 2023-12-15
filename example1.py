#model SimpleExample:
#    20 ?= ? * $2

import torch

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()
def assign_values(self:TorchPlus):
    #elf.all_leaf_tensors['input0']  = TorchTensorPlus(ttype=TTPType.INPUT)
    #self.all_leaf_tensors['input0'].tensor = torch.FloatTensor([2])
    self.all_leaf_tensors['input0']  = TorchTensorPlus(ttype=TTPType.INPUT)
    self.all_leaf_tensors['input0'].tensor = torch.FloatTensor([2])
    self.all_leaf_tensors[1]  = TorchTensorPlus(ttype=TTPType.PARAMETER)
    self.all_leaf_tensors[1].tensor = torch.FloatTensor(1)
    self.all_leaf_tensors[2]  = TorchTensorPlus(ttype=TTPType.DEFAULT)
    self.all_leaf_tensors[2].tensor = torch.FloatTensor([20])
tp.assign_leaf_tensors=assign_values
def assign_process(self:TorchPlus,current_sequence:int):
    #.......
    #
    #self._pred = self.all_leaf_tensors[1] * self.all_leaf_tensors[0]
    #self._label = self.all_leaf_tensors[2]
    self._pred = self.all_leaf_tensors[1][current_sequence] * self.all_leaf_tensors['input0'].tensor
    self._label = self.all_leaf_tensors[2][current_sequence]
tp.assign_process_process = assign_process



print(tp.train())
print(tp.predict(**{'input0':torch.FloatTensor([12])}))
print(tp.predict(**{'input0':torch.FloatTensor([10])}))
print(tp.predict(**{'input0':torch.FloatTensor([13])}))
print(tp.predict(**{'input0':torch.FloatTensor([14])}))
print(tp.predict(**{'input0':torch.FloatTensor([12])}))