#model SimpleExample:
#    20 ?= ? * $2

import torch
import torch.nn as nn

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()
tp.meta_activator = nn.ReLU
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_epoch = 300
tp.meta_optimizer_learning_rate = 0.015
tp.meta_error_measurement = torch.nn.MSELoss
def assign_values(self:TorchPlus):
    self.all_leaf_tensors['input']  = TorchTensorPlus(ttype=TTPType.INPUT)
    self.all_leaf_tensors['input'].tensor = torch.FloatTensor([2.])
    self.all_leaf_tensors['___p']  = TorchTensorPlus(ttype=TTPType.PARAMETER)
    self.all_leaf_tensors['___p'].tensor = torch.FloatTensor(1)
    self.all_leaf_tensors['output']  = TorchTensorPlus(ttype=TTPType.DEFAULT)
    self.all_leaf_tensors['output'].tensor = torch.FloatTensor([20.])
tp.assign_leaf_tensors=assign_values
def assign_process(self:TorchPlus,current_sequence:int):
    proc = self.all_leaf_tensors['input'][-1] * self.all_leaf_tensors['___p'].tensor

    self._pred = proc
    self._label = self.all_leaf_tensors['output'][-1]
tp.assign_process_process = assign_process

print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([12])}))