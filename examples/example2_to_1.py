#model SimpleExample:
#    10 ?= ? * 2

#compile result

import torch
from pyplus.pytorch.simple import TorchPlus

tp = TorchPlus()
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_epoch = 300
tp.meta_optimizer_learning_rate = 0.01
tp.meta_error_measurement = torch.nn.MSELoss
def assign_values(self):
    self.all_tensors['___con_1'] = torch.FloatTensor([10])
    self.all_tensors['___con_2'] = torch.FloatTensor([2])
    self.all_tensors['___par_1'] = torch.rand(1, requires_grad=True)
tp.assign_process_values=assign_values
def assign_process(self):
    self._pred = self.all_tensors['___par_1'] * self.all_tensors['___con_2']
    self._label = self.all_tensors['___con_1']
tp.assign_process_process = assign_process


v=tp.train()
print(v)