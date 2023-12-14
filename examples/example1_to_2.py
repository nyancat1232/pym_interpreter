#!optimizer SGD(lr=0.01)
#!optimizer.epoch 300
#!optimizer.learning_rate 0.01
#!loss MSE
#model SGDTest:
#   x = torch.ones(4)  # input tensor
#   y = torch.zeros(3)  # expected output
#   y ?= torch.matmul(x, ?) + ?

#compile result

import torch
from pyplus.pytorch.simple import TorchPlus

tp = TorchPlus()
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_epoch = 300
tp.meta_optimizer_learning_rate = 0.01
tp.meta_error_measurement = torch.nn.MSELoss
def assign_values(self):
    self.all_tensors['___con_1'] = torch.ones(4)
    self.all_tensors['___con_2'] = torch.zeros(3)
    self.all_tensors['___par_1'] = torch.rand(self.all_tensors['___con_1'].shape[-1], self.all_tensors['___con_2'].shape[0], requires_grad=True)
    self.all_tensors['___par_2'] = torch.rand(self.all_tensors['___par_1'].shape[-1], requires_grad=True)
tp.assign_process_values=assign_values
def assign_process(self):
    self._pred = torch.matmul(self.all_tensors['___con_1'], self.all_tensors['___par_1']) + self.all_tensors['___par_2']
    self._label = self.all_tensors['___con_2']
tp.assign_process_process = assign_process


v=tp.train()
print(v)