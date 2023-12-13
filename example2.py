#!optimizer SGD(lr=0.01)
#!optimizer.epoch 300
#!optimizer.learning_rate 0.01
#!loss MSE
#model SGDTest:
#   x = torch.ones(4)  # input tensor
#   y = torch.zeros(3)  # expected output
#   y ?= torch.matmul(x, ?) + ?


import torch
from pyplus.pytorch.simple import TorchPlus

tp = TorchPlus()
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_epoch = 300
tp.meta_optimizer_learning_rate = 0.01
tp.meta_loss = torch.nn.MSELoss
def SGDTest(_tp:TorchPlus):
    _tp.all_tensors['x'] = torch.ones(4)
    _tp.all_tensors['y'] = torch.zeros(3)
    _tp.all_tensors['___par_1'] = torch.rand(_tp.all_tensors['x'].shape[-1], _tp.all_tensors['y'].shape[0], requires_grad=True)
    _tp.all_tensors['___par_2'] = torch.rand(_tp.all_tensors['___par_1'].shape[-1], requires_grad=True)

    for _ in range(_tp.meta_optimizer_epoch):
        loss = _tp.meta_loss()(_tp.all_tensors['y'], torch.matmul(_tp.all_tensors['x'], _tp.all_tensors['___par_1']) + _tp.all_tensors['___par_2'])
        optim = _tp.meta_optimizer(_tp.get_all_params(),lr=_tp.meta_optimizer_learning_rate)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return _tp.get_all_params()


v=SGDTest(tp)
print(v)