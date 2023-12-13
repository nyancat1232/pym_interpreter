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
def SimpleExample(_tp:TorchPlus):
    #all terminals
    _tp.all_tensors['___con_1'] = torch.ones(4)
    _tp.all_tensors['___con_2'] = torch.zeros(3)
    _tp.all_tensors['___par_1'] = torch.rand(_tp.all_tensors['___con_1'].shape[-1], _tp.all_tensors['___con_2'].shape[0], requires_grad=True)
    _tp.all_tensors['___par_2'] = torch.rand(_tp.all_tensors['___par_1'].shape[-1], requires_grad=True)


    
    for _ in range(_tp.meta_optimizer_epoch):
        #process
        pred = torch.matmul(_tp.all_tensors['___con_1'], _tp.all_tensors['___par_1']) + _tp.all_tensors['___par_2']
        label = _tp.all_tensors['___con_2']

        #train
        _tp.train_one_step_by_equation(label,pred)
    return _tp.get_all_params()


v=SimpleExample(tp)
print(v)