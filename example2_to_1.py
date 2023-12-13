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
def SimpleExample(_tp:TorchPlus):
    #all terminals
    _tp.all_tensors['___con_1'] = torch.FloatTensor([2.])
    _tp.all_tensors['___con_2'] = torch.FloatTensor([10.])
    _tp.all_tensors['___par_1'] = torch.rand(_tp.all_tensors['___con_1'].shape[-1], requires_grad=True)
    
    for _ in range(_tp.meta_optimizer_epoch):
        #process
        pred = _tp.all_tensors['___par_1'] * _tp.all_tensors['___con_1']
        label = _tp.all_tensors['___con_2']

        #train
        loss = _tp.meta_error_measurement()(label,  pred)
        optim = _tp.meta_optimizer(_tp.get_all_params(),lr=_tp.meta_optimizer_learning_rate)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return _tp.get_all_params()


v=SimpleExample(tp)
print(v)