#model SimpleExample:
#    10 ?= ? * 2

#compile result

import torch
from pyplus.pytorch.simple import TorchPlus

tp = TorchPlus()
tp.meta_loss = torch.nn.MSELoss
tp.meta_optimizer = torch.optim.SGD
tp.meta_epoch = 10
def SimpleExample(_tp:TorchPlus):
    _tp.all_tensors['x'] = torch.FloatTensor([2])
    _tp.all_tensors['a'] = torch.FloatTensor(1)
    _tp.all_tensors['a'].requires_grad = True
    _tp.all_tensors['_lval'] = torch.FloatTensor([10])

    for _ in range(_tp.meta_epoch):
        loss = _tp.meta_loss()(_tp.all_tensors['_lval'], _tp.all_tensors['a']*_tp.all_tensors['x'])
        optim = _tp.meta_optimizer(_tp.get_all_params(),lr=0.1)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return _tp.get_all_params()


v=SimpleExample(tp)
print(v)