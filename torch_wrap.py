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

    all_params=[_tp.all_tensors[key] for key in _tp.all_tensors if _tp.all_tensors[key].requires_grad]

    for _ in range(_tp.meta_epoch):
        loss = _tp.meta_loss()(_tp.all_tensors['_lval'], _tp.all_tensors['a']*_tp.all_tensors['x'])
        optim = _tp.meta_optimizer(all_params,lr=0.1)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return all_params


v=SimpleExample(tp)
print(v)