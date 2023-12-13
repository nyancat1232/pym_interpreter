#model SimpleExample:
#    10 ?= ? * 2

#compile result

import torch

def SimpleExample():
    meta_loss = torch.nn.MSELoss
    meta_optimizer = torch.optim.SGD
    meta_epoch = 10

    all_tensors={}
    all_tensors['x'] = torch.FloatTensor([2])
    all_tensors['a'] = torch.FloatTensor(1)
    all_tensors['a'].requires_grad = True
    all_tensors['_lval'] = torch.FloatTensor([10])

    all_params=[all_tensors[key] for key in all_tensors if all_tensors[key].requires_grad]

    for _ in range(meta_epoch):
        loss = meta_loss()(all_tensors['_lval'], all_tensors['a']*all_tensors['x'])
        optim = meta_optimizer(all_params,lr=0.1)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return all_params


v=SimpleExample()
print(v)