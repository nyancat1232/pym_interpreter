import torch

#model SimpleExample:
#    10 ?= ? * 2

#compile result

import torch

meta_loss = torch.nn.MSELoss
meta_optimizer = torch.optim.SGD
meta_epoch = 10

all_tensors=[]
x = torch.FloatTensor([2])
all_tensors.append(x)
a = torch.FloatTensor(1)
a.requires_grad = True
all_tensors.append(a)
y = torch.FloatTensor([10])
all_tensors.append(y)

all_params=[l for l in all_tensors if l.requires_grad]

for _ in range(meta_epoch):
    loss = meta_loss()(y, a*x)
    optim = meta_optimizer(all_params,lr=0.1)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(a)