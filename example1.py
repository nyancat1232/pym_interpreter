#model SimpleExample:
#    s[18,36] ?= ? * $s[2,4]

import torch

from pyplus.pytorch.simple import TorchPlus

tp = TorchPlus()

#assign leaf tensors
def assign_process():
    proc = tp.input('input',torch.FloatTensor([2.,4.])) * tp.parameter('param',torch.FloatTensor(1))

    tp.label(torch.FloatTensor([18.,36.]))
    return proc
tp.process = assign_process


result = tp.train()
print(result(input=torch.FloatTensor([10,21,12])))
