#model SimpleExample:
#    s[18,36] ?= ? * $s[2,4]

import torch
import torch.nn as nn

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlusInternal

tp = TorchPlus()

#assign leaf tensors
def assign_process(current_activator):
    proc = tp.input('input',torch.FloatTensor([2.,4.])) * tp.parameter('param',torch.FloatTensor(1))

    tp.label('label',torch.FloatTensor([18.,36.]))
    return proc
tp.assign_process_prediction = assign_process


result = tp.train()
print(result(input=torch.FloatTensor([10,21,12])))
