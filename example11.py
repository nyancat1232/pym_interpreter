#!sequence
#model SimpleExampleSequence:
#    [10,20,30,40] ?= ?(1) * $[2,4,6,8]

import torch
import torch.nn as nn
from pyplus.pytorch.simple import TorchPlus

tp = TorchPlus()
tp.meta_data_per_iteration=3
def assign_process():
    proc = tp.parameter('param',torch.FloatTensor(1))*tp.input('input',torch.FloatTensor([2,4,6,8]))

    tp.label(torch.FloatTensor([10,20,30,40]))
    return proc
tp.process = assign_process


result = tp.train(show_every_iteration=True)
print(result(input=torch.FloatTensor([1,2,3,4,1,2,3])))
