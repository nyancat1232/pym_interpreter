#!sequence
#model SimpleExampleSequence:
#    [10,20,30,40] ?= ?(1) * $[2,4,6,8]

import torch
import torch.nn as nn
from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlusInternal,TensorsSquence

tp = TorchPlus()


ttp2 = TorchTensorPlusInternal(ttype=TTPType.PARAMETER,axis_sequence=-1)

def assign_process(current_activator):
    proc = tp.parameter('param',torch.FloatTensor(1))*tp.input('input',torch.FloatTensor([2,4,6,8]))

    tp.label('label',torch.FloatTensor([10,20,30,40]))
    return proc
tp.assign_process_prediction = assign_process


result = tp.train()
print(result(input=torch.FloatTensor([1,2,3,4])))
