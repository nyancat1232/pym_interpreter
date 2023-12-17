#!sequence
#model SimpleExampleSequence:
#    [10,20,30,40] ?= ?1 * $[2,4,6,8]

import torch
import torch.nn as nn
from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()

#assign leaf tensors
#tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
#tp['input'].tensor = torch.FloatTensor([[2.],[4.]])
tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
tp['input'].tensor = torch.FloatTensor([2,4,6,8])
tp['param']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)
tp['param'].tensor = torch.FloatTensor(1)
tp['output']  = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
tp['output'].tensor = torch.FloatTensor([10,20,30,40])

def assign_process(tensors_current_sequence,current_activator):
    #proc = tensors_current_sequence['input'] @ tensors_current_sequence['param']
    proc = tensors_current_sequence['param']*tensors_current_sequence['input']

    _pred = proc
    _label = tensors_current_sequence['output']

    return _label,_pred
tp.assign_process_process = assign_process

print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([1,2,3,4])}))