#!sequence
#model SimpleExampleSequence:
#    [10,20,30,40] ?= ?1 * $[2,4,6,8]

import torch
import torch.nn as nn
from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()


#assign leaf tensors
#tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
#tp['input'].tensor = torch.FloatTensor([2.,4.])
#tp.label_tensor = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
#tp.label_tensor.tensor = torch.FloatTensor([18.,36.])
tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
tp['input'].tensor = torch.FloatTensor([2,4,6,8])
tp['param']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)
tp['param'].tensor = torch.FloatTensor(1)
tp.label_tensor = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
tp.label_tensor.tensor = torch.FloatTensor([10,20,30,40])

def assign_process(tensors_current_sequence,current_activator):
    proc = tensors_current_sequence['param']*tensors_current_sequence['input']

    return proc
tp.assign_process_prediction = assign_process


print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([1,2,3,4])}))
