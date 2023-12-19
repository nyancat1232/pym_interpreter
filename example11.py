#!sequence
#model SimpleExampleSequence:
#    [10,20,30,40] ?= ?1 * $[2,4,6,8]

import torch
import torch.nn as nn
from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus,TensorsSquence

tp = TorchPlus()


ttp2 = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)

#assign leaf tensors
tp._all_predict_tensors.new_tensor('input',TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0),torch.FloatTensor([2,4,6,8]))
tp._all_predict_tensors.new_tensor('param',TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1),torch.FloatTensor(1))
tp._all_label_tensors.new_tensor('label',TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0),torch.FloatTensor([10,20,30,40]))

def assign_process(tensors_current_sequence,current_activator):
    proc = tensors_current_sequence['param']*tensors_current_sequence['input']

    return proc
tp.assign_process_prediction = assign_process


print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([1,2,3,4])}))
