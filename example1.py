#model SimpleExample:
#    s[20,40] ?= ? * $s[2,4]

import torch
import torch.nn as nn

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus



tp = TorchPlus()

tp.meta_activator = nn.ReLU
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_epoch = 300
tp.meta_optimizer_learning_rate = 0.015
tp.meta_error_measurement = torch.nn.MSELoss

#assign leaf tensors
tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
tp['input'].tensor = torch.FloatTensor([[2.],[4.]])
tp['___p']  = TorchTensorPlus(ttype=TTPType.PARAMETER)
tp['___p'].tensor = torch.FloatTensor(1)
tp['output']  = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
tp['output'].tensor = torch.FloatTensor([[20.],[40.]])

def assign_process(tensors_current_sequence):
    #input,output => ..[current_sequence], param => ...tensor

    proc = tensors_current_sequence['input'] * tensors_current_sequence['___p']

    _pred = proc
    _label = tensors_current_sequence['output']

    return _label,_pred
tp.assign_process_process = assign_process

print(len(torch.FloatTensor([[10],[11],[12]])))
print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([[10],[11],[12]])}))
