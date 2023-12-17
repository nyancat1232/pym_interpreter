#model SimpleExample:
#    20 ?= ? * $2

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

def assign_process(all_active_tensors):
    #input,output => ..[current_sequence], param => ...tensor

    print(all_active_tensors)
    proc = all_active_tensors['input'] * all_active_tensors['___p']

    _pred = proc
    _label = all_active_tensors['output']

    return _label,_pred
tp.assign_process_process = assign_process

print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([[10],[11],[12]])}))
