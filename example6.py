#!activator ReLU
#model Xor:
#    [[0],[1],[1],[0]] ?= ($[[0,0],[0,1],[1,0],[0,0]]+[[1,1,1,1]]) @ ? #? automatically converted as 2x1

import torch
import torch.nn as nn
from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()

tp.meta_activator = nn.ReLU
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_epoch = 300
tp.meta_optimizer_params = {'lr':0.015}
tp.meta_error_measurement = torch.nn.MSELoss

#assign leaf tensors
tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
tp['input'].tensor = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
tp['def1']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)
tp['def1'].tensor = torch.FloatTensor(2,1)
tp['param2']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=0)
tp['param2'].tensor = torch.FloatTensor(2,1)
tp['output']  = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
tp['output'].tensor = torch.FloatTensor([[0],[1],[1],[0]])

def assign_process(tensors_current_sequence,current_activator):
    #proc = tensors_current_sequence['input'] @ tensors_current_sequence['param']
    proc = tensors_current_sequence['input'] @ tensors_current_sequence['param2']
    proc = current_activator(proc)
    _pred = proc
    _label = tensors_current_sequence['output']

    return _label,_pred
tp.assign_process_process = assign_process

print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([[0,1],[1,1]])}))