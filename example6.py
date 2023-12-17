#!activator ReLU
#model Xor:
#    proc1 := $[[0,0],[0,1],[1,0],[0,0]] @ ?(4) + ?
#    proc2 = proc1 @ ?(1) + ?
#    [0,1,1,0] ?= proc2


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
#tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
#tp['input'].tensor = torch.FloatTensor([2.,4.])
#tp.label_tensor = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
#tp.label_tensor.tensor = torch.FloatTensor([18.,36.])
tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
tp['input'].tensor = torch.FloatTensor([[0,0],[0,1],[1,0],[0,0]])
tp['param1']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)
tp['param1'].tensor = torch.FloatTensor(2,4)
tp['param2']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)
tp['param2'].tensor = torch.FloatTensor(4)
tp['param3']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)
tp['param3'].tensor = torch.FloatTensor(4,1)
tp['param4']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)
tp['param4'].tensor = torch.FloatTensor(1)
tp.label_tensor  = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
tp.label_tensor.tensor = torch.FloatTensor([0,1,1,0])

def assign_process(tensors_current_sequence,current_activator):
    proc = tensors_current_sequence['input']@tensors_current_sequence['param1'] + tensors_current_sequence['param2']
    proc = current_activator(proc)
    proc = proc@tensors_current_sequence['param3'] + tensors_current_sequence['param4']

    return proc
tp.assign_process_prediction = assign_process


print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([[0,1],[1,1],[1,1],[1,0]])}))
