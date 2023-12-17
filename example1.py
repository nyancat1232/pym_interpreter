#model SimpleExample:
#    s[18,36] ?= ? * $s[2,4]

import torch
import torch.nn as nn

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus,ModeType



tp = TorchPlus()

tp.meta_activator = nn.ReLU
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_epoch = 300
tp.meta_optimizer_params = {'lr':0.015}
tp.meta_error_measurement = torch.nn.MSELoss

#assign leaf tensors
tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
tp['input'].tensor = torch.FloatTensor([2.,4.])
tp['___p']  = TorchTensorPlus(ttype=TTPType.PARAMETER)
tp['___p'].tensor = torch.FloatTensor(1)
tp.label_tensor = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
tp.label_tensor.tensor = torch.FloatTensor([18.,36.])

print(tp._all_leaf_tensors[0])
print(tp._all_leaf_tensors[1])
print(tp._all_leaf_tensors.tensors_label[0])
print(tp._all_leaf_tensors.tensors_label[1])
print(tp._all_leaf_tensors.tensors_label)

def assign_process(tensors_current_sequence,current_activator):
    proc = tensors_current_sequence['input'] * tensors_current_sequence['___p']

    return proc
tp.assign_process_prediction = assign_process


print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([10,11,12])}))
