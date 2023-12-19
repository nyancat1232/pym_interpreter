#model SimpleExample:
#    s[18,36] ?= ? * $s[2,4]

import torch
import torch.nn as nn

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()

#assign leaf tensors
tp.all_predict_tensors.new_tensor('input',TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0),torch.FloatTensor([2.,4.]))
tp.all_predict_tensors.new_tensor('param',TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1),torch.FloatTensor(1))
tp.all_label_tensors.new_tensor('label',TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0),torch.FloatTensor([18.,36.]))


def assign_process(tensors_current_sequence,current_activator):
    proc = tensors_current_sequence['input'] * tensors_current_sequence['param']

    return proc
tp.assign_process_prediction = assign_process


result = tp.train()
print(result(input=torch.FloatTensor([10,21,12])))
