#model AND:
#   act := [[0,0],[0,1],[1,0],[1,1]] @ ?(1) + ?(1)
#   [0,0,0,1] = act

import torch
import torch.nn as nn
from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()

#assign leaf tensors
#tp._all_predict_tensors.new_tensor('input',TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0),torch.FloatTensor([2,4,6,8]))
#tp._all_label_tensors.new_tensor('label',TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0),torch.FloatTensor([10,20,30,40]))
tp.all_predict_tensors.new_tensor('input',TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0),torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]))
tp.all_predict_tensors.new_tensor('param1',TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1),torch.randn(2,1))
tp.all_predict_tensors.new_tensor('param2',TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1),torch.randn(1))
tp.all_label_tensors.new_tensor('label',TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0),torch.FloatTensor([0,0,0,1]))

def assign_process(tensors_current_sequence,current_activator):
    #proc = tensors_current_sequence['param']*tensors_current_sequence['input']
    proc1 = tensors_current_sequence['input'] @ tensors_current_sequence['param1'] + tensors_current_sequence['param2']
    if proc1 > 0.:
        proc2 = proc1/proc1
    else:
        proc2 = proc1*torch.FloatTensor([0.])
    print(proc2)
    return proc2
tp.assign_process_prediction = assign_process


print(tp.train())
print(tp.predict(input=torch.FloatTensor([[0,0],[0,1],[1,1]])))
