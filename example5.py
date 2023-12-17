#https://wikidocs.net/55580
#model MiniBatch:
#   x_train = FloatTensor([[73, 80, 75],
#                             [93, 88, 93],
#                             [89, 91, 90],
#                             [96, 98, 100],
#                             [73, 66, 70]])
#   y_train = FloatTensor([[152], [185], [180], [196], [142]])
#   y_train = x_train @ ?(1) + ?


#-------------

import torch
import torch.nn as nn
from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()

tp.meta_activator = torch.relu
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_params = {'lr':1e-5}

#assign leaf tensors
#tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
#tp['input'].tensor = torch.FloatTensor([2.,4.])
#tp.label_tensor = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
#tp.label_tensor.tensor = torch.FloatTensor([18.,36.])
tp['input']  = TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0)
tp['input'].tensor = torch.FloatTensor( [[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]] )
tp['param']  = TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1)
tp['param'].tensor = torch.FloatTensor(3,1)
tp.label_tensor = TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0)
tp.label_tensor.tensor = torch.FloatTensor([[152], [185], [180], [196], [142]])

def assign_process(tensors_current_sequence,current_activator):
    return tensors_current_sequence['input']@tensors_current_sequence['param']
tp.assign_process_prediction = assign_process


print(tp.train())
print(tp.predict(**{'input':torch.FloatTensor([[73,80,75]])}))
