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
tp.meta_optimizer_data_per_iteration = 10
#assign leaf tensors
tp.all_predict_tensors.new_tensor('input',TorchTensorPlus(ttype=TTPType.INPUT,axis_sequence=0),torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]]))
tp.all_predict_tensors.new_tensor('param',TorchTensorPlus(ttype=TTPType.PARAMETER,axis_sequence=-1),torch.FloatTensor(3,1))
tp.all_label_tensors.new_tensor('label',TorchTensorPlus(ttype=TTPType.DEFAULT,axis_sequence=0),torch.FloatTensor([[152], [185], [180], [196], [142]]))

def assign_process(tensors_current_sequence,current_activavtor):
    return tensors_current_sequence['input']@tensors_current_sequence['param']
tp.assign_process_prediction = assign_process


result = tp.train()
print(result(input=torch.FloatTensor([[73,80,75]])))
