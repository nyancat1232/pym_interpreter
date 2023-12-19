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
from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlusInternal

tp = TorchPlus()

tp.meta_activator = torch.relu
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_params = {'lr':1e-5}
tp.meta_data_per_iteration = 10
#assign leaf tensors

def assign_process(current_activavtor):
    proc = tp.input('input',torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])) @ tp.parameter('param',torch.FloatTensor(3,1))
    
    tp.label('label',torch.FloatTensor([[152], [185], [180], [196], [142]]))
    return proc
tp.assign_process_prediction = assign_process


result = tp.train()
print(result(input=torch.FloatTensor([[73,80,75]])))
