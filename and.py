import torch

from simpletorch.simple import TorchPlus

tp = TorchPlus()
#assign leaf tensors
def assign_process(current_activator):
    proc = tp.input('input',torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]] )) @ tp.parameter('param',torch.FloatTensor(2,1)) + tp.parameter('param2',torch.FloatTensor(1))

    tp.label('label',torch.FloatTensor([0,0,0,1]))
    return proc
tp._assign_process_prediction = assign_process


result = tp.train()
print(result(input=tp.input('input',torch.FloatTensor([[0,0]]))))
