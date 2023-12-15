#!activator ReLU
#model Xor:
#    [[0],[1],[1],[0]] ?= [[0,0],[0,1],[1,0],[0,0]] * ? #? automatically converted as 2x1

import torch
import torch.nn as nn

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()
tp.is_sequence=True
tp.meta_activator = nn.ReLU
def assign_values(self:TorchPlus):
    #elf.all_leaf_tensors[0]  = TorchTensorPlus(ttype=TTPType.INPUT)
    #self.all_leaf_tensors[0].tensor = torch.FloatTensor([2])
    self.all_leaf_tensors['input']  = TorchTensorPlus(ttype=TTPType.INPUT)
    self.all_leaf_tensors['input'].tensor = torch.FloatTensor([[0,0],[0,1],[1,0],[0,0]])
    self.all_leaf_tensors['output']  = TorchTensorPlus(ttype=TTPType.DEFAULT)
    self.all_leaf_tensors['output'].tensor = torch.FloatTensor([[0],[1],[1],[0]])
    self.all_leaf_tensors['___0']  = TorchTensorPlus(ttype=TTPType.PARAMETER)
    self.all_leaf_tensors['___0'].tensor = torch.FloatTensor(self.all_leaf_tensors['output'].tensor.shape[-1],self.all_leaf_tensors['input'].tensor.shape[-1])
tp.assign_leaf_tensors=assign_values
def assign_process(self:TorchPlus,current_sequence:int):
    #.......
    #
    #=self._pred = self.all_leaf_tensors['0'][current_sequence] @ self.all_leaf_tensors[2].tensor
    #self._label = self.all_leaf_tensors[1][current_sequence]
    self._proc1 = self.all_leaf_tensors['input'][current_sequence] * self.all_leaf_tensors['___0'].tensor
    self._proc1_a = self._current_activator(self._proc1)

    self._pred = self._proc1_a
    self._label = self.all_leaf_tensors['output'][current_sequence]
tp.assign_process_process = assign_process

print(tp.train())
print(tp.predict(input=torch.FloatTensor([[1,1],[0,1]])))
