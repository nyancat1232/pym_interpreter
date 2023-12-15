#https://wikidocs.net/55580

#-------------
import torch

from pyplus.pytorch.simple import TorchPlus,TTPType,TorchTensorPlus

tp = TorchPlus()
tp.is_sequence=True
tp.meta_optimizer = torch.optim.SGD
tp.meta_optimizer_learning_rate = 1e-5
tp.meta_optimizer_epoch = 20
def assign_values(self:TorchPlus):
    #elf.all_leaf_tensors[0]  = TorchTensorPlus(ttype=TTPType.INPUT)
    #self.all_leaf_tensors[0].tensor = torch.FloatTensor([2])
    self.all_leaf_tensors['0']  = TorchTensorPlus(ttype=TTPType.INPUT)
    self.all_leaf_tensors['0'].tensor = torch.FloatTensor([[73,  80,  75], 
                                [93,  88,  93], 
                                [89,  91,  90], 
                                [96,  98,  100],   
                                [73,  66,  70]])  
    self.all_leaf_tensors[1]  = TorchTensorPlus(ttype=TTPType.DEFAULT)
    self.all_leaf_tensors[1].tensor = torch.FloatTensor([[152], [185], [180], [196], [142]])
    self.all_leaf_tensors[2] = TorchTensorPlus(ttype=TTPType.PARAMETER)
    self.all_leaf_tensors[2].tensor = torch.FloatTensor(self.all_leaf_tensors['0'].tensor.shape[-1],
                                                        1)
tp.assign_leaf_tensors=assign_values
def assign_process(self:TorchPlus,current_sequence:int):
    #.......
    #
    #=self._pred = self.all_leaf_tensors['0'][current_sequence] @ self.all_leaf_tensors[2].tensor
    #self._label = self.all_leaf_tensors[1][current_sequence]
    self._pred = self.all_leaf_tensors['0'][current_sequence] @ self.all_leaf_tensors[2].tensor
    self._label = self.all_leaf_tensors[1][current_sequence]
tp.assign_process_process = assign_process

ins=tp
ins.train()
print(ins.predict(**{'0':torch.FloatTensor([[73, 80, 75]])}))