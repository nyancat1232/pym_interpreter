#!sequence
#model SimpleExampleSequence:
#    [10,20,30,40] ?= ?(1) * $[2,4,6,8]

import torch
import torch.nn as nn
from simpletorch.simple import TorchPlus,MetaDataType

class Test(TorchPlus):
    def process(self):
        proc = self.parameter([1],'param')*self.input([2,4,6,8],meta_data_type=MetaDataType.NUMERICAL,name='input')

        self.label([10,20,30,40],MetaDataType.NUMERICAL)
        return proc

tp = Test()
tp.meta_data_per_iteration=3
result = tp.train()
print(tp.get_parameters())
print(result(input=torch.FloatTensor([1,2,3,4,1,2,3])))
