#!sequence
#model SimpleExampleSequence:
#    [10,20,30,40] ?= ?(1) * $[2,4,6,8]

import torch
import torch.nn as nn
from simpletorch.simple import TorchPlus

class Test(TorchPlus):
    def process(self):
        proc = self.parameter([1],'param')*self.input('input',torch.FloatTensor([2,4,6,8]))

        self.label([10,20,30,40])
        return proc
tp = Test()
tp.meta_data_per_iteration=3

result = tp.train(show_every_iteration=True)
print(tp.get_parameters())
print(result(input=torch.FloatTensor([1,2,3,4,1,2,3])))
