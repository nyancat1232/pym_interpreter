#!sequence
#model SimpleExampleSequence:
#    [10,20,30,40] ?= ?(1) * $[2,4,6,8]

import torch
import torch.nn as nn
from simpletorch.simple import TorchPlus,MetaDataType,CurrentStateInformation

class Test(TorchPlus):
    def process(self):
        proc = self.parameter('param',[1])*self.input(data=[2,4,6,8],meta_data_type=MetaDataType.NUMERICAL,name='input')

        self.label([10,20,30,40],MetaDataType.NUMERICAL)
        return proc
    def show_progress(self,csi:CurrentStateInformation):
        print(f'Epoch : {csi.current_epoch} \tIteration : {csi.current_iteration}/{csi.len_iteration}\tLoss : {csi.current_loss}')

tp = Test()
tp.meta_data_per_iteration=3
result = tp.train()
print(tp.get_parameters())
print(result(input=torch.FloatTensor([1,2,3,4,1,2,3])))
