#model SimpleExample:
#    s[18,36] ?= ? * $s[2,4]

import torch

from simpletorch.simple import TorchPlus,MetaDataType,CurrentStateInformation

class Test(TorchPlus):
    def process(self):
        proc = self.input([2.,4.],MetaDataType.NUMERICAL,'input') * self.parameter([1],'param')
        
        self.label([18.,36.],MetaDataType.NUMERICAL)
        return proc
    def show_progress(self,csi:CurrentStateInformation):
        print(f'Epoch : {csi.current_epoch} \tIteration : {csi.current_iteration}/{csi.len_iteration}\tLoss : {csi.current_loss}')

tp = Test()
result = tp.train(10000)
print(tp.get_parameters())
print(result(input=torch.FloatTensor([10,21,12,95])))
