#model SimpleExample:
#    s[18,36] ?= ? * $s[2,4]

import torch

from simpletorch.simple import TorchPlus,MetaDataType,CurrentStateInformation

class Test(TorchPlus):
    def process(self):
        proc = self.input(data=[2.,4.],meta_data_type=MetaDataType.NUMERICAL,name='input') * self.parameter(size=[1],name='param')
        
        self.label(data=[18.,36.],meta_data_type=MetaDataType.NUMERICAL)
        return proc
    def show_progress(self,csi:CurrentStateInformation):
        print(f'pred : {csi.current_result}')
        print(f'Epoch : {csi.current_epoch} \tIteration : {csi.current_iteration}/{csi.len_iteration}\tLoss : {csi.current_loss}')

tp = Test()
result = tp.train(10000)
print(tp.get_parameters())
print(result(input=torch.FloatTensor([10,21,12,95])))
