#model SimpleExample:
#    s[18,36] ?= ? * $s[2,4]

import torch

from simpletorch.simple import TorchPlus,MetaDataType

class Test(TorchPlus):
    def process(self):
        proc = self.input([2.,4.],MetaDataType.NUMERICAL,'input') * self.parameter([1],'param')

        self.label([18.,36.],MetaDataType.NUMERICAL)
        return proc

tp = Test()
result = tp.train(10000)
print(tp.get_parameters())
print(result(input=torch.FloatTensor([10,21,12,95])))
