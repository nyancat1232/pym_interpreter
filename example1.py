#model SimpleExample:
#    s[18,36] ?= ? * $s[2,4]

import torch

from pyplus.pytorch.simple import TorchPlus

class Test(TorchPlus):
    def process(self):
        proc = self.input('input',torch.FloatTensor([2.,4.])) * self.parameter('param',torch.FloatTensor(1))

        self.label(torch.FloatTensor([18.,36.]))
        return proc


result = Test().train()
print(result(input=torch.FloatTensor([10,21,12])))
