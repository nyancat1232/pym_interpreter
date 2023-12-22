#https://wikidocs.net/55580
#model MiniBatch:
#   x_train = FloatTensor([[73, 80, 75],
#                             [93, 88, 93],
#                             [89, 91, 90],
#                             [96, 98, 100],
#                             [73, 66, 70]])
#   y_train = FloatTensor([[152], [185], [180], [196], [142]])
#   y_train = x_train @ ?(1) + ?

#-------------

import torch
from simpletorch.simple import TorchPlus

class Test(TorchPlus):
        def process(self):
            proc = self.input('input',torch.FloatTensor([[73, 80, 75],
                                [93, 88, 93],
                                [89, 91, 90],
                                [96, 98, 100],
                                [73, 66, 70]])) @ self.parameter((3,1),'param')
            proc = proc.squeeze()
            self.label(torch.FloatTensor([152, 185, 180, 196, 142]))
            return proc

tp = Test(meta_optimizer = torch.optim.SGD,meta_optimizer_params = {'lr':1e-5},
          meta_data_per_iteration = 3,meta_epoch=1000)

result = tp.train(show_every_iteration=True)
print(tp.get_parameters())
print(result(input=torch.FloatTensor([[73,80,75]])))
