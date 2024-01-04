from unittest import TestCase
import torch
import numpy as np

from simpletorch.simpletorch import TorchPlus,MetaDataType,CurrentStateInformation

def type_to(tens:torch.Tensor):
    return tens.detach().numpy()[0]

class SimpleTorchChecker(TestCase):
    def test_example1(self):
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
        print(tp.parameter('param'))

        tests = [10,21,12,95]
        tests_label = [v*9 for v in tests]
        result_result = result(input=torch.FloatTensor(tests))

        for index,label in enumerate(tests_label):
            self.assertAlmostEqual(type_to(result_result[index]), label,delta=1e-1)

    def test_example5(self):
        x=[[73, 80, 75],
            [93, 88, 93],
            [89, 91, 90],
            [96, 98, 100],
            [73, 66, 70]]
        label = [152, 185, 180, 196, 142]

        class Test(TorchPlus):
            def process(self):
                proc = self.input('input',x,MetaDataType.NUMERICAL) @ self.parameter('param',(3,1))
                proc = proc.squeeze()
                self.label(label,MetaDataType.NUMERICAL)
                return proc
            def show_progress(self,csi:CurrentStateInformation):
                print(f'Epoch : {csi.current_epoch} \tIteration : {csi.current_iteration}/{csi.len_iteration}\tLoss : {csi.current_loss}')


        tp = Test(meta_optimizer = torch.optim.SGD,meta_optimizer_params = {'lr':1e-5},
                meta_data_per_iteration = 3)

        result = tp.train(epoch=10000)
        for x,label in zip(x,label):
            self.assertAlmostEqual(result(input=torch.FloatTensor([x]))[0].detach().numpy(),label,delta=1e+1)
