from unittest import TestCase
import torch
import numpy as np

from simpletorch.simple import TorchPlus,MetaDataType,CurrentStateInformation

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
            self.assertAlmostEqual(result_result[index].detach().numpy()[0], label,delta=1e-1)
