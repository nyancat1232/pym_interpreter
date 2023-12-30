from unittest import TestCase
from simpletorch.dezero import *
import numpy as np
class SquareTest(TestCase):
    def test_something(self):
        x = Variable(2.0)
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(expected,y.data)


class SquareExpTest(TestCase):
    def test_forward(self):
        v0=Variable(0.5)
        v3=square(exp(square(v0)))

        v3.grad=np.array(1.0)
        self.assertTrue(np.allclose(v3.data,1.64872))
        v3.backward()
        self.assertTrue(np.allclose(v0.grad,3.29744))