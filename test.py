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

        self.assertTrue(np.allclose(v3.data,1.64872))
        v3.backward()
        self.assertTrue(np.allclose(v0.grad,3.29744))
    
    def test_add(self):  
        v0=Variable(3)
        v=Add()(v0,2)
        self.assertEqual(v.data,5)
        v.backward()
        self.assertEqual(v0.grad,1)

class AddTest(TestCase):
    def test_add(self):
        vv = Add()(3,2)
        self.assertEqual(vv.data,5)

class TypeTest(TestCase):
    def test_types(self):
        self.assertIsInstance(init_variable(3),Variable)
    def test_function_type(self):
        self.assertIsInstance(square(4),Variable)

class GenerationTest(TestCase):
    def test_generation(self):
        v = add(3,4)
        a = add(1,v)
        r = add(4,a)
        s = square(r)
        self.assertEqual(v,1)
        self.assertEqual(a,2)
        self.assertEqual(r,3)
        self.assertEqual(s,4)