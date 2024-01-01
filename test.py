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
    
    def test_add1(self):  
        v0=Variable(3)
        v=add(v0,2)
        self.assertEqual(v.data,5)
        v.backward()
        self.assertEqual(v0.grad,1)

class AddTest(TestCase):
    def test_add2(self):
        vv = add(3,2)
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
        self.assertEqual(v.generation,1)
        self.assertEqual(a.generation,2)
        self.assertEqual(r.generation,3)
        self.assertEqual(s.generation,4)

    def test_generation2(self):
        fsq = Square()
        rfsq=fsq(Variable(3.0))[0]
        self.assertEqual(fsq.generation,0)
        self.assertEqual(rfsq.generation,1)
        fex = Exp()
        gen1 = fex(rfsq)[0]
        self.assertEqual(fex.generation,1)
        self.assertEqual(gen1.generation,2)