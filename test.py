from unittest import TestCase
from simpletorch import Variable
from simpletorch.core_simple import square,exp,add,mul,init_variable,Square,Exp
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
        v3.backward_from_end()
        self.assertTrue(np.allclose(v0.grad,3.29744))
    
    def test_add1(self):  
        v0=Variable(3)
        v=add(v0,2)
        self.assertEqual(v.data,5)
        v.backward_from_end()
        self.assertEqual(v0.grad,1)

    def test_back(self):
        v0 = Variable(3)
        v1 = Variable(4)
        v = add(v0,v1) #7
        a = add(1,v) #8
        r = add(4,a) #12
        s = square(r) #144

        s.backward_from_end()
        self.assertEqual(v0.grad,24.)
        self.assertEqual(v1.grad,24.)

class AddTest(TestCase):
    def test_add2(self):
        vv = add(3,2)
        self.assertEqual(vv.data,5)
    def test_duplicate_reference(self):
        v = Variable(2)

        r = add(v,v)
        r2 = add(r,v)
        r2.backward_from_end()
        self.assertEqual(v.grad,3)

    def test_sub(self):
        r = Variable(3.4)
        v = Variable(2.3)
        rr = r-v
        c = Variable(10.)
        rr2 = rr*c
        
        self.assertAlmostEqual(rr2.data,11.) # 11.
        rr2.backward_from_end()
        self.assertAlmostEqual(r.grad,10.) 
        self.assertAlmostEqual(v.grad,-10.) 
    
    def test_mul(self):
        r = Variable(3.4)
        r1 = Variable(2)
        v0 = mul(r,r1)
        v1 = mul(v0,10.)
        self.assertAlmostEqual(v1.data,68.0)
        v1.backward_from_end()
        self.assertAlmostEqual(r.grad,20.0)
        self.assertAlmostEqual(r1.grad,34.0)
    
    def test_div(self):
        r = Variable(4.5)
        v = Variable(1.5)
        rr = r/v

        self.assertAlmostEqual(rr.data,3.0)
        rr.backward_from_end()
        self.assertAlmostEqual(r.grad,0.6666666666)
        self.assertAlmostEqual(v.grad,-2)

class PowTest(TestCase):
    def test_pow(self):
        x = Variable(2.0)
        y = x** 3
        y.backward_from_end()
        self.assertAlmostEqual(y.data,8.0)
        self.assertAlmostEqual(x.grad,12.0)

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

    def test_matyas(self):
        def matyas(x,y):
            vx = Variable(x,'x')
            vy = Variable(y,'y')
            return 0.26*(vx**2 + vy**2)-0.48 * (vx*vy)

        r = matyas(1.,1.)
        self.assertAlmostEqual(r.data,0.040)