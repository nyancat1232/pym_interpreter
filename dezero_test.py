from simpletorch.dezero import *

v0=Variable(np.array(0.5))
print(v0)
v1=Square()(v0)
v2=Exp()(v1)
v3=Square()(v2)
v3.grad=np.array(1.0)
print(v3)
aa = v3.backward()
print(v0)
print(aa)
while False:
    F1 = Square
    F2 = Exp

    a = Square(v)
    b = Exp(a)
    c = Square(b)
    print(b.data)
    print(c.data)

    c.grad = np.array(1.)
    c.backward()
    print(v.grad)
    print(a.grad)
    print(b.grad)
    print(c.grad)


while False:
    invf1 = c.creator
    inv2 = invf1.input
    inv2.grad = invf1.backward(c.grad)
    invf2 = inv2.creator
    inv3 = invf2.input
    inv3.grad = invf2.backward(inv2.grad)
    invf3 = inv3.creator
    inv4 = invf3.input
    inv4.grad = invf3.backward(inv3.grad)
    print(inv4.data)
    print(inv4.grad)

#bb = F1.backward(F2.backward(F3.backward(aa.grad)))
#print(bb)

