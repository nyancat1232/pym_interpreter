from simpletorch.dezero import *

v0 = Variable(3)
v = add(v0,4)
v2 = add(31,44)
a = add(v2,v)
r = add(4,a)

fs = Square()
s = fs(r)[0]
print(v0)
s.backward()
print(v0)