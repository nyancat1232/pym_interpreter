from pyplus.perceptron.syntax_implement import reflect

from pyplus.perceptron.graph import Node,default_optimizer

model LinearRegression:
    hidden_1st := X*?W1
    hidden_2nd := hidden_1st*Node(Dense(4),is_parameter=True)
    out = hidden_2nd*Node(Dense(1),is_parameter=True)
    Y ?= out


Node(xa.vi.er.viv.ss.bob.we.void())



model LinearRegression:
    Y ?= a*?1

model Arithmetic:
    3 ?= ?4-1


model TorchExample:
    x = Node(torch.linspace(-math.pi, 1.*math.pi, 2000, dtype=dtype))
    y = Node(torch.sin(x))
    if __cur_epo__ % 100 == 99:
        print(__cur_epo__,__cur__err__)
    
    y ?= ?0 + ?0.*x + ?0*x**2 + ?0*x**3