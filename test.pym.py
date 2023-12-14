from pyplus.perceptron.syntax_implement import reflect

from pyplus.perceptron.graph import Node,default_optimizer

model LinearRegression:
    hidden_1st := X*?W1
    hidden_2nd := hidden_1st*?Dense(4)
    out = hidden_2nd*?Dense(1)
    Y ?= out


xa.vi.er.viv.ss.bob.we.void()


!loss MSE
!optimizer Adam(1e-6,2000)
!activator ReLU
model LinearRegression:
    Y ?= a*?1

model Arithmetic:
    3 ?= ?4-1


!loss MSELoss(reduction='sum')
!optimizer SGD(lr=1e-6,2000)
model Polynomial3:
    
    x = torch.linspace(-math.pi, 1.*math.pi, 2000, dtype=dtype)
    y = torch.sin(x)
    
    y ?= ?0 + ?0.*x + ?0*x**2 + ?0*x**3

model SimpleExample:
    10 ?= <<autogen>> * 2

model SimpleExample:
    10 ?= <<autogen>> * 2 * <<autogen>>* <<autogen>>

model Xor:
    [0,1,1,0] ?= [[0,0],[0,1],[1,0],[0,0]] * <<autogen>> 


model AutoConvert:
    10 ?= ?2 + ?2.1 * <<autogen>> @ <<autogen>>(4) * <<autogen>> @ <<autogen>>(1)