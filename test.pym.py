from pyplus.perceptron.syntax_implement import reflect

from pyplus.perceptron.graph import Node,default_optimizer

model LinearRegression:
    hidden_1st := X*?W1
    hidden_2nd := hidden_1st*?Dense(4)
    out = hidden_2nd*?Dense(1)
    Y ?= out


xa.vi.er.viv.ss.bob.we.void()


@model_apply(error = MSE)
@model_apply(optimizer=Adam(1e-6,2000))
@model_apply(activator=ReLU)
model LinearRegression:
    Y ?= a*?1

model Arithmetic:
    3 ?= ?4-1

@model_apply(error = MSE)
@model_apply(optimizer=SCD(1e-6,2000))
model TorchExample:
    x = torch.linspace(-math.pi, 1.*math.pi, 2000, dtype=dtype)
    y = torch.sin(x)
    if __cur_epo__ % 100 == 99:
        print(__cur_epo__,__cur__err__)
    
    y ?= ?0 + ?0.*x + ?0*x**2 + ?0*x**3

@model_apply(error = MSE)
@model_apply(optimizer=SCD(1e-6,2000))
model TorchExample:
    x = torch.linspace(-math.pi, 1.*math.pi, 2000, dtype=dtype)
    y = torch.sin(x)
    if __cur_epo__ % 100 == 99:
        print(__cur_epo__,__cur__err__)
    
    y ?= ?0 + ?0.*x + ?0*x**2 + ?0*x**3