from pyplus.perceptron.syntax_implement import reflect

from pyplus.perceptron.graph import Node,default_optimizer

def LinearRegression():
    hidden_1st := X*Node(W1,is_parameter=True)
    hidden_2nd := hidden_1st*Node(Dense,is_parameter=True)(4)
    out = hidden_2nd*Node(Dense,is_parameter=True)(1)
    (Y,out)
def LinearRegression<__err__=MSE,__opt__=Adam,__act__=ReLU>():
    (Y,Node(a*Node(1,is_parameter=True)))
def Arithmetic():
    (3,Node(4,is_parameter=True)-Node(1))

def TorchExample<__epo__=2000,__opt__=SCD(1e-6),__err__=MSE>():
    #https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    x = Node(torch.linspace(-math.pi, 1.*math.pi, 2000, dtype=dtype))
    y = Node(torch.sin(x))
    if __cur_epo__ % 100 == 99:
        print(__cur_epo__,__cur__err__)
    
    (y,Node(),is_parameter=True) + Node(*x,is_parameter=True) + Node(*x,is_parameter=True)**2 + Node(*x,is_parameter=True)**3