from pyplus.perceptron.syntax_implement import reflect
from pyplus.perceptron.graph import Node,default_optimizer

def Linear():
    return reflect(locals())(default_optimizer,100)(Node(3.2),Node(3)+Node(2.2,is_parameter=True))

aa=Linear()
print(aa)