from pyplus.perceptron.syntax_implement import reflect

from pyplus.perceptron.graph import Node,default_optimizer
def Linear():
    return reflect(locals())(default_optimizer,100)(Node(3.),Node(2.)+Node(0.,is_parameter=True))

a=Linear()
print(a)