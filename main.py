from pyplus.perceptron.graph import Node, default_optimizer
from pyplus.perceptron.syntax_implement import reflect
import re

def interpret(variable_express:str):
    res = ""

    indents = [False,False]
    current_indent_text = None
    current_indent_internal = ""
    for line in variable_express.splitlines():
        if indents[0] and not indents[1]:
            for c in line:
                if c == ' ':
                    current_indent_internal += c
                else:
                    indents[1]=True
                    break
        
        if 'model' in line:
            current_indent_text = line[:line.find('model')]
            indents[0] = True
            line=line.replace('model','def')
            res += line+'\n'

        if all(indents):
            if line.startswith(current_indent_text+current_indent_internal):
                if line.find('reflect') != -1:
                    line=line.replace('reflect','return reflect(local())()')
                res += line+'\n'
            else:
                indents[0]=False
                indents[1]=False
                continue
    return res

data = '''
v=3

    model Xor:
        def MakeFunc(arr):
            return arr[0]*?randint() + arr[1]*?randint()
        reflect(Adam(100)) [0,1,1,0],MakeFunc([[0,0],[1,0],[0,1],[1,1]])

        

        model Xor:
            def MakeFunc(arr):
                return arr[0]*?randint() + arr[1]*?randint()
            reflect(Adam(100)) [0,1,1,0],MakeFunc([[0,0],[1,0],[0,1],[1,1]])

x=1
'''

data2 = '''
ewijfwef
model Simple:
    reflect 3,2+?0
fwoeijfew
'''


#return reflect(locals())(optimizer,epoch)(target,predict)

res = interpret(data2)
print(res)