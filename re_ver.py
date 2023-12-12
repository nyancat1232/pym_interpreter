import sys
import re
#https://regex101.com/

def interpret(variable_express:str):
    pre_str='''from pyplus.perceptron.syntax_implement import reflect\n
from pyplus.perceptron.graph import Node,default_optimizer\n'''

    res_str=variable_express

    model_block_finder=re.compile(r'model (.*):\n')
    res_str = model_block_finder.sub(r'def \1():\n',res_str)
    model_block_finder=re.compile(r'\?([^+?* \n,\(\)]*)')
    res_str = model_block_finder.sub(r'Node(\1,is_parameter=True)',res_str)
    model_block_finder=re.compile(r'\$([^+?* \n,\(\)]*)')
    res_str = model_block_finder.sub(r'Node(\1)',res_str)
    model_block_finder=re.compile(r'reflect(.*) (.*)\n')
    res_str = model_block_finder.sub(r'return reflect(locals())\1(\2)\n',res_str)
    return pre_str+res_str

with open(sys.argv[1],'r') as pym:
    oo = pym.read()
    v=interpret(oo)
    with open(sys.argv[1]+'.py','w') as pyr:
        pyr.write(v)
