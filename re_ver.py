import sys
import re
#https://regex101.com/

def interpret(variable_express:str):
    pre_str='''from pyplus.perceptron.syntax_implement import reflect\n
from pyplus.perceptron.graph import Node,default_optimizer\n'''

    condition_of_var_name=r'[^=][^=\+\-\* \n\(\)]*'
    condition_of_func_name=r'(.*)(\(.*\))*'

    res_str=variable_express

    model_block_finder=re.compile(r'model (.*):\n')
    res_str = model_block_finder.sub(r'def \1():\n',res_str)

    model_block_finder=re.compile(r'( *)([^ ]*) *\?= *([^ \n]*)\n?')
    res_str = model_block_finder.sub(r'\1(\2,\3)',res_str)
    model_block_finder=re.compile(rf'\?({condition_of_var_name})')
    res_str = model_block_finder.sub(r'Node(\1,is_parameter=True)',res_str)
    model_block_finder=re.compile(rf'\?')
    res_str = model_block_finder.sub(r'Node(randint(),is_parameter=True)',res_str)
    model_block_finder=re.compile(r'\$(.*)(\(.*\))')
    res_str = model_block_finder.sub(r'Node(\1\2)',res_str)
    model_block_finder=re.compile(rf'\$({condition_of_var_name})')
    res_str = model_block_finder.sub(r'Node(\1)',res_str)

    return pre_str+res_str

with open(sys.argv[1],'r') as pym:
    oo = pym.read()
    v=interpret(oo)
    with open(sys.argv[1]+'.py','w') as pyr:
        pyr.write(v)
