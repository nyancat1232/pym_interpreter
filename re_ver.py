import sys
import re
#https://regex101.com/


pp=r'[\+\-\*\/\@\s]'


def interpret(variable_express:str):
    pre_str='''from pyplus.perceptron.syntax_implement import reflect\n
from pyplus.perceptron.graph import Node,default_optimizer\n'''
    res_str=variable_express

    #remove comments
    find=rf'\?({pp})'
    model_block_finder=re.compile(find)
    res_str=model_block_finder.sub(r'<<autogen>>\1',res_str)

    return pre_str+res_str

with open(sys.argv[1],'r') as read_file:
    rr = read_file.read()
    with open(sys.argv[1]+'.py','w') as write_file:
        write_file.write(interpret(rr))