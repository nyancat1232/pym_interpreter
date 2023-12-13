import sys
import re
#https://regex101.com/

def interpret(variable_express:str):
    pre_str='''from pyplus.perceptron.syntax_implement import reflect\n
from pyplus.perceptron.graph import Node,default_optimizer\n'''
    res_str=variable_express

    #remove comments
    find=r'\s*#.*'
    model_block_finder=re.compile(find)
    res_str = model_block_finder.sub(r'',res_str)

    return pre_str+res_str