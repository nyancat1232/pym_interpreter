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

    #pop meta props
    #find=r'\<(.*)\>'
    #model_block_finder=re.compile(find)
    #meta_scope=model_block_finder.findall(res_str)
    #meta_scope=[{v.split('=')[0]:v.split('=')[1] for v in m.split(',')} for m in meta_scope]
    #res_str = model_block_finder.sub(r'',res_str)
    #print(meta_scope)

    #class or function wrapper
    #find=r'((\w+\.)+.\w+\(.*\))'
    #model_block_finder=re.compile(find)
    #res_str = model_block_finder.sub(r'Node(\1)',res_str)

    #question mark(func) to parameter
    find=r'\?(\w+\(\w+\))'
    model_block_finder=re.compile(find)
    res_str = model_block_finder.sub(r'Node(\1,is_parameter=True)',res_str)

    return pre_str+res_str
    #question mark(non func) to parameter
    find=r'\?(\w+)'
    model_block_finder=re.compile(find)
    res_str = model_block_finder.sub(r'Node(\1,is_parameter=True)',res_str)

    return pre_str+res_str
    
    condition_of_var_name=r'[^=][^=\+\-\* \n\(\)]*'
    condition_of_func_name=r'(.*)(\(.*\))*'

    res_str=variable_express
 
    model_block_finder=re.compile(r'([ \+\-\*\/])([0-9]+[0-9.e-]*)')
    res_str = model_block_finder.sub(r'\1Node(\2)',res_str)
    model_block_finder=re.compile(r'\?([^=\n \+\-\*\/]+)')
    res_str = model_block_finder.sub(r'Node(\1,is_parameter=True)',res_str)
    return pre_str+res_str
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
