import ply.lex as lex
#https://www.dabeaz.com/ply/ply.html
#https://fwani.tistory.com/17

tokens = (
    'NUMBER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'LPAREN',
    'RPAREN',
    'QUESTION',
    'EXCLAMATION',
    'ID',
    'DOT'
)

# A regular expression rule with some action code
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)    
    return t

t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
# 리턴값을 무시
def t_COMMENT(t):
    r'\#.*'
    pass

# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


t_QUESTION = r'\?'
t_EXCLAMATION = r'!'

reserved = {
    'model' : 'MODEL',
    'reflect' : 'REFLECT',
}

tokens = list(tokens) + list(reserved.values())

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value,'ID')    # Check for reserved words
    return t

t_DOT  = r'.'

# Build the lexer
lexer = lex.lex()

# Test it out
data = '''
model Xor:
    def MakeFunc(arr):
        return arr[0]*?randint() + arr[1]*?randint()
    reflect(Adam(100)) [0,1,1,0],MakeFunc([[0,0],[1,0],[0,1],[1,1]]) MakeFunc
'''

# Give the lexer some input
lexer.input(data)

# Tokenize
while True:
    tok = lexer.token()
    if not tok: 
        break      # No more input
    print(tok)