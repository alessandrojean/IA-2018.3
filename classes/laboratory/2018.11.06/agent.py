# Author: Fabrício Olivetti de França

def createMDP():

    '''
    definição do ambiente
    '''

    S       = [(i,j) for i in range(1,5) 
                     for j in range(1,4) if (i,j) != (2,2)]
        
    goals   = [(4,3), (4,2)]
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    A       = {s : actions if s not in goals else [None]
               for s in S }
               
    R        = {s : -0.04 for s in S}
    R[(4,3)] =  1
    R[(4,2)] = -1

    P        = { (s,a) : pvals(s, a, S) for s in S for a in A[s] }

    gamma    = .9
    
    return (S,A,R,P,gamma)
    
    
def move(s, a, S):
    i, j = s
    if a == "UP":
        sp = (i, j+1)
    elif a == "DOWN":
        sp = (i, j-1)
    elif a == "LEFT":
        sp = (i-1, j)
    elif a == "RIGHT":
        sp = (i+1, j)
    elif a is None:
        return s

    if sp in S:
        return sp

    return s    

def succ(a):
    return {"UP": "RIGHT", "DOWN": "LEFT", "RIGHT": "DOWN", "LEFT": "UP", None : None}[a]

def pred(a):
    return {"UP": "LEFT", "DOWN": "RIGHT", "RIGHT": "UP", "LEFT": "DOWN", None : None}[a]

def pvals(s, a, S):
    return [(0.8, move(s, a, S)), (0.1, move(s, succ(a), S)), (0.1, move(s, pred(a), S))]
