# Author: Fabrício Olivetti de França

# Utility functions
empty = lambda x: len(x) == 0

# Check for feasible states
def feasible( st ):
    s, t = st
    if len(s) == 0: return True
    return (len(s) <= 8) and (not attack(s[0], s[1:])) and feasible((s[1:],t))

# Check if a row is attacking another row of a state
def attack(r, rs):
    return (r in rs) or (r in upperDiag(rs)) or (r in lowerDiag(rs))
    
def upperDiag(rs):
    return [x+y for x,y in zip(rs, range(1,9))]
def lowerDiag(rs):
    return [x-y for x,y in zip(rs, range(1,9))]

# Check if it is a goal state
def isGoal(st):
    return feasible(st) and len(st[0]) == 8

# Perform a move
def move(x, st):
    return ([x] + st[0], [x] + st[1])

def moves(sts):
    new_sts = []
    for st in sts:
        choices = list(range(1,9))
        for c in choices:
            new_sts.append(move(c, st))
    return new_sts

emptySt = ([],[])

# Breadth-first search
def bfs( sts ):
    goal    = list(filter(isGoal, sts))
    n       = 0
    while len(goal) == 0:
        sts  = list(filter(feasible, moves(sts)))
        goal = list(filter(isGoal, sts))
        n    = n + len(sts)
    return goal[0], n

# Depth-first search
def dfs( st, n ):
    s, t = st
    if isGoal(st): return st, n
    if not feasible(st): return None, n
    new_sts = []
    for x in range(1,9):
        new_st, n = dfs(move(x, st), n+1)
        if new_st is not None:
            return new_st, n 
    return None, n

def main():
  s0 = ([],[])      
  print (bfs([s0]))
  st, n =  dfs(s0, 0)
  print (st[0], n)
  
main()
