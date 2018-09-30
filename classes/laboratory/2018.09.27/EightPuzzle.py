# Author: Fabrício Olivetti de França

# Utility functions
empty = lambda x: len(x) == 0

def addTuple(x, y):
    return (x[0] + y[0], x[1] + y[1])
    
def absDiff(x, y):
    return (abs(x[0]-y[0]), abs(x[1]-y[1]))
    
def dist(v, x, y):
    x, y = absDiff( findNum(v,x), findNum(v,y) )
    return x+y
    
def findNum(v, x):
    for i, xi in enumerate(x):
        for j, xij in enumerate(xi):
            if(xij == v): return (i,j)
 
findZero = lambda x: findNum(0, x)

argmin = lambda ls: sorted(ls, key=lambda x: x[0])[0][1]

def inBound(x):
    if 0 <= x <= 2:
        return True
    return False

# Swap the values of a given state
def swap(val, s):
    new_s = [si.copy() for si in s]  # taking proper care about pass by reference
    x1, y1 = findNum(val, s)
    x2, y2 = findZero(s)
    new_s[x1][y1] = 0
    new_s[x2][y2] = val
    return new_s

# Maps a Move to a coordinate
move2coord = {"Up": (-1,0), "Down": (1,0), "Left": (0, -1), "Right": (0,1)}
pos = lambda mv: move2coord[mv]

# Update the state
def move(direc, st):
    s, t = st
    newX, newY = addTuple(findZero(s), pos(direc))    
    if inBound(newX) and inBound(newY):
        val = s[newX][newY]
    else:
        val = 0
    new_s   = swap(val, s)
    return (new_s, t + [direc])

# Initial state
s0 = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]

# Goal state
goalSt = [[1,2,3],[4,5,6],[7,8,0]]

# Check if it is a goal state
def isGoal( st ):
    s, t = st
    return s == goalSt
    
# For this puzzle, every state is feasible
def feasible(x):
    return True

# perform a move
def moves(sts):
    new_sts = []
    for st in sts:
        choices = move2coord.keys()
        for c in choices:
            new_sts.append(move(c, st))
    return new_sts

# Breadth-first search
def bfs( sts ):
    if len(sts) == 0:
        print("Couldn't find a feasible solution")
        return ([],[]), 0
    goal    = list(filter(isGoal, sts))
    n       = 0
    while len(goal) == 0:
        sts  = list(filter(feasible, moves(sts)))
        goal = list(filter(isGoal, sts))
        n    = n+len(sts)
    return goal[0], n

# A* search   
def f1(st):
    s, t = st
    return len(t) + sum([dist(v, s, goalSt) for v in range(1,9)])

def f2(st):
    s, t = st
    return len(t) + sum([s[i][j] != goalSt[i][j] 
                        for i in range(3) 
                        for j in range(3) ]) - 1
    
def f3(st):
    return max(f1(st), f2(st))
    
def astar( sts, f ):
    if len(sts) == 0:
        print("Couldn't find a feasible solution")
        return ([],[]), 0
        
    goal    = list(filter(isGoal, sts))  
    n       = 0  
    sold = sts[0]
    while len(goal) == 0:
        s       = argmin(zip(map(f, sts), sts))
        sts = [si for si in sts if si[0] != s[0]]
        sts += moves([s])
        sold = s
        goal    = list(filter(isGoal, sts))    
        n       = n + len(moves([s]))

    return goal[0], n

def main():
    print (bfs([(s0,[])]))
    print (astar([(s0,[])], f1))
    print (astar([(s0,[])], f2))
    print (astar([(s0,[])], f3))
  
if __name__ == '__main__':
    main()