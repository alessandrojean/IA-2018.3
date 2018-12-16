# Author: Fabrício Olivetti de França

import numpy as np

class Node:
    def __init__(self, state, player, children, wins=0, plays=0, forbidden=False):
        self.state     = state
        self.player    = player
        self.children  = children
        self.wins      = wins
        self.plays     = plays
        self.forbidden = forbidden
        
    def __str__(self):
        return self.state
        
def place2coord(x):
    return (x//3, x%3)
    
def win(b):
    for i in range(3):
       if b[i][0] != ' ' and b[i][0] == b[i][1] and b[i][0] == b[i][2]:
           return True
       elif b[0][i] != ' ' and b[0][i] == b[1][i] and b[0][i] == b[2][i]:
           return True
    if b[0][0] != ' ' and b[0][0] == b[1][1] and b[0][0] == b[2][2]:
        return True
    if b[0][2] != ' ' and b[0][2] == b[1][1] and b[0][2] == b[2][0]:
        return True
    return False
    
def draw(b):
    return len(possibleMoves(b)) == 0 and (not win(b))

def possibleMoves(b):
    flatB = (bij for bi in b for bij in bi)
    return [i for i, bij in enumerate(flatB) if bij == ' ']
    
def nextPlayer(p):
    if p == 'X':
        return 'O'
    return 'X'
    
def move(pos, p, b):
    x, y     = place2coord(pos)
    bt       = [bi.copy() for bi in b]
    bt[x][y] = p
    return bt
    
def confidence(wins, ni, n):
    mu       = wins/ni
    interval = np.sqrt(2*np.log(n)/ni)
    return mu + interval
    
def conf(n, total):
    if n is None or n.forbidden:
        return 1000
    return -confidence(n.wins, n.plays, total)

def maxConfidence(n):
    ns         = n.children
    total      = n.plays
    confidence = [(conf(ni, total), i) for i, ni in enumerate(ns)]
    return sorted(confidence, key=lambda x: x[0])[0][1]

def anyEmpty(n):
    return any( ni is None for ni in n.children )
    
def allForbidden(n):
    return all( bij != ' ' for bi in n.state for bij in bi )
    
def select(root):
    n    = root
    path = []
    while not (anyEmpty(n) or allForbidden(n)):
        idx = maxConfidence(n)
        n   = n.children[idx]
        path.append(idx)
    return n, path
    
def elem(b, i):
    x, y = place2coord(i)
    if b[x][y] == ' ':
        return None
    else:
        return Node( b, 'X', [None for i in range(9)], wins=0, plays=0, forbidden=True )
        
def expansion(n):
    idx    = np.random.choice([i for i, ni in enumerate(n.children) if ni is None])
    new_st = move(idx, n.player, n.state)
    new_n  = Node( new_st, 
                   nextPlayer(n.player),
                   [elem(new_st, i) for i in range(9)], wins=0, plays=0
                 )
    n.children[idx] = new_n
    return new_n, idx
                
def simulation(player, b):
    while not (win(b) or draw(b)):
        b_flat = [bij for bi in b for bij in bi]
        idx    = np.random.choice([i for i, bij in enumerate(b_flat) if bij == ' '])
        b      = move(idx, player, b)
        player = nextPlayer(player)
    
    # did it draw?
    if draw(b):
        return 0
        
    # it's a win
    if player == 'X':
        return -1
    return 1
    
def backpropagation(score, root, path):
    n = root
    update(n, score)
    for p in path:
        n = n.children[p]
        update(n, score)
    
def update(n, score):    
    n.plays += 1
    if score == 0:
        n.wins += 1
    if score == 1 and n.player == 'X':
        n.wins += 1
    if score == -1 and n.player == 'O':
        n.wins += 1
        
def mcts(root, deb=False):
    n, path = select(root)
    if allForbidden(n):
        return
    ni, pi  = expansion(n)
    path.append(pi)
    score   = simulation(ni.player, ni.state)
    backpropagation(score, root, path)
    if deb:
        print(path)
    
def play(root):
    n = root
    while len(possibleMoves(n.state)) > 0 and any([ni is not None for ni in n.children]):
        idx = maxConfidence(n)
        n   = n.children[idx]
        
    return n.state
    
s0 = [[' ' for i in range(3)] for j in range(3)]

emptyChildren = [None for i in range(9)]

root = Node(s0, 'X', emptyChildren, 0, 0)

def main():
    for i in range(100000):
        mcts(root)
    print(root.state)
    print(root.plays, root.wins)
    board = play(root)
    print(board)

    
main()
