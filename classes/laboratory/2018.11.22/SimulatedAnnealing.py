# Author: Fabrício Olivetti de França

import numpy as np

def numero_ataques(sol):
    if len(sol) == 0:
        return 0
    ataques = [s for i, s in enumerate(sol[1:]) 
               if s == sol[0] 
               or s == sol[0] + i + 1 
               or s == sol[0] - i - 1]
    
    return len(ataques) + numero_ataques(sol[1:])

def next_idx(i,j):
    j = j%8 + 1
    i = i if j>1 else i+1
    return i, j
    
def gera_vizinho(v, i, j):
    while v[i] == j:
        i, j = next_idx(i,j)
    
    v[i] = j
    i, j = next_idx(i,j)     
    
    return v, i, j
    
def vizinhanca(sol):
    vizinhos = (sol.copy() for _ in range(8*7))
    i, j = 0, 1
    
    for v in vizinhos:
        vi, i, j = gera_vizinho(v, i, j)
        yield v

def SimulatedAnnealing(s):
    T = 10000
    fs = numero_ataques(s)
    while T > 1e-6:
        vizinhos = [v for v in vizinhanca(s)]
        vi = np.random.choice(len(vizinhos))
        v  = vizinhos[vi]
        fv = numero_ataques(v)

        if fv < fs or np.random.random() <= np.exp( (fs-fv)/T ):
            s, fs = v, fv
        T = 0.95*T
        
    return s

s0 = list(np.random.randint(1,9, size=8))
s = SimulatedAnnealing(s0)       
print(s0, s, numero_ataques(s))
