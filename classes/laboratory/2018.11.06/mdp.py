# Author: Fabrício Olivetti de França

from copy import deepcopy
import numpy as np

from agent import *

def valueIteration(mdp, eps):

    S, A, R, P, gamma = mdp

    # Utilidade inicia começa com tudo 0
    Uprime = { s : 0.0 for s in S }
    
    # delta inicial é o limite + 1 para entrar no laço
    delta  = eps*(1 - gamma)/gamma + 1.

    while delta > eps*(1 - gamma)/gamma:
        U     = deepcopy(Uprime)
        delta = 0
        for s in S:
            if None in A[s]:
                Uprime[s] = R[s]
            else:
                Uprime[s] = R[s] + gamma * max(expVal(P[(s,a)],U) for a in A[s])
            delta     = max(delta, abs(Uprime[s] - U[s]))

    return U

def policyIteration(mdp):
    
    S, A, R, P, gamma = mdp

    U       = { s : 0.0 for s in S }
    # política inicial executa a primeira ação da lista
    pi      = { s : A[s][0] for s in S }
    changed = True

    while changed:
        U       = policyEvaluation(pi, U, mdp)
        changed = False
        for s in S:
            # maior valor esperado utilizando a matriz utilidade atual
            maxU = max(expVal(P[(s,a)],U) for a in A[s])
            # maior valor esperado utilizando a política atual
            piU  = expVal(P[(s,pi[s])],U)
            
            # Se a utilidade ganhar, atualiza a política
            if maxU > piU:
                idx     = np.argmax( [expVal(P[(s,a)],U) for a in A[s]] )
                pi[s]   = A[s][idx]
                changed = True
                #print(s, maxU, piU)
    return pi

def policyEvaluation(pi, U, mdp):
    S, A, R, P, gamma = mdp
    for s in S:
        U[s] = R[s] + gamma * expVal(P[(s, pi[s])],U)
    return U
        
def expVal(ps, U):
    '''
    retorna o valor esperado da utilidade dadas as probabilidades
    de possíveis estados consequentes armazenados em ps.
    '''
    return sum(p*U[s] for p, s in ps)
    
def main():
    mdp = createMDP()
    U  = valueIteration(mdp, 1e-3)
    pi = policyIteration(mdp)

    print("Value Iteration: \n")
    for j in range(3,0,-1):
        for i in range(1,5):
            if (i,j) in U:
                print(f"|  {U[(i,j)]:.2f}  |", end="")
            else:
                print("|       |", end="")
        print("")
        
    print("\n\nPolicy Iteration: \n")
    for j in range(3,0,-1):
        for i in range(1,5):
            if (i,j) in U:
                print(f"|  {pi[(i,j)]}  |", end="")
            else:
                print("|   |", end="")
        print("") 
        
main()
