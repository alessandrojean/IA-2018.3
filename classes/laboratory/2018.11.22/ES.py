# Author: Fabrício Olivetti de França

import numpy as np
from collections import defaultdict
from agent import *

###############################################
mdp = createMDP()

# Escolhe próximo estado dado uma ação
def performAction(pi, P):
    def nextState(s):
        ps     = P[(s, pi[s])]
        probs  = list(map(lambda x: x[0], ps))
        states = list(map(lambda x: x[1], ps))
        idx    = np.random.choice(len(states), p=probs)

        return states[idx]
    return nextState
    
# Estimação direta
def directEst(model, s, goals, R, gamma, nextState, maxLen=20):
    trial = [(s, R[s])]
    while s not in goals:
        s = nextState(s)
        trial.append( (s, R[s]) )
        if len(trial) > maxLen:
            break #return None
    for i, (s, r) in enumerate(trial):
        u        = sum(r*(gamma**j) 
                        for j, (si, r) in enumerate(trial[i:]))
        model[s] = (model[s][0] + u, model[s][1] + 1)
    return model
    
def runDirectEst(mdp, pi, nTrials):
    S, A, R, P, gamma = mdp
    
    model = defaultdict(lambda: (0.0, 0))
    s0    = (1,1)
    goals = [(4,3), (4,2)]
        
    for trials in range(nTrials):
        model = directEst(model, s0, goals, R, gamma, performAction(pi, P))  
        if model is None:
            break

    return model

def evaluate(mdp, pi, nTrials):
    model = runDirectEst(mdp, pi, nTrials)
    if model is None:
        U = { s:-np.inf for s in mdp[0] }
    else:
        U     = {}
        for s, (v, n) in model.items():
            U[s] = v/n
    return U    

def expVal(ps, U):
    '''
    retorna o valor esperado da utilidade dadas as probabilidades
    de possíveis estados consequentes armazenados em ps.
    '''
    return sum(p*U[s] for p, s in ps)
    
def avalia(pi):
    U = evaluate(mdp, pi, 100)
    return U[(1,1)]
    

class Individuo:
    def __init__(self, cromossomo = None):
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        prob = mdp[3]
        def bestAction(s, U):
            return actions[np.argmax([expVal(prob[(s,a)], U)
                           for a in actions])]
                                   
        self.cromossomo = cromossomo
        if cromossomo is None:
            self.cromossomo = 0.5*np.random.randn(12)
            
        self.U  = {(i,j):self.cromossomo[3*i + j - 4] for i in range(1,5) for j in range(1,4)}
        self.pi = {(i,j):bestAction((i,j), self.U) 
                   for i in range(1,5) for j in range(1,4)
                   if (i,j) != (2,2)}
                
        self.fitness = avalia(self.pi)
      
def tournament(P):
    idx1, idx2 = np.random.choice(len(P), 2)
    if P[idx1].fitness > P[idx2].fitness:
        return P[idx1]
    return P[idx2]
    
def seleciona(P, n):
    return [tournament(P) for _ in range(n)]
        
def muta(p):
    cromossomo = p.cromossomo
    idx = np.random.choice(len(cromossomo))
    cromossomo[idx] += 0.1*np.random.randn()
    return Individuo(cromossomo)
        
def cruza(p1, p2):
    cromossomo1 = p1.cromossomo
    cromossomo2 = p2.cromossomo
    alpha = np.random.random()
    return Individuo(alpha*cromossomo1 + (1.0-alpha)*cromossomo2)
    
def cruzamento(P):
    return [cruza(seleciona(P, 1)[0], seleciona(P, 1)[0])
            for _ in range(len(P))]
            
def mutacao(P):
    return [muta(p) for p in P]
            
def melhor(P):
    fitness = [(-p.fitness, i) for i, p in enumerate(P)]
    idx = sorted(fitness)[0][1]
    return P[idx]
    

def ES():
    P = [Individuo() for _ in range(100)]
    for it in range(100):
        filhos = cruzamento(P)
        filhos = mutacao(filhos)
        P = seleciona(P + filhos, len(P))
        print(it, melhor(P).fitness)
    return melhor(P)

p = ES()
print(p.pi, p.fitness)
