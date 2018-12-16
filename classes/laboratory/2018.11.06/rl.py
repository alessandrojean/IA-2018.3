# Author: Fabrício Olivetti de França

from copy import deepcopy
import numpy as np

from collections import defaultdict
from sklearn.linear_model import LinearRegression

from agent import *
from mdp import policyIteration


# Escolhe próximo estado dado uma ação
def performAction(pi, P):
    def nextState(s):
        ps     = P[(s, pi[s])]
        probs  = list(map(lambda x: x[0], ps))
        states = list(map(lambda x: x[1], ps))
        idx    = np.random.choice(len(states), p=probs)

        return states[idx]
    return nextState
    
'''
Algoritmos passivos
'''

#################################
# Estimação direta
def directEst(model, s, goals, R, gamma, nextState, maxLen=100):
    trial = [(s, R[s])]
    while s not in goals:
        s = nextState(s)
        trial.append( (s, R[s]) )
        if len(trial) > maxLen:
            return None
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
#################################

#################################    
# Time difference
def td(percept, state, gamma, alpha):
    sn, rn         = percept
    s, a, r, pi, U = state
    
    if sn not in U:
        U[sn] = rn
    if s != (0,0):
        U[s]  = U[s] + alpha*(r + gamma*U[sn] - U[s])
    return pi[sn], (sn, pi[sn], rn, pi, U)

def runTD(mdp, pi, nTrials, alpha):
    S, A, R, P, gamma = mdp
    U         = {}
    nextState = performAction(pi, P)
    s0        = (1,1)
    goals     = [(4,3), (4,2)]

    for t in range(nTrials):
        # Estado inicial nulo
        state = (0,0), "", 0, pi, U
        s = s0
        percept = (s, R[s])
        g       = gamma
        while state[0] not in goals:
            a, state = td(percept, state, g, alpha)
            s        = nextState(s)
            percept  = (s, R[s])
            
    s, a, r, pi, U = state
    return U 
#################################

#################################
# Q-Learning
def createF(nMax, rPlus):
    def f(q,n):
        if n < nMax:
            return rPlus
        return q
    return f
      
def qlearn(percept, state, goals, A, gamma, alpha, f):
    sn, rn        = percept
    s, a, r, N, Q = state
    
    # se for estado final, usa a recompensa
    if sn in goals:
        Q[(sn, None)] = rn
       
    # se s nao for nulo, atualiza Q
    if s is not None:
        N[(s,a)] = N[(s,a)] + 1
        maxQ     = max(Q[(sn,ai)] for ai in A[sn])
        Q[(s,a)] = Q[(s,a)] + alpha * (r + gamma*maxQ - Q[(s,a)])
    
    # avalia o proximo estado e a acao que deve ser tomada
    if s in goals:
        state = (None, None, 0, N, Q)
    else:
        qvals    = [(f(Q[(sn,ai)], N[(sn,ai)]), ai) for ai in A[sn]]
        an       = max(qvals, key=lambda x: x[0])[1]
        state    = (sn, an, rn, N, Q)
    return an, state

def performQAction(s, a, P):
    ps     = P[(s, a)]
    probs  = list(map(lambda x: x[0], ps))
    states = list(map(lambda x: x[1], ps))
    idx    = np.random.choice(len(states), p=probs)

    return states[idx]
    
def runQ(mdp, nTrials, alpha, nMax, rPlus):
    S, A, R, P, gamma = mdp

    s0        = (1,1)
    goals     = [(4,3), (4,2)]
        
    Q = defaultdict(float)
    N = defaultdict(int)
    f = createF(nMax, rPlus)

    for t in range(nTrials):
        # Estado inicial nulo
        state = None, None, 0, N, Q
        s = s0
        percept = (s, R[s])
        while state[0] not in goals:
            a, state = qlearn(percept, state, goals, A, gamma, alpha, f)
            if s not in goals:
                s   = performQAction(s, a, P)
            percept = (s, R[s])

    # Transforma Q-values em politica
    s, a, r, N, Q = state
    pi            = {}
    for s in S:
        qvals    = [(Q[(s,a)], a) for a in A[s]]
        maxQ, an = max(qvals, key=lambda x: x[0])
        pi[s] = an #.append( (s,an, maxQ) )
    return pi
#################################

'''
Gerando um modelo da função utilidade
'''

#################################
def utilityModel(nTrials, S, R, s, goals, gamma, nextState):
    dataX = []
    dataY = []
    for t in range(nTrials):
        xTrial, yTrial = makeTrial(s, goals, R, gamma, nextState)
        dataX = dataX + xTrial
        dataY = dataY + yTrial

    dataXn = np.zeros((len(dataX), 4))
    dataXn[:, :-1] = dataX
    dataXn[:, -1] = (dataXn[:,0] - 4)**2 + (dataXn[:,1] * 3)**2
    lr = LinearRegression()
    lr.fit(dataXn, dataY)

    model = lambda s: lr.predict([[1, s[0], s[1], (s[0] - 4)**2 + (s[1] - 3)**2]])[0]
    
    print(lr.coef_, lr.intercept_)
    
    return { s : model(s) for s in S }  

def makeTrial(s, goals, R, gamma, nextState):
    trial  = [(s, R[s])]
    xTrial = []
    yTrial = []
    while s not in goals:
        s = nextState(s)
        trial.append( (s, R[s]) )
    for i, (s,r) in enumerate(trial):
        x, y     = s
        u        = sum(r*(gamma**j) 
                        for j, (si, r) in enumerate(trial[i:]))
        xTrial.append([1,x,y])
        yTrial.append(u)
    return xTrial, yTrial

def runUtilityModel(mdp, pi, nTrials):
    S, A, R, P, gamma = mdp
    
    model = defaultdict(lambda: (0.0, 0))
    s0    = (1,1)
    goals = [(4,3), (4,2)]
        
    model = utilityModel(nTrials, S, R, s0, goals, gamma, performAction(pi, P))

    return model    
#################################

'''
Busca por uma política utilizando Hill Climbing
'''
#################################
def hillClimbing(mdp, s0, goals, nTrials):
    S, A, R, P, gamma = mdp
    
    piCurrent = { s : np.random.choice(A[s]) for s in S }

    Ucurrent  = evaluate(mdp, piCurrent, nTrials)
    improved  = True
    
    while improved:
        improved             = False
        bestNeighbor, Uneigh = genNeighbors(mdp, piCurrent, s0, goals, nTrials)
        
        if Uneigh[s0] > Ucurrent[s0]:
            improved  = True
            Ucurrent  = deepcopy(Uneigh)
            piCurrent = deepcopy(bestNeighbor)
    return piCurrent
        
def genNeighbors(mdp, pi, s0, goals, nTrials):
    S, A, R, P, gamma = mdp
    
    candidates = [(s,a) for s in S for a in A[s]
                  if s not in goals and a != pi[s]]
                  
    bestNeighbor = deepcopy(pi)
    Uneigh       = evaluate(mdp, pi, nTrials)
    for s, a in candidates:
       oldAction = pi[s]
       pi[s]     = a
       
       Ucand     = evaluate(mdp, pi, nTrials)
       if Ucand[s0] > Uneigh[s0]:
           Uneigh       = deepcopy(Ucand)
           bestNeighbor = deepcopy(pi)

       pi[s]     = oldAction
       
    return bestNeighbor, Uneigh
    
def evaluate(mdp, pi, nTrials):
    model = runDirectEst(mdp, pi, nTrials)
    if model is None:
        U = { s:-np.inf for s in mdp[0] }
    else:
        U     = {}
        for s, (v, n) in model.items():
            U[s] = v/n
    return U    

#################################

def main():
    mdp = createMDP()
    pi  = policyIteration(mdp)
    
    model = runDirectEst(mdp, pi, 1000)      

    print("Direct Estimation: \n")
    for j in range(3,0,-1):
        for i in range(1,5):
            if (i,j) in model:
                soma, n = model[(i,j)]
                v       = soma / n
                print(f"|  {v:.2f}  |", end="")
            else:
                print("|       |", end="")
        print("")
        
    U = runTD(mdp, pi, 1000, 0.1) 
    
    print("Time Difference: \n")
    for j in range(3,0,-1):
        for i in range(1,5):
            if (i,j) in U:
                v = U[(i,j)]
                print(f"|  {v:.2f}  |", end="")
            else:
                print("|       |", end="")
        print("")
        
    piQ = runQ(mdp, 1000, 0.1, 10, 2)
    
    print("Q-Learning: \n")
    for j in range(3,0,-1):
        for i in range(1,5):
            if (i,j) in piQ:
                v = piQ[(i,j)]
                print(f"|  {v}  |", end="")
            else:
                print("|       |", end="")
        print("")
        
    U = runUtilityModel(mdp, pi, 1000)
    
    print("Utility Model: \n")
    for j in range(3,0,-1):
        for i in range(1,5):
            if (i,j) in U:
                v = U[(i,j)]
                print(f"|  {v:.2f}  |", end="")
            else:
                print("|       |", end="")
        print("")
        
    piHC = hillClimbing(mdp, (1,1), [(4,2), (4,3)], 10)
    
    print("Hill Climbing: \n")
    for j in range(3,0,-1):
        for i in range(1,5):
            if (i,j) in piHC:
                v = piHC[(i,j)]
                print(f"|  {v}  |", end="")
            else:
                print("|       |", end="")
        print("")
main()
