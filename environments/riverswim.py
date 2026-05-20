import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from numpy.random import Generator,SeedSequence

from environments.mdp_riverswim import MDP
def RiverSwim(S,random:Generator):
    model = MDP(S, 2,entropy=random)
    p=np.zeros((2,S,S))
    e=1e-2/(S)
    p[0,0]=[1-(S-1)*e]+[e for x in range(1,S)]
    p[1,0]=[0.7+e,0.3-(S-1)*e]+[e for x in range(2,S)]
    p[1,S-1]=[e for x in range(S-2)]+[0.7+e,0.3-(S-1)*e]
    for x in range(1,S-1):
        p[1,x,:]=e
        p[1,x,x-1]=0.1+e
        p[1,x,x]=0.6+e
        p[1,x,x+1]=0.3-(S-1)*e
        p[0,x,:]=e
        p[0,x,x-1]=1-(S-1)*e
    p[0,S-1,:]=e
    p[0,S-1,S-2]=1-(S-1)*e
    for x in range(S):
        model.set_kernel(x,0,p[0,x])
        model.set_kernel(x,1,p[1,x])

    # rewards
    for x in range(S):
        model.set_reward(x,  0, 0)
        model.set_reward(x, 1, 0)
    model.set_reward(  0,  0, 0.2)
    model.set_reward(S-1, 1, 1)
    
    # Done.
    return model


