from numpy.random import SeedSequence, default_rng,Generator
import numpy as np
from gymnasium import spaces

class MDP_100:

    """ The usual MDP object """

    def __init__(self, n_states=None, n_actions=None, Z=None, prior=1.0,entropy=SeedSequence(243799254704924441050048792905230269161),name="random"):
 


       
        self.set_name(name)
        
        # Initializing shape
        assert((n_states != None and n_actions != None) or Z != None)
        if n_states != None and n_actions != None:
            n_actions = np.array(n_actions)
            if len(n_actions.shape) == 0: # n_actions is vectorialized
                n_actions = [n_actions for _ in range(n_states)]
            elif len (n_actions.shape) == 1:
                pass
            else:
                raise Exception(f"Unvalid 'n_actions': {n_action}")
            self.S = list(range(n_states))
            self.A = [ list(range(n_x)) for n_x in n_actions ] 
            self.Z = set()
            for x in self.S:
                for a in self.A[x]:
                    self.Z.add((x,a))
        elif Z != None:
            self.Z = set(Z.copy())
            self.S = list({s for s, _ in self.Z})
            self.A = [[] for _ in self.S]
            for x, a in self.Z:
                self.A[x].append(a)

        self.n_states = len(self.S)
        self.n_actions = [len(self.A[x]) for x in self.S]
        self.action_space=spaces.Discrete(self.n_actions[0])
    
        # Initializing reward and transition structure
        self.mu=np.zeros((self.n_states,self.n_actions[0]))
        self.p=np.zeros((self.n_states,self.n_actions[0],self.n_states))
      
        sqrew,sqker,sqstate,sqreward=entropy.spawn(4)
        self.reward_seed=sqreward
        self.state_seed=sqstate
        self.randomize_rewards(prior,sqrew)
        self.randomize_kernels(prior,sqker)
    def name(self):
        return self.name_str
    def reset(self,random:Generator):
        self.state=default_rng(random).integers(self.n_states)
        # self.state=0
        return self.state,{}
    def set_name(self, name):
        self.name_str = name

    def __repr__(self):
        tokens = [""]
        tokens.append("Structure:")
        tokens.append(f"| S: {self.n_states}")
        tokens.append(f"| A: {self.n_actions}")

        tokens.append("Reward and transitions:")
        rewards = self.rewards()
        kernels = self.kernels()
        for x in self.S:
            for a in self.A[x]:
                r = round(rewards[x,a], 2)
                k = np.round(np.array(kernels[x, a]), 3)
                tokens.append(f"| r({x},{a}): {r}, p(-|{x},{a}): {k}")

       
        max_len = max(len(token) for token in tokens)
        tokens[0] = "=" * max_len
        tokens.append("=" * max_len)
        return "\n".join(tokens)



    def randomize_rewards(self, prior,random:Generator):
        n_states = self.n_states
        n_actions=self.n_actions[0]
        self.mu=default_rng(random).gamma(0.5,1,size=(n_states,n_actions))
        # mu=default_rng(random).beta(1,1,size=(n_states,n_actions))

       

    def randomize_kernels(self, prior,random:Generator):
        n_states = self.n_states
        n_actions=self.n_actions[0]
        # kernels=default_rng(random).gamma(shape=0.1,scale=10.0,size=(n_states,n_actions,n_states))
        # self.p=default_rng(random).gamma(shape=0.1,scale=10.0,size=(n_states,n_actions,n_states))
        self.p=default_rng(random).gamma(shape=0.01,scale=1000.0,size=(n_states,n_actions,n_states))
        # kernels=default_rng(random).uniform(size=(n_states,n_actions,n_states))
        self.p/=self.p.sum(axis=-1,keepdims=True)
        # print(kernels,self.Z)
        # alpha = [prior for _ in range(self.n_states)]
        
    ### Setters

    def set_reward(self, x, a, mean):
        self.mu[x,a]=mean

    def set_kernel(self, x, a, kernel):
        self.p[x,a]=kernel

    def set_rewards(self, means):
        for x, a in self.Z:
            mean = means[x, a]
            self.set_reward(x, a, mean)
       
    def set_kernels(self, kernels):
        for x, a in self.Z:
            kernel = kernels[x, a]
            self.set_kernel(x, a, kernel)

    ### Getters

    def reward(self, x, a):
        
        return self.mu[x,a]
    
    def rewards(self):
        return { (x, a): self.reward(x, a) for (x, a) in self.Z }

    def kernel(self, x, a):
        
        return self.p[x,a]

    def kernels(self):
        return { (x, a): self.kernel(x, a) for (x, a) in self.Z }
    
    ### Sampler
    def step(self,a):
        x=self.state
        self.state=default_rng(self.state_seed).choice(self.n_states,p=self.p[x,a])
        # self.state=np.random.choice(self.n_states,p=self.p[x,a])
        return self.state, default_rng(self.reward_seed).normal(loc=self.mu[x,a],scale=0.5),False,False,{}
        # default_rng(self.reward_seed).binomial(n=1,p=self.mu[x,a]),np.random.normal(loc=self.mu[x,a],scale=0.5)
        # return self.state,np.random.binomial(n=1,p=self.mu[x,a]),False,False,{}
    