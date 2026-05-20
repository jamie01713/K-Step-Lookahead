import numpy as np
def _binary_search(self, x, low, high, q_a_array,eta):
        tol = 0.0005
        while True:
            lamda = (low+high)/2.0
            y_updated = _update_policy(x, lamda, q_a_array,eta)
            if abs(high-low) < tol:
                return high
            if (y_updated < 0).any():
                low = lamda
            elif sum(y_updated) < 1 - tol:
                high = lamda
            elif sum(y_updated) > 1 + tol:
                low = lamda
            else:
                return lamda

def _update_policy(pi, lamda, q_a_array,A,eta):
        new_pi = np.zeros(A)
        for a in range(A):
            new_pi_reci = 1.0/pi[a] - eta*q_a_array[a] + lamda
            new_pi[a] = 1.0/new_pi_reci
        return new_pi
def Estimate_Q(tau,pi,B,s,N,A):
    i=0
    t1=0
    y=np.zeros(A)
    while t1<=B-N:
        if tau[t1][0]==s:
            i+=1
            R=0
            for t in range(t1,t1+N+1):
                R+=tau[t][2]
            y[tau[t][1]]+=R/pi[s][tau[t][1]]
            t1+=1
        else:
            t1+=1
    if i>0:
        return y/i
    else:
        return 0

def OOMDUPDATE(pi,beta,eta,A,pi0):
    pi_prime_n={k:np.zeros(A) for k in pi}
    pi_n={k:np.zeros(A) for k in pi0}
    for s in beta.keys():
        lamda_prime = _binary_search(pi[s], eta*beta[s].min(), eta*beta[s].max(), beta[s],0.001)
        pi_prime_n[s] = _update_policy(pi[s], lamda_prime, beta[s])

        lamda = _binary_search(pi_prime_n[s], eta*beta[s].min(), eta*beta[s].max(), beta[s],0.001)
        pi_n[s] = _update_policy(pi_prime_n[s], lamda, beta[s])
        pi_n[s] /= pi_n[s].sum()

    return (pi_prime_n,pi_n)
class Optimistic_Q_Learning():
    def __init__(self, model, model_type,gamma=0.99, c=1.0,T=100):
        # _________ constants __________
        self.gamma = gamma
        self.H = gamma/(1.0-gamma)
        self.c = c
        self.T=T
        self.n_actions=model.action_space.n
        self.model_type=model_type
     
        self.reset(model)
        self.set_name(f"Optimistic_Q($\\gamma$={self.gamma})")
    def reset(self, model):
        self.t = 1
        self.n = {}
        self.mu ={}  # Q in the algorithm
        self.mu_hat = {}  # Q_hat in the algorithm
        self.v_hat = {}

        

    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a, r, y,done,truncated):
        if self.model_type=="continuous":
            x=x.tobytes()
            y=y.tobytes()
        if y not in self.n.keys():
            self.n[y]=np.zeros(self.n_actions)
            self.mu[y]=self.H*np.ones(self.n_actions)
            self.mu_hat[y]=self.H*np.ones(self.n_actions)
            self.v_hat[y]=self.H
        self.n[x][a]+=1
        self.t += 1
    
        bonus = 4*self.c * np.sqrt(self.H/self.n[x][a])
        alpha = (self.H + 1)/(self.H + self.n[x][a])
        self.mu[x][a] = (1-alpha)*self.mu[x][a] + alpha*(r + self.gamma*self.v_hat[y] + bonus)
        self.mu_hat[x][a] = min(self.mu_hat[x][a], self.mu[x][a])
        self.v_hat[x] = np.max(self.mu_hat[x])
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.n.keys():
            self.n[x]=np.zeros(self.n_actions)
            self.mu[x]=self.H*np.ones(self.n_actions)
            self.mu_hat[x]=self.H*np.ones(self.n_actions)
            self.v_hat[x]=self.H
        
        candidate=np.where(self.mu_hat[x]==self.mu_hat[x].max())[0]
        return np.random.choice(candidate)
    
class MDP_OOMD():
    def __init__(self,model,B,N,T,model_type):
        self.B=B
        self.N=N
        self.T=T
        self.eta=0.01
        self.n_actions=model.action_space.n
        self.model_type=model_type
        self.reset(model)
        self.set_name(f"MDP_OOMD")
    def reset(self, model):
        self.t = 1
        self.h=0
        self.beta={}
        self.pi={}
        self.pi_prime={}
        self.batch=[]

        

    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a, r, y,done,truncated):
        if self.model_type=="continuous":
            x=x.tobytes()
            y=y.tobytes()
        self.batch.append([x,a,r])
        if self.h==self.B:
            for s in self.beta.keys():
                self.beta[s]=Estimate_Q(tau=self.batch,pi=self.pi,B=self.B,s=s,N=self.N,A=self.n_actions)
            self.pi_prime,self.pi=OOMDUPDATE(pi=self.pi_prime,beta=self.beta,eta=self.eta,pi0=self.pi)
        self.h+=1
        if self.h>=self.B:
            self.h=0
            self.batch=[]
            
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.beta.keys():
            self.beta[x]=np.zeros(self.n_actions)
            self.pi[x]=1/self.n_actions*np.ones(self.n_actions)
            self.pi_prime[x]=1/self.n_actions*np.ones(self.n_actions)
        return np.random.choice(self.n_actions,p=self.pi[x])
    