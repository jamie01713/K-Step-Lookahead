import numpy as np
from policies.agent import *
def bonus(N,H,S,A,P,T):
    return np.sqrt(H**3*np.log(S*A*T/P)/N)
def bonus_(sigma,sigma_in,H,N,N_in,mu,mu_in,P):
    return np.sqrt((sigma/N-(mu/N)**2)/N*np.log(2/P))+np.sqrt((sigma_in/N_in-(mu_in/N_in)**2)/N*np.log(2/P))+(H*np.log(2/P)/N+H*np.log(2/P)/N_in+H*(np.log(2/P))**(3/4)/(N**(3/4))+H*(np.log(2/P))**(3/4)/(N_in**(3/4)))
def bonus_bar(N_in,H,P):
    return 2*np.sqrt(H**2/N_in*np.log(2/P))
def sigma(H,S,A,k,N):
    return H*np.sqrt(np.log(S*A*(k**2)*H*2)/(N+1))+H/(N+1)

class Q_learning():
    def __init__(self, model,model_type,n_horizon,delta,T):
        self.H=n_horizon
        self.P=delta
        self.T=T
        self.model_type=model_type
        self.n_actions=model.action_space.n
        self.reset(model)
        self.set_name(f"Q_learning($H$={self.H},$\\delta$={self.P})")
    def reset(self, model):
        self.t = 1
        self.h=0
        self.Q={}
        self.V={}
        self.N={}
        # self.Q=np.zeros((self.H,self.n_states,self.n_actions[0]))
        # self.Q+=self.H
        # self.V=np.zeros((self.H+1,self.n_states))
        # self.V+=self.H
        # self.V[self.H,:]=0
        # self.N=np.zeros((self.H,self.n_states,self.n_actions[0]))

    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a, r, y,done,truncated):
        if self.model_type=="continuous":
            x=x.tobytes()
            y=y.tobytes()
        if y not in self.N.keys():
            self.N[y]=np.zeros((self.H,self.n_actions))
            self.Q[y]=self.H*np.ones((self.H,self.n_actions))
            self.V[y]=self.H*np.ones(self.H+1)
            self.V[y][self.H]=0

        self.N[x][self.h,a]+=1
        self.t+=1
       
        self.Q[x][self.h,a]=(1-(self.H+1)/(self.H+self.N[x][self.h,a]))*self.Q[x][self.h,a]+(self.H+1)/(self.H+self.N[x][self.h,a])*(r+self.V[y][self.h+1]+bonus(self.N[x][self.h,a],self.H,100000,self.n_actions,self.P,self.T))
        self.V[x][self.h]=min(self.H,np.max(self.Q[x][self.h],axis=-1))
        self.h+=1
        if self.h>=self.H:
            self.h=0
        
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.N[x]=np.zeros((self.H,self.n_actions))
            self.Q[x]=self.H*np.ones((self.H,self.n_actions))
            self.V[x]=self.H*np.ones(self.H+1)
            self.V[x][self.H]=0
        candidate=np.where(self.Q[x][self.h]==self.Q[x][self.h].max())[0]



        return np.random.choice(candidate)
    
class UCB_Advantage(Agent):
    def __init__(self, model,n_horizon,delta,T):
        super().__init__(model)
        self.H=n_horizon
        self.P=delta
        self.T=T
        self.N0=self.n_states*self.n_actions[0]*self.H**6*np.log(2/self.P)
        self.episode=[self.H]
        e=self.H
        while self.episode[-1]<self.T:
            self.episode.append(int(self.episode[-1]+np.floor((1+1/self.H)*e)))
            e=np.floor((1+1/self.H)*e)

        # self.K=T/n_horizon
        self.reset(model)
        self.set_name(f"UCB_Advantage($H$={self.H},$\delta$={self.P})")
    def reset(self, model):
        self.t = 1
        self.h=0
        self.Q=np.zeros((self.H,self.n_states,self.n_actions[0]))
        self.V=np.zeros((self.H+1,self.n_states))
        self.V_ref=np.zeros((self.H+1,self.n_states))
        self.V_ref+=self.H
        self.V[self.H,:]=0
        for h in range(self.H):
            self.Q[h,:,:]=self.H-h
            self.V[h,:]=self.H-h
        
        self.N=np.zeros((self.H,self.n_states,self.n_actions[0]))
        self.N_in=np.zeros((self.H,self.n_states,self.n_actions[0]))
        self.v_in=np.zeros((self.H,self.n_states,self.n_actions[0]))
        self.sigma_in=np.zeros((self.H,self.n_states,self.n_actions[0]))
        self.mu_in=np.zeros((self.H,self.n_states,self.n_actions[0]))
        self.mu_ref=np.zeros((self.H,self.n_states,self.n_actions[0]))
        self.sigma_ref=np.zeros((self.H,self.n_states,self.n_actions[0]))

    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a, r, y,done,truncated):
        self.N[self.h,x,a]+=1
        self.N_in[self.h,x,a]+=1
        self.mu_in[self.h,x,a]+=self.V[self.h+1,y]-self.V_ref[self.h+1,y]
        self.v_in[self.h,x,a]+=self.V[self.h+1,y]
        self.sigma_in[self.h,x,a]=(self.V[self.h+1,y]-self.V_ref[self.h+1,y])**2
        self.mu_ref[self.h,x,a]+=self.V_ref[self.h+1,y]
        self.sigma_ref[self.h,x,a]+=(self.V_ref[self.h+1,y])**2
        self.t+=1
        if self.N[self.h,x,a] in self.episode:
            self.Q[self.h,x,a]=min(r+self.v_in[self.h,x,a]/self.N_in[self.h,x,a]+bonus_bar(N_in=self.N_in[self.h,x,a],H=self.H,P=self.P),r+self.mu_ref[self.h,x,a]/self.N[self.h,x,a]+self.mu_in[self.h,x,a]/self.N_in[self.h,x,a]+bonus_(sigma=self.sigma_ref[self.h,x,a],sigma_in=self.sigma_in[self.h,x,a],H=self.H,N=self.N[self.h,x,a],N_in=self.N_in[self.h,x,a],mu=self.mu_ref[self.h,x,a],mu_in=self.mu_in[self.h,x,a],P=self.P),self.Q[self.h,x,a])
            self.V[self.h,x]=np.max(self.Q[self.h,x,:],axis=-1)
            self.N_in[self.h,x,a]=0
            self.mu_in[self.h,x,a]=0
            self.sigma_in[self.h,x,a]=0
            self.v_in[self.h,x,a]=0
        if np.sum(self.N[self.h,x,:])==self.N0:
            self.V_ref[self.h,x]=self.V[self.h,x]
        self.h+=1
        if self.h>=self.H:
            self.h=0
        
    def act(self,x):
        return np.argmax(self.Q[self.h,x,:],axis=-1)
    
class SSR(Agent):
    def __init__(self, model,n_horizon,T):
        super().__init__(model)
        self.H=n_horizon
        self.T=T
        # self.K=T/n_horizon
        self.reset(model)
        self.set_name(f"SSR($H$={self.H})")
    def reset(self, model):
        self.t = 1
        self.k=1
        self.h=0
        self.Q=np.zeros((self.H,self.n_states,self.n_actions[0]))
        
        self.V=np.zeros((self.H+1,self.n_states))
        self.Nh=np.zeros((self.H,self.n_states,self.n_actions[0]))
        self.N=np.zeros((self.n_states,self.n_actions[0]))
        self.R=np.zeros((self.n_states,self.n_actions[0]))
        self.P=np.zeros((self.n_states,self.n_actions[0],self.n_states))
        self.N_=np.zeros((self.n_states,self.n_actions[0],self.n_states))

    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a, r, y,done,truncated):
        if self.h==0:
            self.Q=np.zeros((self.H,self.n_states,self.n_actions[0]))
        
            self.V=np.zeros((self.H+1,self.n_states))
            z=np.random.normal(loc=0,scale=1)
            for h in reversed(range(self.H)):
                self.Q[h,:,:]=self.R+np.einsum("sax,x->sa",self.P,self.V[h+1,:])+z*sigma(H=self.H,S=self.n_states,A=self.n_actions[0],k=self.k,N=self.Nh[h,:,:])/1000
                self.V[h,:]=np.clip(np.max(self.Q[h,:,:],axis=-1),0,2*(self.H-h+1))
        self.R[x,a]=(self.R[x,a]*self.N[x,a]+r)/(self.N[x,a]+1)
        self.P[x,a,y]=(self.N_[x,a,y]+1)/(self.N[x,a]+1)
        self.N_[x,a,y]+=1
        self.N[x,a]+=1
        self.Nh[self.h,x,a]+=1
        self.h+=1
        if self.h>=self.H:
            self.h=0
            self.k+=1
    def act(self,x):
        return np.argmax(self.Q[self.h,x,:],axis=-1)
