import numpy as np
from policies.agent import *
from policies.evi import EVI_based
def lcb(N):
    Np1=N+1
    return np.sqrt(np.log(1 + Np1) / (1 + Np1))

def ucb(N,T):
    Np1=N+1
    return 2/Np1* 1.714 * np.sqrt((np.log(np.log(Np1 + 1)) + 2 * np.log(T * 10)) / Np1)

class LG2T():
    def __init__(self, model,model_type,threshold,power):
        # super().__init__(model)
        self.threshold=threshold
        self.power=power
        self.model_type=model_type
        self.n_actions=model.action_space.n
        self.reset(model)

        

        self.set_name(f"LG2T($\\gamma$={self.threshold},p={self.power})")
        
        

    def reset(self, model):
        self.t = 1
        self.history=[]

        # Visit count, observations etc.
        self.N = {}
        self.N_={}
        self.rewards = {}
        self.reward={}
        self.reward_immediate={}
        self.ran=0
        self.rsum    = 0.0 
      
      
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a,r,y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        if self.model_type=="continuous":
            x=x.tobytes()
            y=y.tobytes()
        
        self.N[x][a]   += 1
        self.history.append([x,a,r,y])
        

        self.rsum+=r
        self.reward_immediate[x][a]=(self.N[x][a]-1)*self.reward_immediate[x][a] + r
        self.reward_immediate[x][a] /=  self.N[x][a]
        if self.t>=3:
            _x,_a,_r,_=self.history[-2]
            if self.ran==1:
                self.N_[_x][_a]+=1
                self.reward[_x][_a]=(self.N_[_x][_a]-1)*self.reward[_x][_a] +self.reward_immediate[x][a]
                self.reward[_x][_a] /=  self.N_[_x][_a]
        
                self.rewards[_x][_a]=self.reward_immediate[_x][_a]+self.reward[_x][_a]
       
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.N[x]=np.zeros(self.n_actions)
            self.reward[x]=np.zeros(self.n_actions)
            self.reward_immediate[x]=np.zeros(self.n_actions)
            self.rewards[x]=np.zeros(self.n_actions)
            self.N_[x]=np.zeros(self.n_actions)
        lcb_=self.rewards[x]-lcb(self.N[x])-lcb(self.N_[x])
        ucb_=self.rewards[x]+ucb(self.N[x],10000)+ucb(self.N_[x],10000)
        lcb_i=self.reward_immediate[x]-lcb(self.N[x])
        ucb_i=self.reward_immediate[x]+ucb(self.N[x],10000)
        lcb_=np.where(np.minimum(self.N[x],self.N_[x])>0,lcb_,np.inf)
        ucb_=np.where(np.minimum(self.N[x],self.N_[x])>0,ucb_,np.inf)
        lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
        ucb_i=np.where(self.N[x]>0,ucb_i,np.inf)
        
    
        order=np.lexsort((ucb_,np.clip(lcb_, a_min=max(self.threshold,self.threshold),a_max=None)),axis=-1)
        order_i=np.lexsort((ucb_i,np.clip(lcb_i, a_min=0.3,a_max=None)),axis=-1)
        if self.t>=2:
            epsilon=1/((1+(self.N[self.history[-1][0]][self.history[-1][1]]+1)**self.power))
        else:
            epsilon=1
        is_random=np.random.uniform()<epsilon
        if is_random:
            a=order_i[-1]
            self.ran=1
        elif not is_random:
            a=order[-1]
            self.ran=0
        
        return a
class LG2T_I():
    def __init__(self, model,model_type,threshold,power,extent):
        # super().__init__(model)
        self.threshold=threshold
        self.power=power
        self.model_type=model_type
        self.n_actions=model.action_space.n
        self.extent=extent
        self.reset(model)

        

        self.set_name(f"LG2T_I($\\gamma$={self.threshold},$\\Delta$={self.extent})")
        
        

    def reset(self, model):
        self.t = 1
        self.history=[]
        self.thresholds={}

        # Visit count, observations etc.
        self.N = {}
        self.N_threshold={}
        self.N_={}
        self.rewards = {}
        self.reward={}
        self.reward_immediate={}
        self.ran=0
        self.rsum    = 0.0 
      
      
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a,r,y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        if self.model_type=="continuous":
            x=x.tobytes()
            y=y.tobytes()
        
        self.N[x][a]   += 1
        self.N_threshold[x]+=1
        self.history.append([x,a,r,y])
        

        self.rsum+=r
        self.reward_immediate[x][a]=(self.N[x][a]-1)*self.reward_immediate[x][a] + r
        self.reward_immediate[x][a] /=  self.N[x][a]
        if self.t>=3:
            _x,_a,_r,_=self.history[-2]
            if self.ran==1:
                self.N_[_x][_a]+=1
                self.reward[_x][_a]=(self.N_[_x][_a]-1)*self.reward[_x][_a] +self.reward_immediate[x][a]
                self.reward[_x][_a] /=  self.N_[_x][_a]
        
                self.rewards[_x][_a]=self.reward_immediate[_x][_a]+self.reward[_x][_a]
       
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.thresholds[x]=self.threshold
            self.N_threshold[x]=0
            self.N[x]=np.zeros(self.n_actions)
            self.reward[x]=np.zeros(self.n_actions)
            self.reward_immediate[x]=np.zeros(self.n_actions)
            self.rewards[x]=np.zeros(self.n_actions)
            self.N_[x]=np.zeros(self.n_actions)
        lcb_=self.rewards[x]-lcb(self.N[x])-lcb(self.N_[x])
        ucb_=self.rewards[x]+ucb(self.N[x],10000)+ucb(self.N_[x],10000)
        lcb_i=self.reward_immediate[x]-lcb(self.N[x])
        ucb_i=self.reward_immediate[x]+ucb(self.N[x],10000)
        lcb_=np.where(np.minimum(self.N[x],self.N_[x])>0,lcb_,np.inf)
        ucb_=np.where(np.minimum(self.N[x],self.N_[x])>0,ucb_,np.inf)
        lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
        ucb_i=np.where(self.N[x]>0,ucb_i,np.inf)
        if self.N_threshold[x]>=np.log(20000):
            self.thresholds[x]+=self.extent
            self.N_threshold[x]=0
        
        
    
        order=np.lexsort((ucb_,np.clip(lcb_, a_min=max(self.thresholds[x],self.thresholds[x]),a_max=None)),axis=-1)
        order_i=np.lexsort((ucb_i,np.clip(lcb_i, a_min=0.3,a_max=None)),axis=-1)
        if self.t>=2:
            epsilon=1/((1+(self.N[self.history[-1][0]][self.history[-1][1]]+1)**self.power))
        else:
            epsilon=1
        is_random=np.random.uniform()<epsilon
        if is_random:
            a=order_i[-1]
            self.ran=1
        elif not is_random:
            a=order[-1]
            self.ran=0
        
        return a
class LG2TU():
    def __init__(self, model,model_type,threshold,power):
        # super().__init__(model)
        self.threshold=threshold
        self.power=power
        self.model_type=model_type
        self.n_actions=model.action_space.n
        self.reset(model)

        

        self.set_name(f"LG2T_Uniform($\\gamma$={self.threshold},p={self.power})")
        
        

    def reset(self, model):
        self.t = 1
        self.history=[]

        # Visit count, observations etc.
        self.N = {}
        self.N_={}
        self.rewards = {}
        self.reward={}
        self.reward_immediate={}
        self.ran=0
        self.rsum    = 0.0 
      
      
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a,r,y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        if self.model_type=="continuous":
            x=x.tobytes()
            y=y.tobytes()
        
        self.N[x][a]   += 1
        self.history.append([x,a,r,y])
        

        self.rsum+=r
        self.reward_immediate[x][a]=(self.N[x][a]-1)*self.reward_immediate[x][a] + r
        self.reward_immediate[x][a] /=  self.N[x][a]
        if self.t>=3:
            _x,_a,_r,_=self.history[-2]
            if self.ran==1:
                self.N_[_x][_a]+=1
                self.reward[_x][_a]=(self.N_[_x][_a]-1)*self.reward[_x][_a] +self.reward_immediate[x][a]
                self.reward[_x][_a] /=  self.N_[_x][_a]
        
                self.rewards[_x][_a]=self.reward_immediate[_x][_a]+self.reward[_x][_a]
       
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.N[x]=np.zeros(self.n_actions)
            self.reward[x]=np.zeros(self.n_actions)
            self.reward_immediate[x]=np.zeros(self.n_actions)
            self.rewards[x]=np.zeros(self.n_actions)
            self.N_[x]=np.zeros(self.n_actions)
        lcb_=self.rewards[x]-lcb(self.N[x])-lcb(self.N_[x])
        # ucb_=self.rewards[x]+ucb(self.N[x],10000)+ucb(self.N_[x],10000)
        lcb_i=self.reward_immediate[x]-lcb(self.N[x])
        ucb_i=self.reward_immediate[x]+ucb(self.N[x],10000)
        lcb_=np.where(np.minimum(self.N[x],self.N_[x])>0,lcb_,np.inf)
        # ucb_=np.where(np.minimum(self.N[x],self.N_[x])>0,ucb_,np.inf)
        lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
        ucb_i=np.where(self.N[x]>0,ucb_i,np.inf)
        mask=lcb_>=self.threshold
        secondary_key = np.where(mask, lcb_, np.random.random(lcb_.shape))
        primary_key = mask.astype(int)
       
        order=np.lexsort((secondary_key,primary_key),axis=-1)
        
    
        # order=np.lexsort((ucb_,np.clip(lcb_, a_min=max(self.threshold,self.threshold),a_max=None)),axis=-1)
        order_i=np.lexsort((ucb_i,np.clip(lcb_i, a_min=0.3,a_max=None)),axis=-1)
        if self.t>=2:
            epsilon=1/((1+(self.N[self.history[-1][0]][self.history[-1][1]]+1)**self.power))
        else:
            epsilon=1
        is_random=np.random.uniform()<epsilon
        if is_random:
            a=order_i[-1]
            self.ran=1
        elif not is_random:
            a=order[-1]
            self.ran=0
        
        return a
    
class LG1T():
    def __init__(self, model,model_type,threshold,power=1):
        # super().__init__(model)
        self.threshold=threshold
        self.power=power
        self.n_actions=model.action_space.n
        self.model_type=model_type
        self.reset(model)
        self.set_name(f"LG1T($\\gamma$={self.threshold})")
        
        

    def reset(self, model):
        self.t = 1
        self.history=[]

        # Visit count, observations etc.
        self.N = {}
        self.N_={}
        self.rewards = {}
        self.reward={}
        self.reward_immediate={}
        self.rsum    = 0.0 
      
      
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a,r,y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        if self.model_type=="continuous":
    
            x=x.tobytes()
            y=y.tobytes()
        
        self.N[x][a]   += 1
        self.history.append([x,a,r])
        if self.t>=3:
            _x,_a,_r=self.history[-2]

        self.rsum+=r
        self.reward_immediate[x][a]=(self.N[x][a]-1)*self.reward_immediate[x][a] + r
        self.reward_immediate[x][a] /=  self.N[x][a]
       
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.N[x]=np.zeros(self.n_actions)
            self.reward[x]=np.zeros(self.n_actions)
            self.reward_immediate[x]=np.zeros(self.n_actions)
            self.rewards[x]=np.zeros(self.n_actions)
            self.N_[x]=np.zeros(self.n_actions)
        lcb_i=self.reward_immediate[x]-lcb(self.N[x])
        ucb_i=self.reward_immediate[x]+ucb(self.N[x],10000)
        lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
        ucb_i=np.where(self.N[x]>0,ucb_i,np.inf)
        order_i=np.lexsort((ucb_i,np.clip(lcb_i, a_min=max(self.threshold,self.threshold),a_max=None)),axis=-1)
        
       
        

   
        
        return order_i[-1]
class LG1T_I():
    def __init__(self, model,model_type,threshold,power=1,T_max=100,extent=0.1):
        # super().__init__(model)
        self.threshold=threshold
        self.power=power
        self.T=T_max
        self.extent=extent
        self.n_actions=model.action_space.n
        self.model_type=model_type
        self.reset(model)
        self.set_name(f"LG1T_I($\\gamma$={self.threshold},$\\Delta$={self.extent})")
        
        

    def reset(self, model):
        self.t = 1
        self.ti=1
        self.history=[]

        # Visit count, observations etc.
        self.N = {}
        self.N_threshold={}
        self.thresholdx={}
        self.N_={}
        self.rewards = {}
        self.reward={}
        self.reward_immediate={}
        self.rsum    = 0.0 
        self.increase={}
        self.valid_index=np.arange(self.n_actions)
      
      
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a,r,y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        self.N_threshold[x]+=1
        if self.model_type=="continuous":
    
            x=x.tobytes()
            y=y.tobytes()
        
        self.N[x][a]   += 1
        self.history.append([x,a,r])
        if self.t>=3:
            _x,_a,_r=self.history[-2]

        self.rsum+=r
        self.reward_immediate[x][a]=(self.N[x][a]-1)*self.reward_immediate[x][a] + r
        self.reward_immediate[x][a] /=  self.N[x][a]
       
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.thresholdx[x]=self.threshold
            self.N[x]=np.zeros(self.n_actions)
            self.N_threshold[x]=0
            self.reward[x]=np.zeros(self.n_actions)
            self.reward_immediate[x]=np.zeros(self.n_actions)
            self.rewards[x]=np.zeros(self.n_actions)
            self.increase[x]=0
            self.N_[x]=np.zeros(self.n_actions)
        lcb_i=self.reward_immediate[x]-lcb(self.N[x])
        ucb_i=self.reward_immediate[x]+ucb(self.N[x],10000)
        lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
        ucb_i=np.where(self.N[x]>0,ucb_i,np.inf)
        
        
            
        # valid_index=np.flatnonzero(lcb_i>=0.3)
        valid_index=np.arange(self.n_actions)
        if valid_index.size>0 and self.increase[x]==0 and np.sum(self.N[x])>=np.log(self.T):
            self.valid_index=valid_index
            self.thresholdx[x]+=self.extent
            self.increase[x]=1
            self.N_threshold[x]=0
        if valid_index.size>0 and self.increase[x]==1 and self.N_threshold[x]>=np.log(self.T):
            self.valid_index=valid_index
            self.thresholdx[x]+=self.extent
            self.increase[x]=2
            self.N_threshold[x]=0
        if valid_index.size>0 and self.increase[x]==2 and self.N_threshold[x]>=np.log(self.T):
            self.valid_index=valid_index
            self.thresholdx[x]+=self.extent
            self.increase[x]=3
            self.N_threshold[x]=0
        if valid_index.size>0 and self.increase[x]==3 and self.N_threshold[x]>=np.log(self.T):
            self.valid_index=valid_index
            self.thresholdx[x]+=self.extent
            self.increase[x]=4
            self.N_threshold[x]=0
        if valid_index.size>0 and self.increase[x]==4 and self.N_threshold[x]>=np.log(self.T):
            self.valid_index=valid_index
            self.thresholdx[x]+=self.extent
            self.increase[x]=5
            self.N_threshold[x]=0
        if valid_index.size>0 and self.increase[x]==5 and self.N_threshold[x]>=np.log(self.T):
            self.valid_index=valid_index
            self.thresholdx[x]+=self.extent
            self.increase[x]=6
            self.N_threshold[x]=0
        if valid_index.size>0 and self.increase[x]==6 and self.N_threshold[x]>=np.log(self.T):
            self.valid_index=valid_index
            self.thresholdx[x]+=self.extent
            self.increase[x]=7
            self.N_threshold[x]=0
        if valid_index.size>0 and self.increase[x]==7 and self.N_threshold[x]>=np.log(self.T):
            self.valid_index=valid_index
            self.thresholdx[x]+=self.extent
            self.increase[x]=8
            self.N_threshold[x]=0
        # if valid_index.size>0 and self.increase==2 and self.N_threshold[x]>=np.sqrt(self.T)/5:
        #     self.valid_index=valid_index
        #     self.thresholdx[x]+=1.6
        #     self.increase=3
        #     self.N_threshold[x]=0
        # if self.t>=3000:
        #     self.valid_index=np.flatnonzero(lcb_i>=0.3)
        #     if self.valid_index.size==0:
        #         self.valid_index=np.arange(self.n_actions)
        #     self.thresholdx[x]+=1.6
        
        
        # if np.sqrt(self.T)+1>=np.sum(self.N[x])>=np.sqrt(self.T):
        #     keep_mask=lcb_i>=self.thresholdx[x]
        #     valid_index=np.flatnonzero(keep_mask)
        #     if valid_index.size>0:
        #         self.valid_index=valid_index
        #     self.thresholdx[x]+=0.3
        # if 2*np.log(self.T)+1>=np.sum(self.N[x])>2*np.log(self.T):
        #     keep_mask=lcb_i>=self.thresholdx[x]
        #     self.valid_index=np.flatnonzero(keep_mask)
        #     if self.valid_index.size==0:
        #         keep_mask=lcb_i>=self.thresholdx[x]-1000
        #         self.valid_index=np.flatnonzero(keep_mask)
        #     self.thresholdx[x]+=0.3
        # if 3*np.log(self.T)+1>=np.sum(self.N[x])>3*np.log(self.T):
        #     keep_mask=lcb_i>=self.thresholdx[x]
        #     self.valid_index=np.flatnonzero(keep_mask)
        #     if self.valid_index.size==0:
        #         keep_mask=lcb_i>=self.thresholdx[x]-1000
        #         self.valid_index=np.flatnonzero(keep_mask)
        #     self.thresholdx[x]+=0.3
        order_i=self.valid_index[np.lexsort((ucb_i[self.valid_index],np.clip(lcb_i[self.valid_index], a_min=self.thresholdx[x],a_max=None)),axis=-1)]
        
       
        

   
        
        return order_i[-1]

class LG1TU():
    def __init__(self, model,model_type,threshold,power=1):
        # super().__init__(model)
        self.threshold=threshold
        self.power=power
        self.n_actions=model.action_space.n
        self.model_type=model_type
        self.reset(model)
        self.set_name(f"LG1T_Uniform($\\gamma$={self.threshold})")
        
        

    def reset(self, model):
        self.t = 1
        self.history=[]

        # Visit count, observations etc.
        self.N = {}
        self.N_={}
        self.rewards = {}
        self.reward={}
        self.reward_immediate={}
        self.rsum    = 0.0 
      
      
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a,r,y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        if self.model_type=="continuous":
    
            x=x.tobytes()
            y=y.tobytes()
        
        self.N[x][a]   += 1
        self.history.append([x,a,r])
        if self.t>=3:
            _x,_a,_r=self.history[-2]

        self.rsum+=r
        self.reward_immediate[x][a]=(self.N[x][a]-1)*self.reward_immediate[x][a] + r
        self.reward_immediate[x][a] /=  self.N[x][a]
       
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.N[x]=np.zeros(self.n_actions)
            self.reward[x]=np.zeros(self.n_actions)
            self.reward_immediate[x]=np.zeros(self.n_actions)
            self.rewards[x]=np.zeros(self.n_actions)
            self.N_[x]=np.zeros(self.n_actions)
        lcb_i=self.reward_immediate[x]-lcb(self.N[x])
       
        lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
        mask=lcb_i>=self.threshold
        secondary_key = np.where(mask, lcb_i, np.random.random(lcb_i.shape))
        primary_key = mask.astype(int)
       
        order_i=np.lexsort((secondary_key,primary_key),axis=-1)
        
       
        

   
        
        return order_i[-1]
class LG1_2T_Adaptive():
    def __init__(self, model,model_type,threshold,threshold_i,power,cutoff=100):
        # super().__init__(model)
        self.threshold=threshold
        self.threshold_i=threshold_i
        self.power=power
        self.n_actions=model.action_space.n
        self.cutoff=cutoff
        self.model_type=model_type
        self.reset(model)

        

        self.set_name(f"LG1-2T_Adaptive($\\gamma_1$={self.threshold_i},$\\gamma_2$={self.threshold},p={self.power},cutoff={self.cutoff})")
        
        

    def reset(self, model):
        self.t = 1
        self.history=[]

        # Visit count, observations etc.
        self.step={}
        self.N = {}
        self.N_={}
        self.rewards = {}
        self.N2={}
        self.reward={}
        self.reward_immediate={}
        self.ran=0
        self.rsum    = 0.0 
      
      
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a,r,y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        if self.model_type=="continuous":

            x=x.tobytes()
            y=y.tobytes()
        self.N[x][a]   += 1
        self.history.append([x,a,r,y])
        

        self.rsum+=r
        self.reward_immediate[x][a]=(self.N[x][a]-1)*self.reward_immediate[x][a] + r
        self.reward_immediate[x][a] /=  self.N[x][a]
        if self.step==1:
            self.N2[x][a]+=1
        if self.step==0 or self.ran==1:
            
            _x,_a,_r,_=self.history[-2]
           
            self.N_[_x][_a]+=1
            self.reward[_x][_a]=(self.N_[_x][_a]-1)*self.reward[_x][_a] +self.reward_immediate[x][a]
            self.reward[_x][_a] /=  self.N_[_x][_a]
            self.rewards[_x][_a]=self.reward_immediate[_x][_a]+self.reward[_x][_a]
        # elif self.t>=3:
        #     _x,_a,_r,_=self.history[-2]
        #     self.N_[_x][_a]+=1
        #     self.reward[_x][_a]=(self.N_[_x][_a]-1)*self.reward[_x][_a] +self.reward_immediate[x][a]
        #     self.reward[_x][_a] /=  self.N_[_x][_a]
        #     self.rewards[_x][_a]=self.reward_immediate[_x][_a]+self.reward[_x][_a]
       
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.N[x]=np.zeros(self.n_actions)
            self.reward[x]=np.zeros(self.n_actions)
            self.reward_immediate[x]=np.zeros(self.n_actions)
            self.rewards[x]=np.zeros(self.n_actions)
            self.N_[x]=np.zeros(self.n_actions)
            self.N2[x]=np.zeros(self.n_actions)
            self.step[x]=0
        
        lcb_i=self.reward_immediate[x]-lcb(self.N[x])
        ucb_i=self.reward_immediate[x]+ucb(self.N[x],10000)
        
        lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
        ucb_i=np.where(self.N[x]>0,ucb_i,np.inf)
        
        order_i=np.lexsort((ucb_i,np.clip(lcb_i, a_min=self.threshold_i,a_max=None)),axis=-1)
        # if np.sum(self.N[x])>=self.cutoff*np.sqrt(20000):
        #     self.step[x]=1
        if self.t>=np.sqrt(100*25*20000):
            self.step.update({key: 1 for key in self.step})
        if self.t>=2 and self.step[self.history[-1][0]]==1 and self.step[x]==1:
            lcb_=self.rewards[x]-lcb(self.N_[x])
            ucb_=self.rewards[x]+ucb(self.N[x],10000)+ucb(self.N_[x],10000)
            lcb_=np.where(np.minimum(self.N[x],self.N_[x])>0,lcb_,np.inf)
            ucb_=np.where(np.minimum(self.N[x],self.N_[x])>0,ucb_,np.inf)


            order=np.lexsort((ucb_,np.clip(lcb_, a_min=max(self.threshold,self.threshold),a_max=None)),axis=-1)
            epsilon=1/((1+(self.N[self.history[-1][0]][self.history[-1][1]]+1)**self.power))
            is_random=np.random.uniform()<epsilon
            if is_random:
                a=order_i[-1]
                self.ran=1
            elif not is_random:
                a=order[-1]
                self.ran=0
        # elif self.step[x]==1:
        #     lcb_=self.rewards[x]-lcb(self.N_[x])
        #     ucb_=self.rewards[x]+ucb(self.N[x],10000)+ucb(self.N_[x],10000)
        #     lcb_=np.where(np.minimum(self.N[x],self.N_[x])>0,lcb_,-np.inf)
        #     ucb_=np.where(np.minimum(self.N[x],self.N_[x])>0,ucb_,np.inf)
        #     order=np.lexsort((ucb_,np.clip(lcb_, a_min=max(self.threshold,self.threshold),a_max=None)),axis=-1)
        #     a=order[-1]
        #     self.ran=0
    
        else:
            a=order_i[-1]

        
        # if self.t>=self.cutoff:
        #     lcb_=self.rewards[x]-lcb(self.N[x])-lcb(self.N_[x])
        #     ucb_=self.rewards[x]+ucb(self.N[x],10000)+ucb(self.N_[x],10000)
        #     lcb_=np.where(np.minimum(self.N[x],self.N_[x])>0,lcb_,np.inf)
        #     ucb_=np.where(np.minimum(self.N[x],self.N_[x])>0,ucb_,np.inf)


        #     order=np.lexsort((ucb_,np.clip(lcb_, a_min=max(self.threshold,self.threshold),a_max=None)),axis=-1)
        
        
        #     if self.t>=2:
        #         epsilon=1/((1+(self.N2[self.history[-1][0]][self.history[-1][1]]+1)**self.power))
        #     else:
        #         epsilon=1
        #     is_random=np.random.uniform()<epsilon
        #     if is_random:
        #         a=order_i[-1]
        #         self.ran=1
        #     elif not is_random:
        #         a=order[-1]
        #         self.ran=0
        # else:
        #     a=order_i[-1]
        
        return a
    
class LG1_2T():
    def __init__(self, model,model_type,threshold,threshold_i,power,cutoff=100):
        # super().__init__(model)
        self.threshold=threshold
        self.threshold_i=threshold_i
        self.power=power
        self.n_actions=model.action_space.n
        self.cutoff=cutoff
        self.model_type=model_type
        self.reset(model)

        

        self.set_name(f"LG1-2T($\\gamma_1$={self.threshold_i},$\\gamma_2$={self.threshold},p={self.power},cutoff={self.cutoff}")
        
        

    def reset(self, model):
        self.t = 1
        self.history=[]

        # Visit count, observations etc.
        self.N = {}
        self.N_={}
        self.rewards = {}
        self.N2={}
        self.reward={}
        self.reward_immediate={}
        self.ran=0
        self.rsum    = 0.0 
      
      
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a,r,y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        if self.model_type=="continuous":

            x=x.tobytes()
            y=y.tobytes()
        self.N[x][a]   += 1
        self.history.append([x,a,r,y])
        

        self.rsum+=r
        self.reward_immediate[x][a]=(self.N[x][a]-1)*self.reward_immediate[x][a] + r
        self.reward_immediate[x][a] /=  self.N[x][a]
        if self.t>=self.cutoff:
            self.N2[x][a]+=1
            _x,_a,_r,_=self.history[-2]
            if self.ran==1:
                self.N_[_x][_a]+=1
                self.reward[_x][_a]=(self.N_[_x][_a]-1)*self.reward[_x][_a] +self.reward_immediate[x][a]
                self.reward[_x][_a] /=  self.N_[_x][_a]
            self.rewards[_x][_a]=self.reward_immediate[_x][_a]+self.reward[_x][_a]
        elif self.t>=3:
            _x,_a,_r,_=self.history[-2]
            self.N_[_x][_a]+=1
            self.reward[_x][_a]=(self.N_[_x][_a]-1)*self.reward[_x][_a] +self.reward_immediate[x][a]
            self.reward[_x][_a] /=  self.N_[_x][_a]
            self.rewards[_x][_a]=self.reward_immediate[_x][_a]+self.reward[_x][_a]
       
    def act(self,x):
        if self.model_type=="continuous":
            x=x.tobytes()
        if x not in self.N.keys():
            self.N[x]=np.zeros(self.n_actions)
            self.reward[x]=np.zeros(self.n_actions)
            self.reward_immediate[x]=np.zeros(self.n_actions)
            self.rewards[x]=np.zeros(self.n_actions)
            self.N_[x]=np.zeros(self.n_actions)
            self.N2[x]=np.zeros(self.n_actions)
        
        lcb_i=self.reward_immediate[x]-lcb(self.N[x])
        ucb_i=self.reward_immediate[x]+ucb(self.N[x],10000)
        
        lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
        ucb_i=np.where(self.N[x]>0,ucb_i,np.inf)
        
        order_i=np.lexsort((ucb_i,np.clip(lcb_i, a_min=self.threshold_i,a_max=None)),axis=-1)
        
        if self.t>=self.cutoff:
            lcb_=self.rewards[x]-lcb(self.N[x])-lcb(self.N_[x])
            ucb_=self.rewards[x]+ucb(self.N[x],10000)+ucb(self.N_[x],10000)
            lcb_=np.where(np.minimum(self.N[x],self.N_[x])>0,lcb_,np.inf)
            ucb_=np.where(np.minimum(self.N[x],self.N_[x])>0,ucb_,np.inf)


            order=np.lexsort((ucb_,np.clip(lcb_, a_min=max(self.threshold,self.threshold),a_max=None)),axis=-1)
        
        
            if self.t>=2:
                epsilon=1/((1+(self.N2[self.history[-1][0]][self.history[-1][1]]+1)**self.power))
            else:
                epsilon=1
            is_random=np.random.uniform()<epsilon
            if is_random:
                a=order_i[-1]
                self.ran=1
            elif not is_random:
                a=order[-1]
                self.ran=0
        else:
            a=order_i[-1]
        
        return a
    
class LG1T_RL(Agent):
    def __init__(self, model,threshold,cutoff,model_type,config):
        super().__init__(model)
        self.threshold=np.zeros(self.n_states)
        for i in range(self.n_states):
            self.threshold[i]=threshold
        self.threshold_t=self.threshold.copy()
        self.learner_alt=EVI_based(model, config=config,model_type=model_type)
        self.learner_alt.reset(model)
        self.learner_alt_name=self.learner_alt.name()
        self.reset(model)
        self.model_name=model.name()
        self.cutoff=cutoff

        self.set_name(f"LG1T-RL($\\gamma$={self.threshold[0]},threshold={self.cutoff},RL={self.learner_alt_name})")

    def reset(self, model):
        self.t = 1
        self.learner_alt.reset(model)
        # Visit count, observations etc.
        self.N = np.zeros((self.n_states,self.n_actions[0]))
        self.Nr=np.zeros((self.n_states,self.n_actions[0]))
        self.rewards = np.zeros((self.n_states,self.n_actions[0]))
        self.reward_immediate=np.zeros((self.n_states,self.n_actions[0]))
        self.rsum    = []
        self.threshold_t=self.threshold.copy()
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self, x, a, r, y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        self.learner_alt.observe(x,a,r,y,done,truncated)
        if self.t<=2:
            self.rsum.append(r)
        else:
            self.rsum.append(r+self.rsum[-1])
        self.N[x,a]   += 1
        if self.t>=self.cutoff:
            self.Nr[x,a]+=1
        self.reward_immediate[x,a]=(self.N[x,a]-1)*self.reward_immediate[x,a] + r
        self.reward_immediate[x,a] /=  self.N[x,a]
    def act(self,x):
            lcb_i=self.reward_immediate[x]-lcb(self.N[x])
            ucb_i=self.reward_immediate[x]+ucb(self.N[x],10000)
            lcb_i=np.where(self.N[x]>0,lcb_i,np.inf)
            ucb_i=np.where(self.N[x]>0,ucb_i,np.inf)
            order_i=np.lexsort((ucb_i,np.clip(lcb_i, a_min=max(self.threshold_t[x],self.threshold_t[x]),a_max=None)),axis=-1)
            if self.t>=self.cutoff:
          
             

        
        
                a=self.learner_alt.act(x)
           
                  
            else:
                a=order_i[-1]
       
            return a

class Multi_Step():
    def __init__(self,model,step):
        self.n_states=model.n_states
        self.n_actions=model.action_space.n
        self.rewards=np.zeros((self.n_states,self.n_actions))
        self.kernels=np.zeros((self.n_states,self.n_actions,self.n_states))
        rewards=model.rewards().copy()
        kernels=model.kernels().copy()
        for x,a in model.Z:
            self.rewards[x,a]=rewards[x,a]
            self.kernels[x,a]=kernels[x,a]
        rtemp=np.zeros((step+1,self.n_states,self.n_actions))
        rtemp[0,:,:]=self.rewards.copy()
        for l in range(step):
            r_=np.max(rtemp[l,:,:],axis=-1)
            rtemp[l+1,:,:]=self.rewards+np.einsum("sax,x->sa",self.kernels,r_)
        self.reward_greedy=rtemp[step,:,:]
        self.set_name(f"{step+1}-Step")
    def set_name(self,name):
        self.name=name
    def name(self):
        return self.name
    def reset(self,model):
        pass
    def observe(self,x,a,r,y,truncated):
        pass
    def act(self,x):
        return np.argmax(self.reward_greedy[x])

    




 