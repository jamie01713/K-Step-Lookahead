
from policies.agent import iterate_algorithm, parse_history,Random_Agent
from policies.Online_Multiple_Step import LG1T,LG2T,LG1T_RL,LG1_2T
from policies.model_free_average import Optimistic_Q_Learning,MDP_OOMD
from policies.q_learning import Q_learning
from policies.evi import EVI_based
import os
import pickle
from numpy.random import SeedSequence,default_rng
from functools import partial
from itertools import product
import numpy as np
import time
import tqdm
from matplotlib import pyplot as plt
import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
def time_str(sec):
    s = int(sec)
    m = s // 60
    h = m // 60
    m = m % 60
    s = s % 60
    if h: 
        return f"{h}h{m:2d}m{s:2d}s"
    if m:
        return f"{m}m{s:2d}s"
    return f"{s}s"
def main(n_state=10,n_action=5,n_replications=100,n_horizon=200,name_policies={LG1T:{"threshold":0.3}, LG2T:{"threshold": 0.9,"power":1/2}},entropy=243799254704924441050048792905230269161,model_type="continuous"):
    path='./results_random'
    # ensure a path for the results
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    # construct the filename tag
    tag = "__".join(
        [
            
            f"S{n_state}",
       
            f"A{n_action}",
            f"Re{n_replications}",
       
            f"H{n_horizon}",
        ]
    )
    data_pkl = os.path.join(path, f"data__{tag}_stock_new.pkl")
     # create the main seed sequence
    ss = SeedSequence(entropy)  # or however you created "main"
    sq=ss.spawn(n_replications)
 
    learners=[]
    for policy,parameters in name_policies.items():
        #  learners.append(partial(policy,**parameters))
        keys = list(parameters.keys())
        vals = [v if isinstance(v, (list, tuple)) else [v] for v in parameters.values()]

        for combo in product(*vals):
            cfg = dict(zip(keys, combo))
            learners.append(partial(policy, **cfg))
         
    t0=time.time()
    model = gym.make(
    'forex-v0',
    df=FOREX_EURUSD_1H_ASK,
    window_size=10,
    frame_bound=(10, 300),
    unit_side='right'
)
 
    for e in range(n_replications):
        t_spend = time.time() - t0
        t_rem = (n_replications - e) * t_spend/max(e, 1)
        print(f"Run {e+1} ... (spend {time_str(t_spend)}, remains {time_str(t_rem)})")
        # pbar.set_postfix(rep=e+1)
        # model=MDP(n_states=n_state,n_actions=n_action,entropy=sq[e,0])
        # model=RiverSwim(n_state,random=sq[e,0])
        
        # model=gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
        #        enable_wind=False, wind_power=15.0, turbulence_power=1.5)
        # print("\n>>> Initializing learners ...")
        policies=[learner(model,model_type=model_type) for learner in learners]
       

        if e==0:
            results={policy.name():{"average_output": np.zeros((n_horizon,n_replications)), "episode length":np.empty(
    n_replications,dtype=object)
} for policy in policies}
        
             
        for policy in policies:
            policy.reset(model)
            
            name=policy.name()
            s,_=model.reset(seed=int(sq[e].generate_state(1)[0]))
            # s,_=model.reset()
                
                
            history=[[s,False,False]]
            rsum=0
            for _ in tqdm.tqdm(range(n_horizon-1), desc=name):
        #   s,_=model.reset(seed=int(sq[e,run+1].generate_state(1)[0]))
                
                
                    iterate_algorithm(model, policy, history,rsum=int(sq[e].generate_state(1)[0]))
                    
                    
            history.pop()
            info=parse_history(model,history,model_type=model_type)
            le=len(info["cumulative reward"])
            results[name]["average_output"][:,e]=info["cumulative reward"]
            results[name]["episode length"][e]=info["episode length"]
        #   results[name]["history"][run,e]=info["history"]
    with open(data_pkl, "wb") as pkl:
            pickle.dump(results, pkl)
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots()
    legend=[]
    for learner,color in zip(policies,colors):
        name=learner.name() 
        std=np.std(results[name]["average_output"],axis=-1,ddof=1)
        Y=np.mean(results[name]["average_output"],axis=-1)
        line,= ax.plot(Y,color=color,label=name)
        ax.fill_between(np.arange(n_horizon), Y-1.96*std/np.sqrt(n_replications), Y+1.96*std/np.sqrt(n_replications), color=color, alpha=0.2, linewidth=0)
        print(f"name:{name},average leangth:{np.mean([np.mean(x) for x in results[name]["episode length"]])-1.96*np.std([np.mean(x) for x in results[name]["episode length"]],ddof=1)/np.sqrt(n_replications),np.mean([np.mean(x) for x in results[name]["episode length"]])+1.96*np.std([np.mean(x) for x in results[name]["episode length"]],ddof=1)/np.sqrt(n_replications)}")
        

        legend.append((line,name))
    fig.legend(*zip(*legend), loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5,fontsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("test_stock_q_1.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_state",type=int,default=5)
    parser.add_argument("--n_action",type=int,default=2) 
    parser.add_argument("--n_replications",type=int,default=100)      
    parser.add_argument("--n_horizon",type=int,default=20000) 
    parser.add_argument("--name_policies",type=object,default={LG1T:{"threshold":0.3},LG2T:{"threshold":0.9,"power":1/2},LG1_2T:{"threshold":0.9,"threshold_i":0.3,"power":1/2,"cutoff":30},Optimistic_Q_Learning: {"gamma": [0.9,0.99],"T": 20000},Q_learning:{"delta":0.01,"n_horizon": [1,10], "T":20000},MDP_OOMD:{"B":4,"N":2,"T":20000}})     
    parser.add_argument("--model_type",type=str,default="continuous")  
    parser.add_argument("--entropy",type=int,default=243799254704924441050048792905230269161)   
    args = parser.parse_args()
    results = main(**vars(args))  