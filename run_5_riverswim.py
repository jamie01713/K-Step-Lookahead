from environments.mdp_10 import MD_10
from environments.riverswim import RiverSwim
from policies.agent import iterate_algorithm, parse_history,Random_Agent
from policies.Online_Multiple_Step import LG1T,LG2T,LG1_2T,LG1T_RL,LG1TU,LG2TU
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
def main(n_state=10,n_action=5,n_experiments=1000,n_replications_per_experiment=1,n_horizon=20000,name_policies={LG1T:{"threshold":0.3}, LG2T:{"threshold": 0.9,"power":1/2}},entropy=243799254704924441050048792905230269161,model_type="discrete"):
    path='./results_random'
    # ensure a path for the results
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    # construct the filename tag
    tag = "__".join(
        [
            
            f"S{n_state}",
       
            f"A{n_action}",
            f"E{n_experiments}",
            f"Re{n_replications_per_experiment}",
            f"H{n_horizon}",
        ]
    )
    data_pkl = os.path.join(path, f"data__{tag}_riverswim_1_2.pkl")
     # create the main seed sequence
    ss = SeedSequence(entropy)  # or however you created "main"
    children = ss.spawn(n_experiments * (n_replications_per_experiment + 1))

    sq = np.array(children, dtype=object).reshape(
    n_experiments, n_replications_per_experiment + 1
)
 
    learners=[]
    for policy,parameters in name_policies.items():
        #  learners.append(partial(policy,**parameters))
        keys = list(parameters.keys())
        vals = [v if isinstance(v, (list, tuple)) else [v] for v in parameters.values()]

        for combo in product(*vals):
            cfg = dict(zip(keys, combo))
            learners.append(partial(policy, **cfg))
         
    t0=time.time()
    for e in range(n_experiments):
        t_spend = time.time() - t0
        t_rem = (n_experiments - e) * t_spend/max(e, 1)
        print(f"Run {e+1} ... (spend {time_str(t_spend)}, remains {time_str(t_rem)})")
        model=RiverSwim(S=n_state,random=sq[e,0])
        # model=RiverSwim(n_state,random=sq[e,0])
        
        # model=gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
        #        enable_wind=False, wind_power=15.0, turbulence_power=1.5)
        print("\n>>> Initializing learners ...")
        policies=[learner(model,model_type=model_type) for learner in learners]
        print("\n>>> Finished Initialization ...")

        if e==0:
            results={policy.name():{"average_reward": np.zeros((n_horizon,n_replications_per_experiment,n_experiments)), "history":np.empty(
    (n_replications_per_experiment, n_experiments),
    dtype=object
),"time":np.zeros((n_replications_per_experiment,n_experiments))} for policy in policies}
        for run in range(n_replications_per_experiment):
             
             for policy in policies:
                #   s,_=model.reset(seed=int(sq[e,run+1].generate_state(1)[0]))
                  s,_=model.reset(sq[e,run+1])
                  policy.reset(model)
                  name=policy.name()
                  history=[[s,False,False]]
                  rsum=0
                  for _ in tqdm.tqdm(range(n_horizon-1), desc=name):
                    rsum=iterate_algorithm(model, policy, history,rsum)
                  history.pop()
                  info=parse_history(model,history,model_type)
                  results[name]["average_reward"][:,run,e]=info["average expected reward"]
                  results[name]["time"][run,e]=rsum
                #   results[name]["history"][run,e]=info["history"]
    with open(data_pkl, "wb") as pkl:
            pickle.dump(results, pkl)
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots()
    legend=[]
    for learner,color in zip(policies,colors):
        name=learner.name() 
        std=np.std(results[name]["average_reward"],axis=(-2,-1),ddof=1)
        Y=np.mean(results[name]["average_reward"],axis=(-2,-1))
        line,= ax.plot(Y,color=color,label=name)
        ax.fill_between(np.arange(n_horizon), Y-1.96*std/np.sqrt(n_experiments*n_replications_per_experiment), Y+1.96*std/np.sqrt(n_experiments*n_replications_per_experiment), color=color, alpha=0.2, linewidth=0)

        legend.append((line,name))
    fig.legend(*zip(*legend), loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5,fontsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{tag}_riverswim_1_2.pdf", bbox_inches="tight")
    plt.show()
    # for learner in policies:
    #     name=learner.name()
    #     std=np.std(results[name]["time"],axis=(-2,-1),ddof=1)
    #     Y=np.mean(results[name]["time"],axis=(-2,-1))
    #     print(Y-std,Y+std)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_state",type=int,default=5)
    parser.add_argument("--n_action",type=int,default=2) 
    parser.add_argument("--n_experiments",type=int,default=100)
    parser.add_argument("--n_replications_per_experiment",type=int,default=1)      
    parser.add_argument("--n_horizon",type=int,default=20000) 
    parser.add_argument("--name_policies",type=object,default={LG1T:{"threshold":0.3},LG2T:{"threshold": 0.9,"power":1/2},LG1_2T:{"threshold":0.9,"threshold_i":0.3,"power":1/2,"cutoff":100},LG1T_RL:{"threshold":0.3,"cutoff":10000,"config":[EVI_based.prefabs["PMEVI-KLUCRL"]]},EVI_based:{ "config": [EVI_based.prefabs["UCRL2"],EVI_based.prefabs["KLUCRL"],EVI_based.prefabs["PMEVI-KLUCRL"]]},Optimistic_Q_Learning: {"gamma": [0.9,0.99],"T": 20000},Q_learning:{"delta":0.01,"n_horizon": [1,10], "T":20000},MDP_OOMD:{"B":30,"N":10,"T":20000}})   
    # parser.add_argument("--name_policies",type=object,default={LG1T:{"threshold":0.3},LG2T:{"threshold":0.9,"power":1/2},LG1_2T:{"threshold":0.9,"threshold_i":0.3,"power":1/2}})    
    # parser.add_argument("--name_policies",type=object,default={LG2T:{"threshold":0.9,"power":1/2}})    
    # parser.add_argument("--name_policies",type=object,default={LG1T:{"threshold":0.3},LG1_2T:{"threshold":0.9,"threshold_i":0.3,"power":1/2,"cutoff":[10,100,1000,10000]},LG2T:{"threshold":0.9,"power":1/2}})  
    parser.add_argument("--model_type",type=str,default="discrete")  
    parser.add_argument("--entropy",type=int,default=243799254704924441050048792905230269161)   
    args = parser.parse_args()
    results = main(**vars(args))  