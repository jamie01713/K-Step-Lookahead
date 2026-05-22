from concurrent.futures import ProcessPoolExecutor
from itertools import product
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import SeedSequence

from environments.mdp_100 import MDP_100
from policies.Online_Multiple_Step import LG1T, LG2T, LG1_2T_Adaptive
from policies.agent import iterate_algorithm, parse_history


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


def build_learners(name_policies):
    learners = []
    for policy, parameters in name_policies.items():
        keys = list(parameters.keys())
        vals = [v if isinstance(v, (list, tuple)) else [v] for v in parameters.values()]
        for combo in product(*vals):
            learners.append((policy, dict(zip(keys, combo))))
    return learners


def policy_names(learners, n_state, n_action, model_type):
    model = MDP_100(n_states=n_state, n_actions=n_action, entropy=SeedSequence(0))
    return [policy(model, model_type=model_type, **cfg).name() for policy, cfg in learners]


def run_single_replication(args):
    n_state, n_action, n_horizon, learners, model_type, env_seed, run_seed, worker_seed = args
    np.random.seed(int(worker_seed.generate_state(1)[0]))
    model = MDP_100(n_states=n_state, n_actions=n_action, entropy=env_seed)
    run_result = {}
    for policy_cls, cfg in learners:
        policy = policy_cls(model, model_type=model_type, **cfg)
        s, _ = model.reset(run_seed)
        policy.reset(model)
        history = [[s, False, False]]
        for _ in range(n_horizon - 1):
            iterate_algorithm(model, policy, history, 0)
        history.pop()
        info = parse_history(model, history, model_type=model_type)
        run_result[policy.name()] = {"average_reward": np.asarray(info["average expected reward"])}
    return run_result


def main(
    n_state=100,
    n_action=25,
    n_experiments=1000,
    n_replications_per_experiment=1,
    n_horizon=20000,
    name_policies={
        LG1T: {"threshold": 0.3},
        LG2T: {"threshold": [0.9], "power": 1 / 2},
        LG1_2T_Adaptive: {"threshold": [0.9], "power": 1 / 2, "threshold_i": 0.3, "change_point": 100 * 25 * 20000},
    },
    entropy=243799254704924441050048792905230269161,
    model_type="discrete",
    n_cpus=16,
):
    path = os.path.abspath("./results_random")
    os.makedirs(path, exist_ok=True)
    tag = "__".join([f"S{n_state}", f"A{n_action}", f"E{n_experiments}", f"Re{n_replications_per_experiment}", f"H{n_horizon}"])
    data_pkl = os.path.join(path, f"data__{tag}_synthetic_parallel_1_2.pkl")

    ss = SeedSequence(entropy)
    children = ss.spawn(n_experiments * (n_replications_per_experiment + 1))
    sq = np.array(children, dtype=object).reshape(n_experiments, n_replications_per_experiment + 1)

    learners = build_learners(name_policies)
    names = policy_names(learners, n_state, n_action, model_type)
    results = {name: {"average_reward": np.zeros((n_horizon, n_replications_per_experiment, n_experiments))} for name in names}

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_cpus, mp_context=ctx) as executor:
        for e in range(n_experiments):
            t_spend = time.time() - t0
            t_rem = (n_experiments - e) * t_spend / max(e, 1)
            print(f"Run {e + 1} ... (spend {time_str(t_spend)}, remains {time_str(t_rem)})")
            tasks = [(n_state, n_action, n_horizon, learners, model_type, sq[e, 0], sq[e, run + 1], sq[e, run + 1]) for run in range(n_replications_per_experiment)]
            replications = list(executor.map(run_single_replication, tasks))
            for run, replication in enumerate(replications):
                for name, info in replication.items():
                    results[name]["average_reward"][:, run, e] = info["average_reward"]
            with open(data_pkl, "wb") as pkl:
                pickle.dump(results, pkl)

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots()
    legend = []
    for name, color in zip(names, colors):
        std = np.std(results[name]["average_reward"], axis=(-2, -1), ddof=1)
        y = np.mean(results[name]["average_reward"], axis=(-2, -1))
        line, = ax.plot(y, color=color, label=name)
        ax.fill_between(np.arange(n_horizon), y - 1.96 * std / np.sqrt(n_experiments * n_replications_per_experiment), y + 1.96 * std / np.sqrt(n_experiments * n_replications_per_experiment), color=color, alpha=0.2, linewidth=0)
        legend.append((line, name))
    fig.legend(*zip(*legend), loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{tag}_synthetic_parallel_1_2.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_state", type=int, default=100)
    parser.add_argument("--n_action", type=int, default=25)
    parser.add_argument("--n_experiments", type=int, default=1000)
    parser.add_argument("--n_replications_per_experiment", type=int, default=1)
    parser.add_argument("--n_horizon", type=int, default=20000)
    parser.add_argument("--name_policies", type=object, default={LG1T: {"threshold": 0.3}, LG2T: {"threshold": [0.9], "power": 1 / 2}, LG1_2T_Adaptive: {"threshold": [0.9], "power": 1 / 2, "threshold_i": 0.3, "change_point": 100 * 25 * 20000}})
    parser.add_argument("--model_type", type=str, default="discrete")
    parser.add_argument("--entropy", type=int, default=243799254704924441050048792905230269161)
    parser.add_argument("--n_cpus", type=int, default=16)
    main(**vars(parser.parse_args()))
