from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import SeedSequence

from environments.mdp_10 import MD_10
from environments.mdp_100 import MDP_100
from policies.Online_Multiple_Step import LG1T, LG1TU, LG2T, LG2TU
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
            cfg = dict(zip(keys, combo))
            learners.append((policy, cfg))
    return learners


def build_model(kind, n_state, n_action, env_seed):
    if kind == "synthetic10":
        return MD_10(n_states=n_state, n_actions=n_action, entropy=env_seed)
    if kind == "synthetic100":
        return MDP_100(n_states=n_state, n_actions=n_action, entropy=env_seed)
    raise ValueError(f"Unknown model kind: {kind}")


def policy_names(learners, kind, n_state, n_action, model_type):
    model = build_model(kind, n_state, n_action, SeedSequence(0))
    return [policy(model, model_type=model_type, **cfg).name() for policy, cfg in learners]


def run_one_experiment(entropy, kind, n_state, n_action, n_replications_per_experiment, n_horizon, learners, model_type):
    model = build_model(kind, n_state, n_action, entropy[0])
    policies = [policy_cls(model, model_type=model_type, **cfg) for policy_cls, cfg in learners]
    results = {
        policy.name(): {
            "average_reward": np.zeros((n_horizon, n_replications_per_experiment)),
            "time": np.zeros(n_replications_per_experiment),
        }
        for policy in policies
    }

    for run in range(n_replications_per_experiment):
        for policy in policies:
            s, _ = model.reset(entropy[run + 1])
            policy.reset(model)
            history = [[s, False, False]]
            for _ in range(n_horizon - 1):
                iterate_algorithm(model, policy, history, 0)
            history.pop()
            info = parse_history(model, history, model_type=model_type)
            results[policy.name()]["average_reward"][:, run] = info["average expected reward"]
            results[policy.name()]["time"][run] = getattr(policy, "rsum", 0.0)

    return results


def run_parallel_suite(
    *,
    kind,
    n_state,
    n_action,
    n_experiments,
    n_replications_per_experiment,
    n_horizon,
    name_policies,
    entropy,
    model_type,
    n_cpus,
):
    path = os.path.abspath("./results_random")
    os.makedirs(path, exist_ok=True)

    tag = "__".join(
        [
            f"S{n_state}",
            f"A{n_action}",
            f"E{n_experiments}",
            f"Re{n_replications_per_experiment}",
            f"H{n_horizon}",
        ]
    )
    data_pkl = os.path.join(path, f"data__{tag}_{kind}_parallel.pkl")

    ss = entropy if isinstance(entropy, SeedSequence) else SeedSequence(entropy)
    children = ss.spawn(n_experiments * (n_replications_per_experiment + 1))
    sq = np.array(children, dtype=object).reshape(n_experiments, n_replications_per_experiment + 1)

    learners = build_learners(name_policies)
    names = policy_names(learners, kind, n_state, n_action, model_type)
    results = {
        name: {
            "average_reward": np.zeros((n_horizon, n_replications_per_experiment, n_experiments)),
            "time": np.zeros((n_replications_per_experiment, n_experiments)),
        }
        for name in names
    }

    print(f"\n=== Starting {kind} ===")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    configs = [
        {
            "entropy": sq[e],
            "kind": kind,
            "n_state": n_state,
            "n_action": n_action,
            "n_replications_per_experiment": n_replications_per_experiment,
            "n_horizon": n_horizon,
            "learners": learners,
            "model_type": model_type,
        }
        for e in range(n_experiments)
    ]
    with ProcessPoolExecutor(max_workers=n_cpus, mp_context=ctx) as executor:
        futures = [executor.submit(run_one_experiment, **config) for config in configs]
        all_results = []
        for future in as_completed(futures):
            all_results.append(future.result())

        with open(data_pkl, "wb") as pkl:
            for e in range(n_experiments):
                for name in results:
                    results[name]["average_reward"][:, :, e] = all_results[e][name]["average_reward"]
                    results[name]["time"][:, e] = all_results[e][name]["time"]
            pickle.dump(results, pkl)

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots()
    legend = []
    for name, color in zip(names, colors):
        std = np.std(results[name]["average_reward"], axis=(-2, -1), ddof=1)
        y = np.mean(results[name]["average_reward"], axis=(-2, -1))
        line, = ax.plot(y, color=color, label=name)
        ax.fill_between(
            np.arange(n_horizon),
            y - 1.96 * std / np.sqrt(n_experiments * n_replications_per_experiment),
            y + 1.96 * std / np.sqrt(n_experiments * n_replications_per_experiment),
            color=color,
            alpha=0.2,
            linewidth=0,
        )
        legend.append((line, name))

    fig.legend(*zip(*legend), loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{tag}_{kind}_parallel.pdf", bbox_inches="tight")
    plt.show()

    return results


def main(
    n_experiments_10=100,
    n_experiments_100=1000,
    n_replications_per_experiment=1,
    n_horizon=20000,
    model_type="discrete",
    entropy=243799254704924441050048792905230269161,
    n_cpus=16,
):
    policies_10 = {
        LG1T: {"threshold": 0.3},
        LG1TU: {"threshold": [-0.1, 0.3, 1.1, 1.9, 2.7]},
        LG2T: {"threshold": 0.9, "power": 1 / 2},
        LG2TU: {"threshold": [0.3, 0.9, 2.1, 3.3], "power": 1 / 2},
    }
    policies_100 = {
        LG1T: {"threshold": 0.3},
        LG1TU: {"threshold": [0.3, 1.3, 2.3, 3.3]},
        LG2T: {"threshold": 0.9, "power": 1 / 2},
        LG2TU: {"threshold": [0.9, 2.9, 4.9, 6.9], "power": 1 / 2},
    }

    run_parallel_suite(
        kind="synthetic10",
        n_state=10,
        n_action=5,
        n_experiments=n_experiments_10,
        n_replications_per_experiment=n_replications_per_experiment,
        n_horizon=n_horizon,
        name_policies=policies_10,
        entropy=entropy,
        model_type=model_type,
        n_cpus=n_cpus,
    )
    run_parallel_suite(
        kind="synthetic100",
        n_state=100,
        n_action=25,
        n_experiments=n_experiments_100,
        n_replications_per_experiment=n_replications_per_experiment,
        n_horizon=n_horizon,
        name_policies=policies_100,
        entropy=entropy,
        model_type=model_type,
        n_cpus=n_cpus,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_experiments_10", type=int, default=100)
    parser.add_argument("--n_experiments_100", type=int, default=1000)
    parser.add_argument("--n_replications_per_experiment", type=int, default=1)
    parser.add_argument("--n_horizon", type=int, default=20000)
    parser.add_argument("--model_type", type=str, default="discrete")
    parser.add_argument("--entropy", type=int, default=243799254704924441050048792905230269161)
    parser.add_argument("--n_cpus", type=int, default=16)
    args = parser.parse_args()
    main(**vars(args))
