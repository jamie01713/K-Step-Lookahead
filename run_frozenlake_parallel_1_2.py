from concurrent.futures import ProcessPoolExecutor
from itertools import product
import multiprocessing as mp
import os
import pickle
import time

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import SeedSequence

from policies.Online_Multiple_Step import LG1T, LG2T, LG1_2T_Adaptive


FROZENLAKE_DESC = ["SFFF", "FHFH", "FFFH", "HFFG"]


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


def build_model(n_horizon):
    return gym.make(
        "FrozenLake-v1",
        desc=FROZENLAKE_DESC,
        is_slippery=True,
        max_episode_steps=n_horizon,
        success_rate=1.0 / 3.0,
        reward_schedule=(1, 0, 0.2),
    )


def policy_names(learners, n_horizon, model_type):
    model = build_model(n_horizon)
    try:
        return [policy(model, model_type=model_type, **cfg).name() for policy, cfg in learners]
    finally:
        model.close()


def average_reward_from_history(history):
    rewards = np.array([step[2] for step in history], dtype=float)
    return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)


def run_single_replication(args):
    n_horizon, learners, model_type, run_seed, worker_seed = args

    np.random.seed(int(worker_seed.generate_state(1)[0]))
    env_seed = int(run_seed.generate_state(1)[0])
    model = build_model(n_horizon)
    run_result = {}

    try:
        for policy_cls, cfg in learners:
            policy = policy_cls(model, model_type=model_type, **cfg)
            s, _ = model.reset(seed=env_seed)
            policy.reset(model)
            history = [[s, False, False]]

            for _ in range(n_horizon):
                x, done, truncated = history[-1]
                if done or truncated:
                    s, _ = model.reset()
                    history[-1] = [s, False, False]
                    x = s

                action = policy.act(x)
                a = action[0] if isinstance(action, tuple) else action
                y, r, done, truncated, _ = model.step(a)
                policy.observe(x, a, r, y, done, truncated)
                history[-1] = (x, a, r, y, done, truncated)
                history.append((y, done, truncated))

            history.pop()
            run_result[policy.name()] = {
                "average_reward": average_reward_from_history(history),
                "time": getattr(policy, "rsum", 0.0),
            }
    finally:
        model.close()

    return run_result


def main(
    n_state=5,
    n_action=4,
    n_experiments=100,
    n_replications_per_experiment=1,
    n_horizon=20000,
    name_policies={
        LG1T: {"threshold": 0.3},
        LG2T: {"threshold": [0.9], "power": 1 / 2},
        LG1_2T_Adaptive: {"threshold": [0.9], "power": 1 / 2, "threshold_i": 0.3},
    },
    entropy=243799254704924441050048792905230269161,
    model_type="discrete",
    n_cpus=16,
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
    data_pkl = os.path.join(path, f"data__{tag}_frozenlake_parallel_1_2.pkl")

    ss = SeedSequence(entropy)
    run_children = ss.spawn(n_experiments * n_replications_per_experiment)
    worker_children = ss.spawn(n_experiments * n_replications_per_experiment)
    run_seeds = np.array(run_children, dtype=object).reshape(n_experiments, n_replications_per_experiment)
    worker_seeds = np.array(worker_children, dtype=object).reshape(n_experiments, n_replications_per_experiment)

    learners = build_learners(name_policies)
    names = policy_names(learners, n_horizon, model_type)
    results = {
        name: {
            "average_reward": np.zeros((n_horizon, n_replications_per_experiment, n_experiments)),
            "time": np.zeros((n_replications_per_experiment, n_experiments)),
        }
        for name in names
    }

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_cpus, mp_context=ctx) as executor:
        for e in range(n_experiments):
            t_spend = time.time() - t0
            t_rem = (n_experiments - e) * t_spend / max(e, 1)
            print(f"Run {e + 1} ... (spend {time_str(t_spend)}, remains {time_str(t_rem)})")

            tasks = [
                (
                    n_horizon,
                    learners,
                    model_type,
                    run_seeds[e, run],
                    worker_seeds[e, run],
                )
                for run in range(n_replications_per_experiment)
            ]

            replications = list(executor.map(run_single_replication, tasks))
            for run, replication in enumerate(replications):
                for name, info in replication.items():
                    results[name]["average_reward"][:, run, e] = info["average_reward"]
                    results[name]["time"][run, e] = info["time"]

            with open(data_pkl, "wb") as pkl:
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
    plt.savefig(f"{tag}_frozenlake_parallel_1_2.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_state", type=int, default=5)
    parser.add_argument("--n_action", type=int, default=4)
    parser.add_argument("--n_experiments", type=int, default=100)
    parser.add_argument("--n_replications_per_experiment", type=int, default=1)
    parser.add_argument("--n_horizon", type=int, default=20000)
    parser.add_argument(
        "--name_policies",
        type=object,
        default={
            LG1T: {"threshold": 0.3},
            LG2T: {"threshold": [0.9], "power": 1 / 2},
            LG1_2T_Adaptive: {"threshold": [0.9], "power": 1 / 2, "threshold_i": 0.3},
        },
    )
    parser.add_argument("--model_type", type=str, default="discrete")
    parser.add_argument("--entropy", type=int, default=243799254704924441050048792905230269161)
    parser.add_argument("--n_cpus", type=int, default=16)
    args = parser.parse_args()
    main(**vars(args))
