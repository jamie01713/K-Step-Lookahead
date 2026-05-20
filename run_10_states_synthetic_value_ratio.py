from environments.mdp_10_input_variance import MD_10_InputVariance
from policies.Online_Multiple_Step import Multi_Step
import os
import pickle
from numpy.random import SeedSequence
import numpy as np
from matplotlib import pyplot as plt


def evaluate_policy_finite_horizon(model, policy_actions, horizon):
    values = np.zeros((horizon + 1, model.n_states))
    rewards = model.mu
    kernels = model.p

    for t in range(horizon - 1, -1, -1):
        for s in range(model.n_states):
            a = policy_actions[s]
            values[t, s] = rewards[s, a] + np.dot(kernels[s, a], values[t + 1])

    return values[0]


def optimal_value_finite_horizon(model, horizon):
    values = np.zeros((horizon + 1, model.n_states))
    rewards = model.mu
    kernels = model.p

    for t in range(horizon - 1, -1, -1):
        q_values = rewards + np.einsum("san,n->sa", kernels, values[t + 1])
        values[t] = np.max(q_values, axis=1)

    return values[0]


def optimal_policy_and_value_finite_horizon(model, horizon):
    values = np.zeros((horizon + 1, model.n_states))
    actions = np.zeros((horizon, model.n_states), dtype=int)
    rewards = model.mu
    kernels = model.p

    for t in range(horizon - 1, -1, -1):
        q_values = rewards + np.einsum("san,n->sa", kernels, values[t + 1])
        actions[t] = np.argmax(q_values, axis=1)
        values[t] = np.max(q_values, axis=1)

    return actions, values[0]


def multi_step_value_finite_horizon(model, step, horizon):
    policy = Multi_Step(model, step=step)
    actions = np.array([policy.act(s) for s in range(model.n_states)])
    return evaluate_policy_finite_horizon(model, actions, horizon)


def multi_step_actions(model, step):
    policy = Multi_Step(model, step=step)
    return np.array([policy.act(s) for s in range(model.n_states)])


def summed_policy_transition_l1_distance(model, multi_step_one_actions, optimal_actions_by_time):
    total_distance = 0.0
    for t in range(optimal_actions_by_time.shape[0]):
        max_state_gap = 0.0
        for s in range(model.n_states):
            a_multi = multi_step_one_actions[s]
            a_opt = optimal_actions_by_time[t, s]
            gap = np.abs(model.p[s, a_multi] - model.p[s, a_opt]).sum()
            max_state_gap = max(max_state_gap, float(gap))
        total_distance += max_state_gap
    return total_distance


def main(
    n_state=10,
    n_action=5,
    n_experiments_per_variance=100,
    horizon=1000,
    step=1,
    entropy=243799254704924441050048792905230269161,
    transition_variances=(10.0, 5.0, 2.0),
):
    path = os.path.abspath("./results_random")
    os.makedirs(path, exist_ok=True)
    transition_variances = tuple(transition_variances)
    n_experiments = n_experiments_per_variance * len(transition_variances)

    tag = "__".join(
        [
            f"S{n_state}",
            f"A{n_action}",
            f"E{n_experiments_per_variance}each",
            f"H{horizon}",
            f"K{step}",
            "Var" + "-".join(str(v) for v in transition_variances),
        ]
    )
    data_pkl = os.path.join(path, f"data__{tag}_value_ratio.pkl")
    fig_path = os.path.join(path, f"{tag}_value_ratio_vs_l1.pdf")

    ss = SeedSequence(entropy)
    children = ss.spawn(n_experiments)

    l1_distances = np.zeros(n_experiments)
    ratios = np.zeros(n_experiments)
    multi_step_values = np.zeros((n_experiments, n_state))
    optimal_values = np.zeros((n_experiments, n_state))
    variances_used = np.zeros(n_experiments)

    for e, transition_variance in enumerate(
        np.repeat(np.array(transition_variances, dtype=float), n_experiments_per_variance)
    ):
        model = MD_10_InputVariance.generate_instance(
            n_states=n_state,
            n_actions=n_action,
            entropy=children[e],
            transition_variance=float(transition_variance),
            name=f"synthetic_{e}",
        )

        multi_step_one_actions = multi_step_actions(model, step=1)
        optimal_actions, optimal_v = optimal_policy_and_value_finite_horizon(model, horizon=horizon)
        l1_distances[e] = summed_policy_transition_l1_distance(
            model,
            multi_step_one_actions,
            optimal_actions,
        )
        multi_step_v = multi_step_value_finite_horizon(model, step=step, horizon=horizon)

        multi_step_values[e] = multi_step_v
        optimal_values[e] = optimal_v
        variances_used[e] = transition_variance

        numerator = np.mean(multi_step_v)
        denominator = np.mean(optimal_v)
        ratios[e] = numerator / denominator if denominator != 0 else np.nan

    results = {
        "l1_distances": l1_distances,
        "ratios": ratios,
        "multi_step_values": multi_step_values,
        "optimal_values": optimal_values,
        "step": step,
        "horizon": horizon,
        "n_state": n_state,
        "n_action": n_action,
        "n_experiments": n_experiments,
        "n_experiments_per_variance": n_experiments_per_variance,
        "transition_variances": transition_variances,
        "variances_used": variances_used,
    }

    with open(data_pkl, "wb") as pkl:
        pickle.dump(results, pkl)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(l1_distances, ratios, alpha=0.28, s=35, label="Instances")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"$\sum_t \max_s \|P_{s,a_{s,1}} - P_{s,a^*_{s,t}}\|_1$")
    ax.set_ylabel("Mean value ratio: Multi_Step / Optimal")
    ax.set_title(f"Value ratio vs transition L1 distance (step={step})")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_state", type=int, default=10)
    parser.add_argument("--n_action", type=int, default=5)
    parser.add_argument("--n_experiments_per_variance", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--entropy", type=int, default=243799254704924441050048792905230269161)
    parser.add_argument("--transition_variances", type=float, nargs="+", default=[10.0, 5.0, 2.0])
    args = parser.parse_args()
    main(**vars(args))
