# K-STEP Lookahead Thresholding

This contains the experiment results of LGKT, an online algorithm that learns K-step lookahead thresholding policy to improve the performance of finite-horizon, nonepisodic RL.

## Structure

- `environments/`: environment and MDP definitions
- `policies/`: policy implementations
- `run_5_riverswim.py`: 5-state RiverSwim runner
- `run_8_riverswim.py`: 8-state RiverSwim runner
- `run_15_riverswim.py`: 15-state RiverSwim runner
- `run_10_states_synthetic.py`: synthetic benchmark runner
- `run_100_states_synthetic.py`: larger synthetic benchmark runner
- `run_frozenlake.py`: FrozenLake-style runner
- `environment.yml`: environment definition

## Setup

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate <environment-name>
```

## Running

Example commands:

```bash
python run_5_riverswim.py
python run_8_riverswim.py
```

The scripts write experiment outputs to `results_random/` and may also generate figures in the project root unless you change the paths in the runners.
