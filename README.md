# ICML_2026_K_STEP

This repository contains experimental code for online reinforcement learning policies on synthetic environments, RiverSwim, and FrozenLake-style setups.

## Structure

- `environments/`: environment and MDP definitions
- `policies/`: policy implementations
- `run_5_riverswim.py`: 5-state RiverSwim runner
- `run_5_riverswim_parallel.py`: 5-state RiverSwim runner with CPU parallelism over repetitions
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
python run_5_riverswim_parallel.py --n_cpus 16
python run_8_riverswim.py
```

The scripts write experiment outputs to `results_random/` and may also generate figures in the project root unless you change the paths in the runners.

## GitHub

Suggested initialization:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-user>/<repo-name>.git
git push -u origin main
```
