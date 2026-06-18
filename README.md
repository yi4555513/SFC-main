# SFC-main

SFC-main is a Python simulation project for Service Function Chain (SFC) deployment and routing in satellite networks. It supports satellite topology snapshots, SFC request generation, resource-constrained deployment, and routing simulation.

## Structure

```text
.
├── main.py              # simulation entry
├── main/                # simulator, controller, environment and solvers
├── settings/            # configuration files
├── scripts/             # utility scripts
├── TLE/                 # satellite topology generation utilities
└── datasets/            # topology snapshot directory
```

Generated datasets, logs, models and results are ignored by Git.

## Installation

Python 3.10 is recommended.

```bash
git clone https://github.com/yi4555513/SFC-main.git
cd SFC-main
conda create -n sfc-main python=3.10
conda activate sfc-main
pip install -r requirements.txt
```

If PyTorch Geometric fails to install, install the version matching your PyTorch/CUDA or CPU environment first.

## Basic Workflow

### 1. Generate topology snapshots

```bash
python scripts/generate_topology_snapshots.py --config-name main_ch3
```

The generated `.gml` snapshots will be saved in:

```text
datasets/topology/snapshots/
```

### 2. Run a quick simulation

```bash
python main.py --config-name main_ch3 solver.solver_name=aco_meta v_sim_setting.num_v_nets=5 v_sim_setting.auto_num_v_nets_from_arrival_rate=false v_sim_setting.snapshot_duration_ms=60000
```

If progress information is printed and files are written to `results/`, the project is running correctly.

### 3. Run with your own settings

Most parameters can be changed directly from the command line. Example:

```bash
python main.py --config-name main_ch3 solver.solver_name=aco_meta v_sim_setting.num_v_nets=1000 v_sim_setting.arrival_rate.lam=0.016 experiment.seed=1
```

Common solver names include:

```text
aco_meta, mip, ppo_mlp+, ppo_gat_seq2seq+
```

For QoS-related configuration, replace `main_ch3` with `main_ch4`.

## Outputs

Simulation records, logs and models are saved under `results/`. Generated request and topology datasets are saved under `dataset/` or `datasets/` according to the configuration.

## License

This repository is intended for research and simulation use.
