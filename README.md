# SFC-main

SFC-main is a Python simulation project for Service Function Chain (SFC) deployment and routing in satellite networks. It provides configurable network/request generation, topology snapshots, several baseline solvers, and reinforcement-learning based solvers.

## Features

- Dynamic satellite topology snapshots
- SFC request generation with node and link resource requirements
- Node placement and path routing under resource and latency constraints
- Supported solvers: `mip`, `aco_meta`, `ppo_mlp+`, `ppo_gat_seq2seq+`
- Hydra-based configuration and command-line overrides

## Project Structure

```text
.
├── main.py              # entry point
├── main/                # core simulator, environments, controllers and solvers
├── settings/            # configuration files
├── scripts/             # utility scripts
├── TLE/                 # satellite topology generation utilities
└── datasets/            # generated topology/request datasets
```

Generated results, logs, models and figures are not included in this repository.

## Installation

Python 3.10 is recommended.

```bash
git clone https://github.com/yi4555513/SFC-main.git
cd SFC-main

conda create -n sfc-main python=3.10
conda activate sfc-main
pip install -r requirements.txt
```

If PyTorch Geometric packages fail to install, install the versions that match your local PyTorch and CUDA/CPU environment, then run `pip install -r requirements.txt` again.

## Prepare Topology Snapshots

Before running simulations, generate satellite topology snapshots:

```bash
python scripts/generate_topology_snapshots.py --config-name main_ch3
```

For the QoS configuration, use:

```bash
python scripts/generate_topology_snapshots.py --config-name main_ch4
```

The generated GML files will be saved under:

```text
datasets/topology/snapshots/
```

## Quick Test

Run a small ACO test with 5 SFC requests:

```bash
python main.py --config-name main_ch3 solver.solver_name=aco_meta v_sim_setting.num_v_nets=5 v_sim_setting.auto_num_v_nets_from_arrival_rate=false v_sim_setting.snapshot_duration_ms=60000
```

If the program prints progress information and writes records under `results/`, the basic environment is ready.

## Run Simulations

Basic configuration:

```bash
python main.py --config-name main_ch3 solver.solver_name=aco_meta
python main.py --config-name main_ch3 solver.solver_name=mip
python main.py --config-name main_ch3 solver.solver_name=ppo_mlp+
python main.py --config-name main_ch3 solver.solver_name=ppo_gat_seq2seq+
```

QoS configuration:

```bash
python main.py --config-name main_ch4 solver.solver_name=aco_meta
python main.py --config-name main_ch4 solver.solver_name=mip
python main.py --config-name main_ch4 solver.solver_name=ppo_mlp+
python main.py --config-name main_ch4 solver.solver_name=ppo_gat_seq2seq+
```

Common command-line overrides:

```bash
# number of SFC requests
v_sim_setting.num_v_nets=1000

# arrival rate
v_sim_setting.arrival_rate.lam=0.016

# disable automatic request number calculation
v_sim_setting.auto_num_v_nets_from_arrival_rate=false

# snapshot duration in milliseconds
v_sim_setting.snapshot_duration_ms=60000

# random seed
experiment.seed=1
```

Example:

```bash
python main.py --config-name main_ch3 solver.solver_name=aco_meta v_sim_setting.num_v_nets=1000 v_sim_setting.auto_num_v_nets_from_arrival_rate=false v_sim_setting.arrival_rate.lam=0.016 v_sim_setting.snapshot_duration_ms=60000 experiment.seed=1
```

## Pretrained PPO Models

To evaluate a saved PPO model, set `training.num_train_epochs=0` and provide the model path:

```bash
python main.py --config-name main_ch3 solver.solver_name=ppo_gat_seq2seq+ training.num_train_epochs=0 solver.pretrained_model_path=path/to/model.pkl
```

To train from scratch, set `training.num_train_epochs` to a positive value.

## Outputs

Simulation outputs are saved under `results/`. Generated datasets are saved under `dataset/` or `datasets/`, depending on the configuration. These files are ignored by Git.

## License

This repository is intended for research and simulation use.
