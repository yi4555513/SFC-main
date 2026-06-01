# PPO-MGT SFC Deployment Simulator

This is a minimized runnable copy of the satellite SFC deployment simulator used for the thesis experiments. It keeps the simulation pipeline and the experiment solvers, but excludes generated datasets, results, figures, plotting scripts, thesis files, and temporary logs.

## Included

- Generate idealized TLE records.
- Generate multi-snapshot LEO topology GML files.
- Generate physical network and virtual SFC request datasets during simulation.
- Run online SFC deployment experiments.
- Solvers used in the thesis experiments:
  - `ppo_gat_seq2seq+` / PPO-MGT
  - `ppo_mlp+` / PPO-Baseline
  - `aco_meta`
  - `mip`
  - PPO-MGT ablation names are also kept for reproducibility.

## Not Included

- Generated `dataset/` files.
- `results/`, trained model checkpoints, logs, and records.
- Plotting scripts and thesis figures.
- Pretrained PPO models. For test-only reproduction, pass your own model path through `solver.pretrained_model_path=...`; otherwise train first.

## Install

Use the same Python environment as the original experiment when possible.

```powershell
pip install -r requirements.txt
```

For PyTorch Geometric, install wheels matching your local PyTorch and CUDA version if the generic install is not enough.

## 1. Generate Satellite Snapshots

Chapter 3:

```powershell
python scripts\generate_topology_snapshots.py --config-name main_ch3
```

Chapter 4:

```powershell
python scripts\generate_topology_snapshots.py --config-name main_ch4
```

Generated topology snapshots are written to:

```text
datasets/topology/snapshots/
```

## 2. Run Simulations

Chapter 3 PPO-MGT:

```powershell
python main.py --config-name main_ch3 solver.solver_name=ppo_gat_seq2seq+
```

Chapter 3 PPO-Baseline:

```powershell
python main.py --config-name main_ch3 solver.solver_name=ppo_mlp+
```

Chapter 3 ACO-Meta:

```powershell
python main.py --config-name main_ch3 solver.solver_name=aco_meta
```

Chapter 3 MIP:

```powershell
python main.py --config-name main_ch3 solver.solver_name=mip +solver.mip_time_limit_seconds=10
```

Chapter 4 PPO-MGT with uniform service ratios:

```powershell
python main.py --config-name main_ch4 solver.solver_name=ppo_gat_seq2seq+
```

Chapter 4 PPO-Baseline with uniform service ratios:

```powershell
python main.py --config-name main_ch4 solver.solver_name=ppo_mlp+
```

Chapter 4 dominant service ratio example:

```powershell
python main.py --config-name main_ch4 solver.solver_name=ppo_gat_seq2seq+ "v_sim_setting.service_qos.service_ratios={delay_sensitive:0.40,bandwidth_sensitive:0.20,reliability_sensitive:0.20,compute_sensitive:0.20}"
```

## Smoke Test

After generating topology snapshots, run a tiny ACO simulation:

```powershell
python main.py --config-name main_ch3 solver.solver_name=aco_meta v_sim_setting.num_v_nets=2 ++v_sim_setting.auto_num_v_nets_from_arrival_rate=false experiment.if_save_p_net=false experiment.if_save_v_nets=false recorder.if_save_records=false logger.backends=[console]
```
