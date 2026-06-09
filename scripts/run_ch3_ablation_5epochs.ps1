$ErrorActionPreference = "Stop"
Set-Location "D:\any download\virne-main\virne-main"
$pythonExe = "D:\anaconda\envs\virne-main\python.exe"

# Ch3 ablation quick rerun after current code changes.
# Setting: 1000 SFC, 10 snapshots, lambda=0.004, train 5 epochs, then final test automatically.
# Variants:
# 1) w/o M-GAT
# 2) w/o Transformer
# 3) w/o M-GAT & Transformer
$epochs = 5
$solvers = @(
    "ppo_gat_seq2seq+_noGAT",
    "ppo_gat_seq2seq+_noTransformer",
    "ppo_gat_seq2seq+_noGAT_noTransformer"
)

foreach ($solver in $solvers) {
    Write-Host "============================================================"
    Write-Host "Running Ch3 ablation QUICK solver=$solver, num_requests=1000, epochs=$epochs"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        solver.solver_name=$solver `
        training.num_train_epochs=$epochs `
        experiment.seed=0 `
        v_sim_setting.num_v_nets=1000 `
        v_sim_setting.arrival_rate.lam=0.004 `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
        v_sim_setting.snapshot_duration_ms=100000 `
        p_net_setting.topology.num_snapshots=10 `
        p_net_setting.topology.cycle_snapshots=true

    if ($LASTEXITCODE -ne 0) {
        throw "Ablation run failed: solver=$solver, exit_code=$LASTEXITCODE"
    }
}

Write-Host "============================================================"
Write-Host "All quick Ch3 ablation runs finished."
Write-Host "============================================================"
