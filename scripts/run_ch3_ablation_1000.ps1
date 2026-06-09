$ErrorActionPreference = "Stop"
Set-Location "D:\any download\virne-main\virne-main"
$pythonExe = "D:\anaconda\envs\virne-main\python.exe"

# Ch3 ablation: train only the three ablation variants under the medium-load
# 1000-SFC, 10-snapshot setting. The full PPO-MGT medium-load result is used
# as the reference in plotting.
$solvers = @(
    "ppo_gat_seq2seq+_noGAT",
    "ppo_gat_seq2seq+_noTransformer",
    "ppo_gat_seq2seq+_noGAT_noTransformer"
)

foreach ($solver in $solvers) {
    Write-Host "============================================================"
    Write-Host "Running Ch3 ablation solver=$solver, num_requests=1000, epochs=10"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        solver.solver_name=$solver `
        training.num_train_epochs=10 `
        experiment.seed=0 `
        v_sim_setting.num_v_nets=1000 `
        v_sim_setting.arrival_rate.lam=0.004 `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
        v_sim_setting.snapshot_duration_ms=100000 `
        p_net_setting.topology.num_snapshots=10 `
        p_net_setting.topology.cycle_snapshots=true

    if ($LASTEXITCODE -ne 0) {
        throw "Ch3 ablation run failed: solver=$solver, exit_code=$LASTEXITCODE"
    }
}
