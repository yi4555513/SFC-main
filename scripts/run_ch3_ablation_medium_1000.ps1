$ErrorActionPreference = "Stop"
Set-Location "D:\any download\virne-main\virne-main"
$pythonExe = "D:\anaconda\envs\virne-main\python.exe"

# Ch3 ablation under a medium-load setting.
# PPO-MGT uses the trained model directly; only ablation variants are trained from scratch.
$lambda = 0.009
$numRequests = 1000
$totalTimeMs = 111111   # approx 1000 / 0.009, so auto-generated SFC count is about 1000.
$numSnapshots = 3
$pretrainedMgt = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260526T165819-1245\models\model.pkl"

Write-Host "============================================================"
Write-Host "Testing pretrained PPO-MGT: lambda=$lambda, num_requests≈$numRequests"
Write-Host "============================================================"
& $pythonExe main.py --config-name main_ch3 `
    solver.solver_name=ppo_gat_seq2seq+ `
    "solver.pretrained_model_path=$pretrainedMgt" `
    training.num_train_epochs=0 `
    v_sim_setting.arrival_rate.lam=$lambda `
    v_sim_setting.fixed_total_time_ms=$totalTimeMs `
    v_sim_setting.auto_num_v_nets_from_arrival_rate=true `
    p_net_setting.topology.num_snapshots=$numSnapshots

$ablationSolvers = @(
    "ppo_gat_seq2seq+_noGAT",
    "ppo_gat_seq2seq+_noTransformer",
    "ppo_gat_seq2seq+_noGAT_noTransformer"
)

foreach ($solver in $ablationSolvers) {
    Write-Host "============================================================"
    Write-Host "Training and testing ablation solver=$solver: lambda=$lambda, num_requests≈$numRequests"
    Write-Host "============================================================"
    & $pythonExe main.py --config-name main_ch3 `
        solver.solver_name=$solver `
        v_sim_setting.arrival_rate.lam=$lambda `
        v_sim_setting.fixed_total_time_ms=$totalTimeMs `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=true `
        p_net_setting.topology.num_snapshots=$numSnapshots
}
