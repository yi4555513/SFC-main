$ErrorActionPreference = "Stop"

# 第三章基础实验：MIP，66 节点，中等负载，1000 条 SFC

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

& $pythonExe main.py --config-name main_ch3 `
    experiment.seed=0 `
    solver.solver_name=mip `
    training.num_train_epochs=0 `
    p_net_setting.topology.num_nodes=66 `
    p_net_setting.topology.planes=6 `
    p_net_setting.topology.nums_per_plane=11 `
    p_net_setting.topology.num_snapshots=10 `
    p_net_setting.topology.cycle_snapshots=true `
    v_sim_setting.num_v_nets=1000 `
    v_sim_setting.arrival_rate.lam=0.004 `
    v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
    v_sim_setting.snapshot_duration_ms=100000 `
    solver.mip_time_limit_seconds=5 `
    solver.objective.name=resource_latency `
    solver.objective.resource_weight=1.0 `
    solver.objective.latency_weight=0.1 `
    solver.objective.soft_latency_constraint=false `
    solver.mip_post_reroute=false

if ($LASTEXITCODE -ne 0) {
    throw "Ch3 MIP 66-node basic test failed: exit_code=$LASTEXITCODE"
}
