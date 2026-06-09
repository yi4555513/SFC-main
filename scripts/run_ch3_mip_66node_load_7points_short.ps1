$ErrorActionPreference = "Stop"

# 第三章 MIP 不同负载实验：66 节点正式口径
# 注意：不要再读取 results/virne/mip/旧 config，旧 config 里是 240 节点。

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$seed = 0
$numSnapshots = 10
$snapshotDurationMs = 100000
$fixedTotalTimeMs = 30000
$mipTimeLimitSeconds = 5
$loadRates = @(0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021)

foreach ($lam in $loadRates) {
    $expectedSfc = [int][math]::Round($fixedTotalTimeMs * $lam)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running Ch3 MIP 66-node load: lambda=$lam, SFC=$expectedSfc, time_limit=${mipTimeLimitSeconds}s"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        experiment.seed=$seed `
        solver.solver_name=mip `
        training.num_train_epochs=0 `
        p_net_setting.topology.num_nodes=66 `
        p_net_setting.topology.planes=6 `
        p_net_setting.topology.nums_per_plane=11 `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        p_net_setting.topology.cycle_snapshots=true `
        v_sim_setting.arrival_rate.lam=$lam `
        v_sim_setting.fixed_total_time_ms=$fixedTotalTimeMs `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=true `
        v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
        solver.objective.name=resource `
        solver.objective.soft_latency_constraint=false `
        solver.mip_post_reroute=false

    if ($LASTEXITCODE -ne 0) {
        throw "Ch3 MIP 66-node load failed: lambda=$lam, exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Ch3 MIP 66-node load tests finished."
