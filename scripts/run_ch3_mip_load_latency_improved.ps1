# ============================================================
# Ch3 MIP load test restored to the original resource-objective setting.
# It only uses MIP resource optimization and then applies the common final
# total_latency check, without latency objective / soft latency / post-reroute.
# ============================================================

$ErrorActionPreference = "Stop"

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$seed = 0
$numSnapshots = 10
$snapshotDurationMs = 100000
$mipTimeLimitSeconds = 5
$fixedTotalTimeMs = 30000
$loadRates = @(0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021)

foreach ($lam in $loadRates) {
    $expectedSfc = [int][math]::Round($fixedTotalTimeMs * $lam)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Original-objective Ch3 MIP load: lambda=$lam, expected SFC=$expectedSfc, time_limit=${mipTimeLimitSeconds}s"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        experiment.seed=$seed `
        solver.solver_name=mip `
        training.num_train_epochs=0 `
        v_sim_setting.arrival_rate.lam=$lam `
        v_sim_setting.fixed_total_time_ms=$fixedTotalTimeMs `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=true `
        v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        p_net_setting.topology.cycle_snapshots=true `
        solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
        solver.objective.name=resource

    if ($LASTEXITCODE -ne 0) {
        throw "Improved Ch3 MIP load failed: lambda=$lam, exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Original-objective Ch3 MIP load tests finished."
