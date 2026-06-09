$ErrorActionPreference = "Stop"

# Ch3 MIP low-arrival load test.
# Generate 1000 SFC requests with seed=0, then simulate only the first 200.
# Arrival rates: 0.001, 0.002, ..., 0.007.

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$mipConfigDir = "results/virne_ch3/mip/DESKTOP-2NM5LP6-20260526T134031-9051"
$loadRates = @(0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007)
$generatedRequests = 1000
$simulatedPrefixRequests = 200
$mipTimeLimitSeconds = 5
$snapshotDurationMs = 100000
$numSnapshots = 10

foreach ($lam in $loadRates) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Ch3 MIP low-arrival load: lambda=$lam, generate=$generatedRequests, simulate first=$simulatedPrefixRequests, lifetime=2000~4000, time_limit=${mipTimeLimitSeconds}s"
    Write-Host "============================================================"

    & $pythonExe main.py `
        --config-path $mipConfigDir `
        --config-name config `
        hydra.run.dir=$projectRoot/results `
        experiment.seed=0 `
        experiment.run_id=auto `
        experiment.save_root_dir=virne_ch3/ `
        v_sim_setting.num_v_nets=$generatedRequests `
        ++v_sim_setting.truncate_num_v_nets=$simulatedPrefixRequests `
        v_sim_setting.arrival_rate.lam=$lam `
        v_sim_setting.lifetime.low=2000 `
        v_sim_setting.lifetime.high=4000 `
        ++v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
        ++v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        p_net_setting.topology.num_nodes=66 `
        p_net_setting.topology.planes=6 `
        p_net_setting.topology.nums_per_plane=11 `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        ++p_net_setting.topology.cycle_snapshots=true `
        solver.objective.name=resource_latency `
        solver.objective.resource_weight=1.0 `
        solver.objective.latency_weight=0.1 `
        ++solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
        ++solver.fix_endpoint_vnfs=true `
        ++solver.include_endpoint_latency=true

    if ($LASTEXITCODE -ne 0) {
        throw "Ch3 MIP low-arrival load failed: lambda=$lam, exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Finished: Ch3 MIP low-arrival 1000-prefix-200 load tests."
