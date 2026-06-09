$ErrorActionPreference = "Stop"
Set-Location "D:\any download\virne-main\virne-main"
$pythonExe = "D:\anaconda\envs\virne-main\python.exe"

# MIP load test: same arrival-rate points, but only about 200 SFC requests per point.
# fixed_total_time_ms is adjusted as num_requests / lambda so each point keeps the target arrival rate.
$rates = @(0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021)
$numRequests = 200
$numSnapshots = 3
$timeLimitSeconds = 5

foreach ($lam in $rates) {
    $totalTimeMs = [int][Math]::Round($numRequests / $lam)
    Write-Host "============================================================"
    Write-Host "Running Ch3 MIP: lambda=$lam, num_requests=$numRequests, total_time_ms=$totalTimeMs, snapshots=$numSnapshots, mip_time_limit=${timeLimitSeconds}s"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        solver.solver_name=mip `
        v_sim_setting.arrival_rate.lam=$lam `
        v_sim_setting.fixed_total_time_ms=$totalTimeMs `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=true `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        solver.mip_time_limit_seconds=$timeLimitSeconds `
        solver.objective.name=resource_latency `
        solver.objective.resource_weight=1.0 `
        solver.objective.latency_weight=1.0 `
        solver.objective.enforce_latency_constraint=false `
        solver.objective.latency_constraint_margin=0.0 `
        solver.mip_debug_print_solution=false
}
