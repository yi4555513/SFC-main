$ErrorActionPreference = "Stop"

# Wait for the currently running Ch4 ACO-META dominant-ratio experiment, then
# run the same dominant-ratio settings with MIP on a smaller sample.

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$acoScriptName = "run_ch4_dominant_ratio_aco_1000.ps1"
$pollSeconds = 60

function Get-AcoDominantProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $cmd = [string]$_.CommandLine
        (
            $cmd -like "*$acoScriptName*" -or
            ($cmd -like "*main.py*" -and $cmd -like "*solver.solver_name=aco_meta*" -and $cmd -like "*main_ch4*")
        ) -and
        $cmd -notlike "*wait_aco_then_run_ch4_mip_dominant_ratio_100.ps1*"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Waiting for Ch4 ACO-META dominant-ratio experiment to finish"
Write-Host "============================================================"

while ($true) {
    $running = @(Get-AcoDominantProcesses)
    if ($running.Count -eq 0) {
        break
    }

    $timeText = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timeText] ACO-META is still running. Process count: $($running.Count). Check again in ${pollSeconds}s."
    Start-Sleep -Seconds $pollSeconds
}

Write-Host ""
Write-Host "ACO-META process has ended. Starting Ch4 MIP dominant-ratio experiment."

# ============================================================
# Ch4 dominant-service ratio experiment: MIP, 200 requests/run
# ============================================================
$lambda = 0.004
$numRequests = 200
$snapshotDurationMs = 100000
$numSnapshots = 10
$mipTimeLimitSeconds = 5

$services = @(
    "delay_sensitive",
    "bandwidth_sensitive",
    "reliability_sensitive",
    "compute_sensitive"
)

$jobs = @()
$jobs += @{
    Label = "uniform"
    Ratio = 0.25
    ServiceRatios = "{delay_sensitive:0.25,bandwidth_sensitive:0.25,reliability_sensitive:0.25,compute_sensitive:0.25}"
}

foreach ($dominant in $services) {
    foreach ($ratio in @(0.40, 0.55, 0.70)) {
        $bg = [Math]::Round((1.0 - [double]$ratio) / 3.0, 6)
        $d = $bg; $b = $bg; $r = $bg; $c = $bg
        if ($dominant -eq "delay_sensitive") { $d = $ratio }
        if ($dominant -eq "bandwidth_sensitive") { $b = $ratio }
        if ($dominant -eq "reliability_sensitive") { $r = $ratio }
        if ($dominant -eq "compute_sensitive") { $c = $ratio }
        $jobs += @{
            Label = $dominant
            Ratio = $ratio
            ServiceRatios = "{delay_sensitive:$d,bandwidth_sensitive:$b,reliability_sensitive:$r,compute_sensitive:$c}"
        }
    }
}

$index = 0
foreach ($job in $jobs) {
    $index += 1
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[$index/$($jobs.Count)] Ch4 dominant ratio: algorithm=MIP, label=$($job.Label), ratio=$($job.Ratio), requests=$numRequests, time_limit=${mipTimeLimitSeconds}s"
    Write-Host "service_ratios=$($job.ServiceRatios)"
    Write-Host "============================================================"

    & $pythonExe main.py `
        --config-name main_ch4 `
        hydra.run.dir="$projectRoot/results" `
        experiment.seed=0 `
        experiment.run_id=auto `
        solver.solver_name=mip `
        training.num_train_epochs=0 `
        v_sim_setting.num_v_nets=$numRequests `
        v_sim_setting.arrival_rate.lam=$lambda `
        ++v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
        v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        ++p_net_setting.topology.cycle_snapshots=true `
        v_sim_setting.service_qos.exact_ratio=true `
        solver.objective.name=resource_latency `
        solver.objective.resource_weight=1.0 `
        solver.objective.latency_weight=0.1 `
        solver.fix_endpoint_vnfs=true `
        solver.include_endpoint_latency=true `
        ++solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
        "v_sim_setting.service_qos.service_ratios=$($job.ServiceRatios)"

    if ($LASTEXITCODE -ne 0) {
        throw "Ch4 MIP dominant ratio failed: label=$($job.Label), ratio=$($job.Ratio), exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Finished: Ch4 MIP dominant-ratio tests with $numRequests requests per run."
Write-Host "============================================================"
