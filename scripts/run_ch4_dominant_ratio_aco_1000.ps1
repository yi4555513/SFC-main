$ErrorActionPreference = "Stop"

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$lambda = 0.004
$numRequests = 1000
$snapshotDurationMs = 100000
$numSnapshots = 10

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

foreach ($job in $jobs) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Ch4 dominant ratio: algorithm=ACO-META, label=$($job.Label), ratio=$($job.Ratio), service_ratios=$($job.ServiceRatios)"
    Write-Host "============================================================"

    & $pythonExe main.py `
        --config-name main_ch4 `
        hydra.run.dir="$projectRoot/results" `
        experiment.seed=0 `
        experiment.run_id=auto `
        solver.solver_name=aco_meta `
        training.num_train_epochs=0 `
        v_sim_setting.num_v_nets=$numRequests `
        v_sim_setting.arrival_rate.lam=$lambda `
        ++v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
        v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        ++p_net_setting.topology.cycle_snapshots=true `
        v_sim_setting.service_qos.exact_ratio=true `
        "v_sim_setting.service_qos.service_ratios=$($job.ServiceRatios)"

    if ($LASTEXITCODE -ne 0) {
        throw "Ch4 ACO-META dominant ratio failed: label=$($job.Label), ratio=$($job.Ratio), exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Finished: Ch4 dominant-ratio tests for ACO-META with 1000 requests per run."
