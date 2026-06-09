# ============================================================
# Wait for Ch3 ablation+MIP-load script to finish,
# then run Ch3 basic MIP mid-load and Ch4 dominant-ratio tests.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\wait_then_run_ch3_mip_basic_ch4_ratio.ps1
# ============================================================

$ErrorActionPreference = "Stop"

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$previousScriptName = "run_ch3_ablation10_then_mip_load.ps1"
$checkIntervalSeconds = 60

function Get-RunningPreviousExperimentProcesses {
    $projectPattern = "virne-main"
    $processes = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.ProcessId -ne $PID -and
        (
            ($_.CommandLine -like "*$previousScriptName*") -or
            (($_.CommandLine -like "*main.py*") -and ($_.CommandLine -like "*$projectPattern*"))
        )
    }
    return $processes
}

Write-Host "============================================================"
Write-Host "Waiting for previous experiment to finish: $previousScriptName"
Write-Host "It will also wait for python main.py processes in this VIRNE project."
Write-Host "============================================================"

while ($true) {
    $running = @(Get-RunningPreviousExperimentProcesses)
    if ($running.Count -eq 0) {
        Write-Host "No previous experiment process detected. Continue."
        break
    }

    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$now] Previous experiment still running. Detected $($running.Count) process(es). Wait ${checkIntervalSeconds}s..."
    foreach ($p in $running | Select-Object -First 5) {
        Write-Host "  PID=$($p.ProcessId) Name=$($p.Name)"
    }
    Start-Sleep -Seconds $checkIntervalSeconds
}

# -------------------- 1. Ch3 basic MIP mid-load --------------------
Write-Host ""
Write-Host "============================================================"
Write-Host "[1/2] Running Ch3 basic MIP mid-load: lambda=0.004, SFC=1000, mip_time_limit=5s"
Write-Host "============================================================"

& $pythonExe main.py --config-name main_ch3 `
    experiment.seed=0 `
    solver.solver_name=mip `
    training.num_train_epochs=0 `
    v_sim_setting.num_v_nets=1000 `
    v_sim_setting.arrival_rate.lam=0.004 `
    v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
    v_sim_setting.snapshot_duration_ms=100000 `
    p_net_setting.topology.num_snapshots=10 `
    p_net_setting.topology.cycle_snapshots=true `
    solver.mip_time_limit_seconds=5 `
    solver.objective.name=resource_latency `
    solver.objective.resource_weight=1.0 `
    solver.objective.latency_weight=2.0 `
    solver.objective.enforce_latency_constraint=false

if ($LASTEXITCODE -ne 0) {
    throw "Ch3 basic MIP mid-load failed: exit_code=$LASTEXITCODE"
}

# -------------------- 2. Ch4 dominant-service ratio tests --------------------
$ch4Model = "D:\any download\virne-main\virne-main\results\virne_ch4\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260525T014928-4449\models\model.pkl"
if (-not (Test-Path -LiteralPath $ch4Model)) {
    throw "Ch4 PPO-MGT pretrained model not found: $ch4Model"
}

$dominantRatios = @(0.25, 0.40, 0.55, 0.70)
$services = @(
    "delay_sensitive",
    "bandwidth_sensitive",
    "reliability_sensitive",
    "compute_sensitive"
)

foreach ($dominant in $services) {
    foreach ($ratio in $dominantRatios) {
        $bg = [Math]::Round((1.0 - [double]$ratio) / 3.0, 6)
        $d = $bg; $b = $bg; $r = $bg; $c = $bg
        if ($dominant -eq "delay_sensitive") { $d = $ratio }
        if ($dominant -eq "bandwidth_sensitive") { $b = $ratio }
        if ($dominant -eq "reliability_sensitive") { $r = $ratio }
        if ($dominant -eq "compute_sensitive") { $c = $ratio }

        $ratioCfg = "{delay_sensitive:$d,bandwidth_sensitive:$b,reliability_sensitive:$r,compute_sensitive:$c}"

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "[2/2] Running Ch4 dominant ratio: dominant=$dominant, ratio=$ratio"
        Write-Host "service_ratios=$ratioCfg"
        Write-Host "============================================================"

        & $pythonExe main.py --config-name main_ch4 `
            experiment.seed=0 `
            solver.solver_name=ppo_gat_seq2seq+ `
            solver.pretrained_model_path=$ch4Model `
            training.num_train_epochs=0 `
            v_sim_setting.num_v_nets=1000 `
            v_sim_setting.arrival_rate.lam=0.004 `
            v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
            v_sim_setting.snapshot_duration_ms=100000 `
            p_net_setting.topology.num_snapshots=10 `
            p_net_setting.topology.cycle_snapshots=true `
            v_sim_setting.service_qos.exact_ratio=true `
            "v_sim_setting.service_qos.service_ratios=$ratioCfg"

        if ($LASTEXITCODE -ne 0) {
            throw "Ch4 dominant ratio failed: dominant=$dominant ratio=$ratio exit_code=$LASTEXITCODE"
        }
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "All done: Ch3 basic MIP mid-load + Ch4 dominant-ratio tests."
Write-Host "============================================================"
