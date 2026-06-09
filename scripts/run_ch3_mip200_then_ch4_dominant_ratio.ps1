$ErrorActionPreference = "Stop"

# Run order:
# 1) Ch3 MIP load experiment: exactly call run_ch3_mip_load_lifetime2000_4000_fast.ps1.
# 2) Ch4 PPO-MGT dominant-service ratio experiment.

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$snapshotDurationMs = 100000
$numSnapshots = 10

# ============================================================
# 1) Ch3 MIP load experiment
# ============================================================
$mipScript = Join-Path $projectRoot "scripts\run_ch3_mip_load_lifetime2000_4000_fast.ps1"
if (-not (Test-Path -LiteralPath $mipScript)) {
    throw "Ch3 MIP load script not found: $mipScript"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[1/2] Ch3 MIP load: exactly running $mipScript"
Write-Host "============================================================"

& powershell -ExecutionPolicy Bypass -File $mipScript

if ($LASTEXITCODE -ne 0) {
    throw "Ch3 MIP load script failed: exit_code=$LASTEXITCODE"
}

# ============================================================
# 2) Ch4 dominant-service ratio experiment: PPO-MGT, testing only
# ============================================================
$ch4Model = "D:\any download\virne-main\virne-main\results\virne_ch4\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260525T014928-4449\models\model.pkl"
if (-not (Test-Path -LiteralPath $ch4Model)) {
    throw "Ch4 PPO-MGT pretrained model not found: $ch4Model"
}

$ch4Lambda = 0.004
$ch4NumRequests = 1000
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
        Write-Host "[2/2] Ch4 dominant ratio: dominant=$dominant, ratio=$ratio, service_ratios=$ratioCfg"
        Write-Host "============================================================"

        & $pythonExe main.py --config-name main_ch4 `
            experiment.seed=0 `
            solver.solver_name=ppo_gat_seq2seq+ `
            solver.pretrained_model_path=$ch4Model `
            training.num_train_epochs=0 `
            v_sim_setting.num_v_nets=$ch4NumRequests `
            v_sim_setting.arrival_rate.lam=$ch4Lambda `
            v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
            v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
            p_net_setting.topology.num_snapshots=$numSnapshots `
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
Write-Host "Finished: Ch3 MIP 200-request load + Ch4 dominant-ratio experiments."
Write-Host "============================================================"
