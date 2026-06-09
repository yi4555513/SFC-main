$ErrorActionPreference = "Stop"

Set-Location "D:\any download\virne-main\virne-main"

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

# Chapter 4 dominant-service ratio experiments for PPO-MGT.
# It uses the trained Ch4 PPO-MGT model and only runs testing.
$model = "D:\any download\virne-main\virne-main\results\virne_ch4\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260525T014928-4449\models\model.pkl"
if (-not (Test-Path -LiteralPath $model)) {
    throw "Ch4 PPO-MGT pretrained model not found: $model"
}

# Medium-load testing setting: lambda=0.004, 1000 SFCs, 100 seconds per snapshot.
$lambda = 0.004
$numRequests = 1000
$snapshotDurationMs = 100000
$numSnapshots = 10

# Dominant ratios on x-axis. Background services equally share the remaining proportion.
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
        Write-Host "============================================================"
        Write-Host "Running Ch4 dominant ratio: dominant=$dominant, ratio=$ratio, service_ratios=$ratioCfg"
        Write-Host "============================================================"

        & $pythonExe main.py --config-name main_ch4 `
            hydra.run.dir="$PWD/results" `
            experiment.seed=0 `
            solver.solver_name=ppo_gat_seq2seq+ `
            solver.pretrained_model_path=$model `
            training.num_train_epochs=0 `
            v_sim_setting.num_v_nets=$numRequests `
            v_sim_setting.arrival_rate.lam=$lambda `
            ++v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
            v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
            p_net_setting.topology.num_snapshots=$numSnapshots `
            ++p_net_setting.topology.cycle_snapshots=true `
            v_sim_setting.service_qos.exact_ratio=true `
            "v_sim_setting.service_qos.service_ratios=$ratioCfg"

        if ($LASTEXITCODE -ne 0) {
            throw "Ch4 dominant ratio run failed: dominant=$dominant ratio=$ratio exit_code=$LASTEXITCODE"
        }
    }
}

Write-Host "============================================================"
Write-Host "All Ch4 dominant-ratio PPO-MGT tests finished."
Write-Host "============================================================"
