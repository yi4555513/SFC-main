$ErrorActionPreference = "Stop"
Set-Location "D:\any download\virne-main\virne-main"
$pythonExe = "D:\anaconda\envs\virne-main\python.exe"

Write-Host "============================================================"
Write-Host "Paper extra experiments started"
Write-Host "Medium-load setting: 1000 SFC, lambda=0.004, snapshot_duration=100s"
Write-Host "Order: Ch3 ablation -> Ch3 MIP stable load -> Ch4 uniform -> Ch4 dominant ratios"
Write-Host "============================================================"

# Common medium-load setting for basic/ablation/Ch4 ratio experiments.
# 1000 requests at lambda=0.004 last about 250 seconds on average, so they cover about 3 original 100s snapshots.
# cycle_snapshots=true is only a safety switch if a run exceeds the available snapshot sequence.
$mediumLambda = 0.004
$mediumNumRequests = 1000
$snapshotDurationMs = 100000
$numSnapshots = 10

# ============================================================
# 1) Chapter 3 ablation quick rerun under medium load
# ============================================================
$ablationEpochs = 4
$ablationSolvers = @(
    "ppo_gat_seq2seq+_noGAT",
    "ppo_gat_seq2seq+_noTransformer",
    "ppo_gat_seq2seq+_noGAT_noTransformer"
)

foreach ($solver in $ablationSolvers) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[1/4] Ch3 medium-load ablation: solver=$solver, epochs=$ablationEpochs, 1000 SFC, lambda=$mediumLambda"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        experiment.seed=0 `
        solver.solver_name=$solver `
        training.num_train_epochs=$ablationEpochs `
        v_sim_setting.num_v_nets=$mediumNumRequests `
        v_sim_setting.arrival_rate.lam=$mediumLambda `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
        v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        p_net_setting.topology.cycle_snapshots=true

    if ($LASTEXITCODE -ne 0) {
        throw "Ch3 ablation failed: solver=$solver, exit_code=$LASTEXITCODE"
    }
}

# ============================================================
# 2) Chapter 3 MIP stable load tests
#    Keep this as the load curve experiment, not the 1000-SFC medium basic setting.
# ============================================================
$mipRates = @(0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021)
$mipNumRequests = 500
$mipNumSnapshots = 3
$mipTimeLimitSeconds = 5
$mipLatencyWeight = 2.0

foreach ($lam in $mipRates) {
    $totalTimeMs = [int][Math]::Round($mipNumRequests / [double]$lam)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[2/4] Ch3 MIP stable load: lambda=$lam, num_requests=$mipNumRequests, total_time_ms=$totalTimeMs, snapshots=$mipNumSnapshots, time_limit=${mipTimeLimitSeconds}s"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        experiment.seed=0 `
        solver.solver_name=mip `
        training.num_train_epochs=0 `
        v_sim_setting.arrival_rate.lam=$lam `
        v_sim_setting.num_v_nets=$mipNumRequests `
        v_sim_setting.fixed_total_time_ms=$totalTimeMs `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=true `
        p_net_setting.topology.num_snapshots=$mipNumSnapshots `
        solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
        solver.objective.name=resource_latency `
        solver.objective.resource_weight=1.0 `
        solver.objective.latency_weight=$mipLatencyWeight `
        solver.objective.enforce_latency_constraint=false `
        solver.objective.latency_constraint_margin=0.0 `
        solver.mip_debug_print_solution=false

    if ($LASTEXITCODE -ne 0) {
        throw "Ch3 MIP stable load failed: lambda=$lam, exit_code=$LASTEXITCODE"
    }
}

# ============================================================
# 3) Chapter 4 uniform multi-service test under medium load
# ============================================================
$ch4Model = "D:\any download\virne-main\virne-main\results\virne_ch4\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260525T014928-4449\models\model.pkl"

Write-Host ""
Write-Host "============================================================"
Write-Host "[3/4] Ch4 medium-load uniform multi-service test"
Write-Host "============================================================"

& $pythonExe main.py --config-name main_ch4 `
    experiment.seed=0 `
    solver.solver_name=ppo_gat_seq2seq+ `
    solver.pretrained_model_path=$ch4Model `
    training.num_train_epochs=0 `
    v_sim_setting.num_v_nets=$mediumNumRequests `
    v_sim_setting.arrival_rate.lam=$mediumLambda `
    v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
    v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
    p_net_setting.topology.num_snapshots=$numSnapshots `
    p_net_setting.topology.cycle_snapshots=true `
    v_sim_setting.service_qos.exact_ratio=true `
    v_sim_setting.service_qos.service_ratios=uniform

if ($LASTEXITCODE -ne 0) {
    throw "Ch4 uniform medium-load test failed: exit_code=$LASTEXITCODE"
}

# ============================================================
# 4) Chapter 4 dominant service-ratio tests under medium load
# ============================================================
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
        Write-Host "[4/4] Ch4 medium-load dominant ratio: dominant=$dominant, ratio=$ratio, service_ratios=$ratioCfg"
        Write-Host "============================================================"

        & $pythonExe main.py --config-name main_ch4 `
            experiment.seed=0 `
            solver.solver_name=ppo_gat_seq2seq+ `
            solver.pretrained_model_path=$ch4Model `
            training.num_train_epochs=0 `
            v_sim_setting.num_v_nets=$mediumNumRequests `
            v_sim_setting.arrival_rate.lam=$mediumLambda `
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
Write-Host "All paper extra experiments finished successfully."
Write-Host "============================================================"
