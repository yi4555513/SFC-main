$ErrorActionPreference = "Stop"
Set-Location "D:\any download\virne-main\virne-main"
$pythonExe = "D:\anaconda\envs\virne-main\python.exe"

# ============================================================
# Final missing experiments only
# 1) Ch3 ablation, medium load
# 2) Ch3 basic comparison, only missing ACO-META and MIP
# 3) Ch3 unified short-window load, only MIP
# 4) Ch4 dominant-service ratio tests, medium load
# ============================================================

$mediumLambda = 0.004
$mediumNumRequests = 1000
$snapshotDurationMs = 100000
$numSnapshots = 10
$mipTimeLimitSeconds = 5
$mipLatencyWeight = 2.0
$ch4PpoMgtModel = "D:\any download\virne-main\virne-main\results\virne_ch4\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260525T014928-4449\models\model.pkl"

Write-Host "============================================================"
Write-Host "Final missing experiments started"
Write-Host "Only missing runs are included. MIP time limit = ${mipTimeLimitSeconds}s."
Write-Host "============================================================"

# 1) Ch3 ablation under medium load
$ablationEpochs = 4
$ablationSolvers = @(
    "ppo_gat_seq2seq+_noGAT",
    "ppo_gat_seq2seq+_noTransformer",
    "ppo_gat_seq2seq+_noGAT_noTransformer"
)
foreach ($solver in $ablationSolvers) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[1/4] Ch3 ablation: solver=$solver, epochs=$ablationEpochs, lambda=$mediumLambda, SFC=$mediumNumRequests"
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
    if ($LASTEXITCODE -ne 0) { throw "Ch3 ablation failed: solver=$solver, exit_code=$LASTEXITCODE" }
}

# 2) Ch3 basic medium-load comparison: only missing ACO-META and MIP
# PPO-MLP+ and PPO-MGT have already been run separately.
$basicMissingAlgorithms = @("aco_meta", "mip")
foreach ($solver in $basicMissingAlgorithms) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[2/4] Ch3 basic medium-load missing run: solver=$solver, lambda=$mediumLambda, SFC=$mediumNumRequests"
    Write-Host "============================================================"
    if ($solver -eq "mip") {
        & $pythonExe main.py --config-name main_ch3 `
            experiment.seed=0 `
            solver.solver_name=mip `
            training.num_train_epochs=0 `
            v_sim_setting.num_v_nets=$mediumNumRequests `
            v_sim_setting.arrival_rate.lam=$mediumLambda `
            v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
            v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
            p_net_setting.topology.num_snapshots=$numSnapshots `
            p_net_setting.topology.cycle_snapshots=true `
            solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
            solver.objective.name=resource_latency `
            solver.objective.resource_weight=1.0 `
            solver.objective.latency_weight=$mipLatencyWeight `
            solver.objective.enforce_latency_constraint=false `
            solver.mip_debug_print_solution=false
    } else {
        & $pythonExe main.py --config-name main_ch3 `
            experiment.seed=0 `
            solver.solver_name=$solver `
            training.num_train_epochs=0 `
            v_sim_setting.num_v_nets=$mediumNumRequests `
            v_sim_setting.arrival_rate.lam=$mediumLambda `
            v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
            v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
            p_net_setting.topology.num_snapshots=$numSnapshots `
            p_net_setting.topology.cycle_snapshots=true
    }
    if ($LASTEXITCODE -ne 0) { throw "Ch3 basic missing run failed: solver=$solver, exit_code=$LASTEXITCODE" }
}

# 3) Ch3 unified short-window load: only MIP
# Other algorithms are assumed to have been run under the same 30000 ms setting.
$loadRates = @(0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021)
$loadFixedTotalTimeMs = 30000
$loadNumSnapshots = 3
foreach ($lam in $loadRates) {
    $expectedSfc = [int][Math]::Round($loadFixedTotalTimeMs * [double]$lam)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[3/4] Ch3 unified short-window load MIP: lambda=$lam, expected SFC=$expectedSfc, time_limit=${mipTimeLimitSeconds}s"
    Write-Host "============================================================"
    & $pythonExe main.py --config-name main_ch3 `
        experiment.seed=0 `
        solver.solver_name=mip `
        training.num_train_epochs=0 `
        v_sim_setting.arrival_rate.lam=$lam `
        v_sim_setting.fixed_total_time_ms=$loadFixedTotalTimeMs `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=true `
        p_net_setting.topology.num_snapshots=$loadNumSnapshots `
        solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
        solver.objective.name=resource_latency `
        solver.objective.resource_weight=1.0 `
        solver.objective.latency_weight=$mipLatencyWeight `
        solver.objective.enforce_latency_constraint=false `
        solver.mip_debug_print_solution=false
    if ($LASTEXITCODE -ne 0) { throw "Ch3 unified load MIP failed: lambda=$lam, exit_code=$LASTEXITCODE" }
}

# 4) Ch4 dominant-service ratio tests under medium load
$dominantRatios = @(0.25, 0.40, 0.55, 0.70)
$services = @("delay_sensitive", "bandwidth_sensitive", "reliability_sensitive", "compute_sensitive")
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
        Write-Host "[4/4] Ch4 dominant ratio: dominant=$dominant, ratio=$ratio, service_ratios=$ratioCfg"
        Write-Host "============================================================"
        & $pythonExe main.py --config-name main_ch4 `
            experiment.seed=0 `
            solver.solver_name=ppo_gat_seq2seq+ `
            solver.pretrained_model_path=$ch4PpoMgtModel `
            training.num_train_epochs=0 `
            v_sim_setting.num_v_nets=$mediumNumRequests `
            v_sim_setting.arrival_rate.lam=$mediumLambda `
            v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
            v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
            p_net_setting.topology.num_snapshots=$numSnapshots `
            p_net_setting.topology.cycle_snapshots=true `
            v_sim_setting.service_qos.exact_ratio=true `
            "v_sim_setting.service_qos.service_ratios=$ratioCfg"
        if ($LASTEXITCODE -ne 0) { throw "Ch4 dominant ratio failed: dominant=$dominant ratio=$ratio exit_code=$LASTEXITCODE" }
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "All final missing experiments finished successfully."
Write-Host "============================================================"
