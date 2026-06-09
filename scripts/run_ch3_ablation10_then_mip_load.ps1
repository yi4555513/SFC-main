# ============================================================
# Ch3 ablation 10 epochs, then MIP load test
# 1) Ablation: w/o M-GAT, w/o Transformer, w/o Both
#    Medium load: lambda=0.004, 1000 SFC, 10 snapshots, 100s/snapshot
# 2) MIP load: lambda=0.003~0.021, fixed window=30000ms, MIP time limit=5s
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

# -------------------- 1. Ch3 ablation --------------------
$ablationEpochs = 10
$mediumLambda = 0.004
$mediumNumRequests = 1000

$ablationSolvers = @(
    "ppo_gat_seq2seq+_noGAT",
    "ppo_gat_seq2seq+_noTransformer",
    "ppo_gat_seq2seq+_noGAT_noTransformer"
)

foreach ($solver in $ablationSolvers) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[1/2] Ch3 ablation: solver=$solver, epochs=$ablationEpochs, lambda=$mediumLambda, SFC=$mediumNumRequests"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        experiment.seed=$seed `
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

# -------------------- 2. MIP load test --------------------
$mipTimeLimitSeconds = 5
$mipFixedTotalTimeMs = 30000
$loadRates = @(0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021)

foreach ($lam in $loadRates) {
    $expectedSfc = [int][math]::Round($mipFixedTotalTimeMs * $lam)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[2/2] Ch3 MIP load: lambda=$lam, fixed_total_time_ms=$mipFixedTotalTimeMs, expected SFC=$expectedSfc, time_limit=${mipTimeLimitSeconds}s"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        experiment.seed=$seed `
        solver.solver_name=mip `
        training.num_train_epochs=0 `
        v_sim_setting.arrival_rate.lam=$lam `
        v_sim_setting.fixed_total_time_ms=$mipFixedTotalTimeMs `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=true `
        v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        p_net_setting.topology.cycle_snapshots=true `
        solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
        solver.objective.name=resource_latency `
        solver.objective.resource_weight=1.0 `
        solver.objective.latency_weight=0.1

    if ($LASTEXITCODE -ne 0) {
        throw "Ch3 MIP load failed: lambda=$lam, exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "All done: Ch3 ablation 10 epochs + MIP load test."
Write-Host "============================================================"
