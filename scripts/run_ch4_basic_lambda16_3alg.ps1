$ErrorActionPreference = "Stop"

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$ppoMlpModel = "D:\any download\virne-main\virne-main\results\virne_ch4\ppo_mlp+\DESKTOP-2NM5LP6-20260525T011303-1264\models\model.pkl"
$ppoMgtModel = "D:\any download\virne-main\virne-main\results\virne_ch4\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260525T014928-4449\models\model.pkl"

if (-not (Test-Path -LiteralPath $ppoMlpModel)) {
    throw "Ch4 PPO-Baseline pretrained model not found: $ppoMlpModel"
}
if (-not (Test-Path -LiteralPath $ppoMgtModel)) {
    throw "Ch4 PPO-MGT pretrained model not found: $ppoMgtModel"
}

$seed = 0
$lambda = 0.016
$numRequests = 1000
$lifetimeLow = 4000
$lifetimeHigh = 8000
$snapshotDurationMs = 60000
$numSnapshots = 20
$serviceRatios = "uniform"

$algorithms = @(
    @{
        Label = "ACO-META"
        Solver = "aco_meta"
        PretrainedModel = ""
    },
    @{
        Label = "PPO-Baseline"
        Solver = "ppo_mlp+"
        PretrainedModel = $ppoMlpModel
    },
    @{
        Label = "PPO-MGT"
        Solver = "ppo_gat_seq2seq+"
        PretrainedModel = $ppoMgtModel
    }
)

Write-Host "============================================================"
Write-Host "Chapter 4 basic uniform-service experiment"
Write-Host "Algorithms: ACO-META, PPO-Baseline, PPO-MGT"
Write-Host "MIP is skipped."
Write-Host "lambda=$lambda, SFC=$numRequests, lifetime=${lifetimeLow}-${lifetimeHigh}ms, snapshot=${snapshotDurationMs}ms, snapshots=$numSnapshots"
Write-Host "============================================================"

foreach ($alg in $algorithms) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running Ch4 basic: algorithm=$($alg.Label), lambda=$lambda"
    Write-Host "============================================================"

    $argsList = @(
        "main.py",
        "--config-name", "main_ch4",
        "hydra.run.dir=$projectRoot/results",
        "experiment.seed=$seed",
        "solver.solver_name=$($alg.Solver)",
        "training.num_train_epochs=0",
        "v_sim_setting.num_v_nets=$numRequests",
        "v_sim_setting.arrival_rate.lam=$lambda",
        "v_sim_setting.lifetime.low=$lifetimeLow",
        "v_sim_setting.lifetime.high=$lifetimeHigh",
        "++v_sim_setting.auto_num_v_nets_from_arrival_rate=false",
        "v_sim_setting.snapshot_duration_ms=$snapshotDurationMs",
        "p_net_setting.topology.num_snapshots=$numSnapshots",
        "++p_net_setting.topology.cycle_snapshots=true",
        "v_sim_setting.service_qos.exact_ratio=true",
        "v_sim_setting.service_qos.service_ratios=$serviceRatios"
    )

    if ($alg.PretrainedModel) {
        $argsList += "solver.pretrained_model_path=$($alg.PretrainedModel)"
    }

    & $pythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Ch4 basic run failed: algorithm=$($alg.Label), exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Finished: Ch4 basic uniform-service tests for 3 algorithms."
Write-Host "============================================================"
