$ErrorActionPreference = "Stop"

$repo = "D:\any download\virne-main\virne-main"
$python = "D:\anaconda\envs\virne-main\python.exe"

$lambda = "0.012"          # 12 requests/s; internal unit is requests/ms
$numRequests = "1000"
$snapshotDurationMs = "20000"
$numSnapshots = "10"
$seed = "1"

$ppoMlpModel = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_mlp+\DESKTOP-2NM5LP6-20260526T232004-4022\models\model.pkl"
$ppoMgtModel = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260526T165819-1245\models\model.pkl"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = Join-Path $repo "paper_outputs\ch3_basic_1000_lambda12_seed${seed}_$timestamp"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Set-Location -LiteralPath $repo

$jobs = @(
    @{
        name = "PPO-Baseline"
        solver = "ppo_mlp+"
        extra = @("solver.pretrained_model_path=$ppoMlpModel")
    },
    @{
        name = "PPO-MGT"
        solver = "ppo_gat_seq2seq+"
        extra = @("solver.pretrained_model_path=$ppoMgtModel")
    },
    @{
        name = "ACO-META"
        solver = "aco_meta"
        extra = @()
    }
)

Write-Host "============================================================"
Write-Host "Chapter 3 basic experiment: 1000 SFC, 12 req/s"
Write-Host "Seed: $seed"
Write-Host "Output: $outputDir"
Write-Host "============================================================"

foreach ($job in $jobs) {
    Write-Host ""
    Write-Host "Running $($job.name) ..."

    $argsList = @(
        "main.py",
        "--config-name", "main_ch3",
        "experiment.seed=$seed",
        "solver.solver_name=$($job.solver)",
        "training.num_train_epochs=0",
        "v_sim_setting.num_v_nets=$numRequests",
        "v_sim_setting.auto_num_v_nets_from_arrival_rate=false",
        "v_sim_setting.arrival_rate.lam=$lambda",
        "v_sim_setting.snapshot_duration_ms=$snapshotDurationMs",
        "p_net_setting.topology.num_snapshots=$numSnapshots",
        "p_net_setting.topology.cycle_snapshots=true"
    ) + $job.extra

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @argsList
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference

    if ($exitCode -ne 0) {
        throw "$($job.name) failed with exit code $exitCode"
    }
}

Write-Host ""
Write-Host "Plotting basic figures ..."
& $python "scripts\plot_ch3_basic_1000_lambda12_3alg.py" `
    --lambda-rate $lambda `
    --num-sfc $numRequests `
    --snapshot-duration-ms $snapshotDurationMs `
    --num-snapshots $numSnapshots `
    --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
    throw "Plotting failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Done. Figures and selected CSV paths are in:"
Write-Host $outputDir
Write-Host "============================================================"
