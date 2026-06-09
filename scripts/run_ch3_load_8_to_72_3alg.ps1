$ErrorActionPreference = "Stop"

$repo = "D:\any download\virne-main\virne-main"
$python = "D:\anaconda\envs\virne-main\python.exe"

$seed = "1"
$numRequests = "1000"
$snapshotDurationMs = "60000"
$numSnapshots = "20"
$ratesPerSecond = @(8, 16, 24, 32, 40, 48, 56, 64, 72)

$ppoMlpModel = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_mlp+\DESKTOP-2NM5LP6-20260526T232004-4022\models\model.pkl"
$ppoMgtModel = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260526T165819-1245\models\model.pkl"

Set-Location -LiteralPath $repo

$jobs = @(
    @{
        label = "PPO-MLP"
        solver = "ppo_mlp+"
        extra = @("solver.pretrained_model_path=$ppoMlpModel")
    },
    @{
        label = "PPO-MGT"
        solver = "ppo_gat_seq2seq+"
        extra = @("solver.pretrained_model_path=$ppoMgtModel")
    },
    @{
        label = "ACO-META"
        solver = "aco_meta"
        extra = @()
    }
)

Write-Host "============================================================"
Write-Host "Chapter 3 load experiment: PPO-MGT, PPO-MLP, ACO-META"
Write-Host "Rates: $($ratesPerSecond -join ', ') requests/s"
Write-Host "SFC requests: $numRequests"
Write-Host "Snapshot duration: $snapshotDurationMs ms"
Write-Host "Snapshots: $numSnapshots"
Write-Host "Seed: $seed"
Write-Host "============================================================"

foreach ($rate in $ratesPerSecond) {
    $lambda = "{0:0.000}" -f ($rate / 1000.0)

    foreach ($job in $jobs) {
        Write-Host ""
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Running $($job.label), rate=$rate requests/s, lambda=$lambda"

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

        & $python @argsList
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            throw "$($job.label) failed at rate=$rate requests/s with exit code $exitCode"
        }

        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished $($job.label), rate=$rate requests/s"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Finished all load experiments."
Write-Host "Logs saved to:"
Write-Host $logDir
Write-Host "Results are under:"
Write-Host (Join-Path $repo "results\virne_ch3")
Write-Host "============================================================"
