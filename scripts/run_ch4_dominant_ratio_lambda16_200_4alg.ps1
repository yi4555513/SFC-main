param(
    [switch]$DryRun,
    [switch]$SkipMip,
    [switch]$NoDuplicateUniform
)

$ErrorActionPreference = "Stop"

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$ppoMlpModel = Join-Path $projectRoot "results\virne_ch4\ppo_mlp+\DESKTOP-2NM5LP6-20260525T011303-1264\models\model.pkl"
$ppoMgtModel = Join-Path $projectRoot "results\virne_ch4\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260525T014928-4449\models\model.pkl"

if (-not (Test-Path -LiteralPath $ppoMlpModel)) {
    throw "Ch4 PPO-Baseline pretrained model not found: $ppoMlpModel"
}
if (-not (Test-Path -LiteralPath $ppoMgtModel)) {
    throw "Ch4 PPO-MGT pretrained model not found: $ppoMgtModel"
}

# Same load/dynamic-topology setting as the current Chapter 4 basic experiment.
$seed = 0
$lambda = 0.016
$numRequests = 200
$lifetimeLow = 4000
$lifetimeHigh = 8000
$snapshotDurationMs = 60000
$numSnapshots = 20
$mipTimeLimitSeconds = 5

$services = @(
    "delay_sensitive",
    "bandwidth_sensitive",
    "reliability_sensitive",
    "compute_sensitive"
)
$dominantRatios = @(0.25, 0.40, 0.55, 0.70)

$algorithms = @()
if (-not $SkipMip) {
    $algorithms += @{
        Label = "MIP"
        Solver = "mip"
        PretrainedModel = ""
        IsMip = $true
    }
}
$algorithms += @(
    @{
        Label = "ACO-META"
        Solver = "aco_meta"
        PretrainedModel = ""
        IsMip = $false
    },
    @{
        Label = "PPO-Baseline"
        Solver = "ppo_mlp+"
        PretrainedModel = $ppoMlpModel
        IsMip = $false
    },
    @{
        Label = "PPO-MGT"
        Solver = "ppo_gat_seq2seq+"
        PretrainedModel = $ppoMgtModel
        IsMip = $false
    }
)

$jobs = @()
$uniformAdded = $false
foreach ($dominant in $services) {
    foreach ($ratio in $dominantRatios) {
        if ($NoDuplicateUniform -and ([double]$ratio -eq 0.25) -and $uniformAdded) {
            continue
        }

        $bg = [Math]::Round((1.0 - [double]$ratio) / 3.0, 6)
        $d = $bg
        $b = $bg
        $r = $bg
        $c = $bg
        if ($dominant -eq "delay_sensitive") { $d = $ratio }
        if ($dominant -eq "bandwidth_sensitive") { $b = $ratio }
        if ($dominant -eq "reliability_sensitive") { $r = $ratio }
        if ($dominant -eq "compute_sensitive") { $c = $ratio }

        $jobs += @{
            Dominant = $dominant
            Ratio = $ratio
            ServiceRatios = "{delay_sensitive:$d,bandwidth_sensitive:$b,reliability_sensitive:$r,compute_sensitive:$c}"
        }
        if ([double]$ratio -eq 0.25) {
            $uniformAdded = $true
        }
    }
}

$totalRuns = $algorithms.Count * $jobs.Count
Write-Host "============================================================"
Write-Host "Chapter 4 dominant-service ratio experiment"
Write-Host "Algorithms: $($algorithms.Label -join ', ')"
Write-Host "Runs: $totalRuns"
Write-Host "lambda=$lambda, SFC=$numRequests, lifetime=${lifetimeLow}-${lifetimeHigh}ms"
Write-Host "snapshot=${snapshotDurationMs}ms, snapshots=$numSnapshots, seed=$seed"
Write-Host "MIP time limit=${mipTimeLimitSeconds}s"
Write-Host "DryRun=$DryRun"
Write-Host "============================================================"

$runIndex = 0
foreach ($alg in $algorithms) {
    foreach ($job in $jobs) {
        $runIndex += 1
        Write-Host ""
        Write-Host "============================================================"
        Write-Host "[$runIndex/$totalRuns] Ch4 ratio: algorithm=$($alg.Label), dominant=$($job.Dominant), ratio=$($job.Ratio)"
        Write-Host "service_ratios=$($job.ServiceRatios)"
        Write-Host "============================================================"

        $argsList = @(
            "main.py",
            "--config-name", "main_ch4",
            "hydra.run.dir=$projectRoot/results",
            "experiment.seed=$seed",
            "experiment.run_id=auto",
            "experiment.save_root_dir=virne_ch4/",
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
            "v_sim_setting.service_qos.service_ratios=$($job.ServiceRatios)",
            "solver.objective.name=resource_latency",
            "solver.objective.resource_weight=1.0",
            "solver.objective.latency_weight=0.1",
            "solver.fix_endpoint_vnfs=true",
            "solver.include_endpoint_latency=true"
        )

        if ($alg.PretrainedModel) {
            $argsList += "solver.pretrained_model_path=$($alg.PretrainedModel)"
        }
        if ($alg.IsMip) {
            $argsList += "++solver.mip_time_limit_seconds=$mipTimeLimitSeconds"
        }

        if ($DryRun) {
            Write-Host "$pythonExe $($argsList -join ' ')"
            continue
        }

        & $pythonExe @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "Ch4 dominant-ratio run failed: algorithm=$($alg.Label), dominant=$($job.Dominant), ratio=$($job.Ratio), exit_code=$LASTEXITCODE"
        }
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Finished: Chapter 4 dominant-service ratio experiments."
Write-Host "============================================================"
