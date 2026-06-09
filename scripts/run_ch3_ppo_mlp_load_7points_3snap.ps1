# Chapter 3 PPO-MLP+ official load test script
# 7 个到达率点：0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021
# Official setting: 10 snapshots, fixed_total_time_ms=1000000 ms, 100 seconds per snapshot
# 用法：powershell -ExecutionPolicy Bypass -File "D:\any download\virne-main\virne-main\scripts\run_ch3_ppo_mlp_load_7points_3snap.ps1"

$repo = "D:\any download\virne-main\virne-main"
$python = "D:\anaconda\envs\virne-main\python.exe"
$model = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_mlp+\DESKTOP-2NM5LP6-20260526T232004-4022\models\model.pkl"
$rates = @("0.003", "0.006", "0.009", "0.012", "0.015", "0.018", "0.021")

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repo "overnight_logs\ch3_ppo_mlp_load_7points_10snap_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Set-Location -LiteralPath $repo

Write-Host "============================================================"
Write-Host "Chapter 3 PPO-MLP+ official load test started"
Write-Host "Python: $python"
Write-Host "模型路径: $model"
Write-Host "日志目录: $logDir"
Write-Host "到达率: $($rates -join ', ')"
Write-Host "Setting: fixed_total_time_ms=1000000, num_snapshots=10"
Write-Host "============================================================"

foreach ($rate in $rates) {
    $start = Get-Date
    Write-Host ""
    Write-Host "[$start] 开始测试 lambda=$rate"

    $argsList = @(
        "main.py",
        "--config-name", "main_ch3",
        "solver.solver_name=ppo_mlp+",
        "solver.pretrained_model_path=$model",
        "training.num_train_epochs=0",
        "v_sim_setting.arrival_rate.lam=$rate",
        "v_sim_setting.fixed_total_time_ms=1000000",
        "p_net_setting.topology.num_snapshots=10"
    )

    & $python @argsList
    $exitCode = $LASTEXITCODE
    $end = Get-Date

    if ($exitCode -ne 0) {
        Write-Host "[$end] lambda=$rate 运行失败，退出码=$exitCode" -ForegroundColor Red
        "[$end] lambda=$rate FAILED, exit_code=$exitCode" | Out-File -FilePath (Join-Path $logDir "summary.txt") -Append -Encoding UTF8
    } else {
        Write-Host "[$end] lambda=$rate 运行完成" -ForegroundColor Green
        "[$end] lambda=$rate OK" | Out-File -FilePath (Join-Path $logDir "summary.txt") -Append -Encoding UTF8
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "All PPO-MLP+ official load tests finished"
Write-Host "日志目录: $logDir"
Write-Host "============================================================"
