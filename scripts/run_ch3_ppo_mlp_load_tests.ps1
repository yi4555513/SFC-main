# 第三章 PPO-MLP+ 不同到达率/负载自动测试脚本
# 用法：powershell -ExecutionPolicy Bypass -File "D:\any download\virne-main\virne-main\scripts\run_ch3_ppo_mlp_load_tests.ps1"

$repo = "D:\any download\virne-main\virne-main"
$python = "D:\anaconda\envs\virne-main\python.exe"
$model = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_mlp+\DESKTOP-2NM5LP6-20260526T232004-4022\models\model.pkl"
$rates = @("0.001", "0.002", "0.003", "0.004", "0.005", "0.006", "0.007", "0.008", "0.009", "0.010")

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repo "overnight_logs\ch3_ppo_mlp_load_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Set-Location -LiteralPath $repo

Write-Host "============================================================"
Write-Host "第三章 PPO-MLP+ 负载测试开始"
Write-Host "Python: $python"
Write-Host "模型路径: $model"
Write-Host "日志目录: $logDir"
Write-Host "到达率: $($rates -join ', ')"
Write-Host "============================================================"

foreach ($rate in $rates) {
    $start = Get-Date
    $logFile = Join-Path $logDir "ppo_mlp_load_lambda_$rate.log"
    Write-Host ""
    Write-Host "[$start] 开始测试 lambda=$rate"
    Write-Host "进度条将原地刷新；只保存 summary.txt。"

    $argsList = @(
        "main.py",
        "--config-name", "main_ch3",
        "solver.solver_name=ppo_mlp+",
        "solver.pretrained_model_path=$model",
        "training.num_train_epochs=0",
        "v_sim_setting.arrival_rate.lam=$rate"
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
Write-Host "全部 PPO-MLP+ 负载测试已结束"
Write-Host "日志目录: $logDir"
Write-Host "============================================================"



