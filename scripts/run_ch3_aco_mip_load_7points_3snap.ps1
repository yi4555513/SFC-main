# 第三章 ACO-Meta + MIP 不同到达率/负载自动测试脚本
# 7 个到达率点：0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021
# ACO-Meta official setting: 10 snapshots, fixed_total_time_ms=1000000 ms, 100 seconds per snapshot
# MIP keeps the 3-snapshot quick test and is not included in official load plots.
# 用法：powershell -ExecutionPolicy Bypass -File "D:\any download\virne-main\virne-main\scripts\run_ch3_aco_mip_load_7points_3snap.ps1"

$repo = "D:\any download\virne-main\virne-main"
$python = "D:\anaconda\envs\virne-main\python.exe"
$rates = @("0.003", "0.006", "0.009", "0.012", "0.015", "0.018", "0.021")
$solvers = @("aco_meta", "mip")

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repo "overnight_logs\ch3_aco_mip_load_7points_mixedsnap_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Set-Location -LiteralPath $repo

Write-Host "============================================================"
Write-Host "第三章 ACO-Meta + MIP 负载测试开始"
Write-Host "Python: $python"
Write-Host "日志目录: $logDir"
Write-Host "算法顺序: $($solvers -join ' -> ')"
Write-Host "到达率: $($rates -join ', ')"
Write-Host "Setting: ACO-Meta fixed_total_time_ms=1000000, num_snapshots=10; MIP keeps fixed_total_time_ms=180000, num_snapshots=3"
Write-Host "============================================================"

foreach ($solver in $solvers) {
    Write-Host ""
    Write-Host "================ 开始算法: $solver ================" -ForegroundColor Cyan

    foreach ($rate in $rates) {
        $start = Get-Date
        Write-Host ""
        Write-Host "[$start] 开始测试 solver=$solver, lambda=$rate"

        if ($solver -eq "aco_meta") {
            $totalTimeMs = "1000000"
            $numSnapshots = "10"
        } else {
            # MIP keeps the small quick test to avoid excessive running time
            $totalTimeMs = "180000"
            $numSnapshots = "3"
        }

        $argsList = @(
            "main.py",
            "--config-name", "main_ch3",
            "solver.solver_name=$solver",
            "training.num_train_epochs=0",
            "v_sim_setting.arrival_rate.lam=$rate",
            "v_sim_setting.fixed_total_time_ms=$totalTimeMs",
            "p_net_setting.topology.num_snapshots=$numSnapshots"
        )

        & $python @argsList
        $exitCode = $LASTEXITCODE
        $end = Get-Date

        if ($exitCode -ne 0) {
            Write-Host "[$end] solver=$solver, lambda=$rate 运行失败，退出码=$exitCode" -ForegroundColor Red
            "[$end] solver=$solver, lambda=$rate FAILED, exit_code=$exitCode" | Out-File -FilePath (Join-Path $logDir "summary.txt") -Append -Encoding UTF8
        } else {
            Write-Host "[$end] solver=$solver, lambda=$rate 运行完成" -ForegroundColor Green
            "[$end] solver=$solver, lambda=$rate OK" | Out-File -FilePath (Join-Path $logDir "summary.txt") -Append -Encoding UTF8
        }
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "全部 ACO-Meta + MIP 负载测试已结束"
Write-Host "日志目录: $logDir"
Write-Host "============================================================"
