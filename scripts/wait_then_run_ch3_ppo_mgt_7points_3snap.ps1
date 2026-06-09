# 等 PPO-MLP+ 高负载测试脚本结束后，自动继续运行 PPO-MGT 高负载测试脚本
# 用法：另开一个 PyCharm Terminal，运行：
# powershell -ExecutionPolicy Bypass -File "D:\any download\virne-main\virne-main\scripts\wait_then_run_ch3_ppo_mgt_7points_3snap.ps1"

$repo = "D:\any download\virne-main\virne-main"
$mgtScript = "D:\any download\virne-main\virne-main\scripts\run_ch3_ppo_mgt_load_7points_3snap.ps1"
$mlpScriptName = "run_ch3_ppo_mlp_load_7points_3snap.ps1"

Set-Location -LiteralPath $repo

Write-Host "============================================================"
Write-Host "等待 PPO-MLP+ 测试结束，然后自动启动 PPO-MGT 测试"
Write-Host "检测中的 PPO-MLP+ 脚本: $mlpScriptName"
Write-Host "之后启动的 PPO-MGT 脚本: $mgtScript"
Write-Host "============================================================"

while ($true) {
    $running = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*$mlpScriptName*" -and
            $_.CommandLine -notlike "*wait_then_run_ch3_ppo_mgt_7points_3snap.ps1*"
        }

    if (-not $running) {
        break
    }

    $now = Get-Date
    Write-Host "[$now] PPO-MLP+ 还在运行，60 秒后再次检查..."
    Start-Sleep -Seconds 60
}

$start = Get-Date
Write-Host ""
Write-Host "[$start] 检测到 PPO-MLP+ 已结束，开始运行 PPO-MGT..." -ForegroundColor Green

powershell -ExecutionPolicy Bypass -File $mgtScript
$exitCode = $LASTEXITCODE

$end = Get-Date
if ($exitCode -ne 0) {
    Write-Host "[$end] PPO-MGT 运行失败，退出码=$exitCode" -ForegroundColor Red
} else {
    Write-Host "[$end] PPO-MGT 全部运行完成" -ForegroundColor Green
}
