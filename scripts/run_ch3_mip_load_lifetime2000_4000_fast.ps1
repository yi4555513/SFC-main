$ErrorActionPreference = "Stop"

# 绗笁绔?MIP 涓嶅悓鍒拌揪鐜囪礋杞藉疄楠岋紙蹇€熺増锛?# 鍙ｅ緞锛?# - 66 涓崼鏄熻妭鐐?# - 10 涓揩鐓э紝100 绉?蹇収
# - SFC 鐢熷懡鍛ㄦ湡 2000~4000 ms
# - 鍥哄畾绔偣銆佽绠楃鐐规椂寤?# - MIP 鐩爣鍑芥暟娌跨敤鏃ф晥鏋滆緝濂界殑 resource_latency锛宭atency_weight=0.1
# - 姣忎釜鍒拌揪鐜囧彧璺?300 鏉?SFC锛岃妭鐪佹椂闂达紱濡傞渶鏇寸ǔ鍙妸 $numVnets 鏀规垚 500 鎴?1000

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

$oldConfigDir = "results/virne_ch3/mip/DESKTOP-2NM5LP6-20260526T134031-9051"

$seed = 0
$numVnets = 300
$mipTimeLimitSeconds = 5
$snapshotDurationMs = 100000
$numSnapshots = 10

# Keep the same arrival rates as the other Ch3 load-comparison experiments.
$loadRates = @(0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021)

foreach ($lam in $loadRates) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running Ch3 MIP load fast: lambda=$lam, num_v_nets=$numVnets, lifetime=2000~4000, time_limit=${mipTimeLimitSeconds}s"
    Write-Host "============================================================"

    & $pythonExe main.py `
        --config-path $oldConfigDir `
        --config-name config `
        hydra.run.dir=$projectRoot/results `
        experiment.seed=$seed `
        experiment.run_id=auto `
        experiment.save_root_dir=virne_ch3/ `
        v_sim_setting.num_v_nets=$numVnets `
        v_sim_setting.arrival_rate.lam=$lam `
        v_sim_setting.lifetime.low=2000 `
        v_sim_setting.lifetime.high=4000 `
        ++v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
        ++v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        p_net_setting.topology.num_nodes=66 `
        p_net_setting.topology.planes=6 `
        p_net_setting.topology.nums_per_plane=11 `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        ++p_net_setting.topology.cycle_snapshots=true `
        solver.objective.name=resource_latency `
        solver.objective.resource_weight=1.0 `
        solver.objective.latency_weight=0.1 `
        +solver.mip_time_limit_seconds=$mipTimeLimitSeconds `
        solver.fix_endpoint_vnfs=true `
        solver.include_endpoint_latency=true

    if ($LASTEXITCODE -ne 0) {
        throw "Ch3 MIP load fast failed: lambda=$lam, exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Ch3 MIP load fast experiments finished."
