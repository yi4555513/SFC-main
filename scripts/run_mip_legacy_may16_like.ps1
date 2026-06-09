$ErrorActionPreference = "Stop"

# 复现 2026-05-16 旧 MIP 配置口径：
# - 读取当时保存的 config.yaml
# - 重新生成 run_id，避免覆盖旧结果
# - 关闭当前新增的端点固定/端点时延口径，更接近旧代码效果
# - MIP 单条求解时间默认 10 秒

$PythonExe = "D:\anaconda\envs\virne-main\python.exe"
$OldConfigDir = "results/virne/mip/DESKTOP-2NM5LP6-20260516T163657-8124"

& $PythonExe main.py `
  --config-path $OldConfigDir `
  --config-name config `
  experiment.run_id=auto `
  experiment.save_root_dir=virne_legacy_mip/ `
  +solver.mip_time_limit_seconds=10 `
  +solver.fix_endpoint_vnfs=false `
  +solver.include_endpoint_latency=false
