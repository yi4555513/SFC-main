$ErrorActionPreference = "Stop"
Set-Location "D:\any download\virne-main\virne-main"
$pythonExe = "D:\anaconda\envs\virne-main\python.exe"

# Ch3 basic 1000-SFC mid-load test using trained PPO models.
# lambda=0.004 means 1000 requests arrive in about 250 seconds on average.
# snapshot_duration_ms=100000 keeps the original 100 seconds per topology snapshot.
# cycle_snapshots=true is safe if a test exceeds 10 snapshots; otherwise it has no effect.
$lambda = 0.004
$numRequests = 1000
$snapshotDurationMs = 100000
$numSnapshots = 10

$ppoMlpModel = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_mlp+\DESKTOP-2NM5LP6-20260526T232004-4022\models\model.pkl"
$ppoMgtModel = "D:\any download\virne-main\virne-main\results\virne_ch3\ppo_gat_seq2seq+\DESKTOP-2NM5LP6-20260526T165819-1245\models\model.pkl"

$models = @(
    @{ solver = "ppo_mlp+"; model = $ppoMlpModel },
    @{ solver = "ppo_gat_seq2seq+"; model = $ppoMgtModel }
)

foreach ($item in $models) {
    $solver = $item.solver
    $model = $item.model
    Write-Host "============================================================"
    Write-Host "Running Ch3 basic 1000 mid-load test: solver=$solver, lambda=$lambda"
    Write-Host "============================================================"

    & $pythonExe main.py --config-name main_ch3 `
        experiment.seed=0 `
        solver.solver_name=$solver `
        solver.pretrained_model_path=$model `
        training.num_train_epochs=0 `
        v_sim_setting.num_v_nets=$numRequests `
        v_sim_setting.arrival_rate.lam=$lambda `
        v_sim_setting.auto_num_v_nets_from_arrival_rate=false `
        v_sim_setting.snapshot_duration_ms=$snapshotDurationMs `
        p_net_setting.topology.num_snapshots=$numSnapshots `
        p_net_setting.topology.cycle_snapshots=true

    if ($LASTEXITCODE -ne 0) {
        throw "Basic 1000 mid-load test failed: solver=$solver, exit_code=$LASTEXITCODE"
    }
}

Write-Host "============================================================"
Write-Host "Ch3 PPO basic 1000 mid-load tests finished."
Write-Host "If you also need ACO/MIP under the same setting, run them separately because MIP is slow."
Write-Host "============================================================"
