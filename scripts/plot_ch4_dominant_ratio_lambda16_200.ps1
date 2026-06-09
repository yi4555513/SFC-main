param(
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = "D:\any download\virne-main\virne-main"
Set-Location -LiteralPath $projectRoot

$pythonExe = "D:\anaconda\envs\virne-main\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = "D:\毕业+就业\毕业大论文\最新图\ch4_dominant_lambda16_200_$stamp"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $pythonExe .\scripts\plot_ch4_dominant_ratio_figures.py `
    --service all `
    --figure-set all `
    --algorithms mip,aco_meta,ppo_mlp+,ppo_gat_seq2seq+ `
    --arrival-rate 0.016 `
    --num-v-nets 200 `
    --snapshot-duration-ms 60000 `
    --num-snapshots 20 `
    --ratios 0.25,0.40,0.55,0.70 `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "Chapter 4 dominant-ratio plotting failed, exit_code=$LASTEXITCODE"
}

Write-Host ""
Write-Host "Figures saved to: $OutputDir"
