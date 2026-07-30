$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$project = Split-Path -Parent $PSScriptRoot
$progressPath = Join-Path $project "outputs\raw\vnstock_progress.json"
$manifestPath = Join-Path $project "outputs\raw\vnstock_manifest.json"
$logPath = Join-Path $project "outputs\vnstock_supervisor.log"
Set-Location -LiteralPath $project

for ($cycle = 1; $cycle -le 30; $cycle++) {
    Add-Content -LiteralPath $logPath -Value "cycle=$cycle started=$(Get-Date -Format o)"
    & python -u -m src.cli crawl --stage 1 --source vnstock `
        --from 2020-01-01 --to 2025-12-31 --tickers auto --max-tickers 300 `
        *>> $logPath

    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        if ($manifest.tickers_collected -ge 300) {
            Add-Content -LiteralPath $logPath -Value "completed=300 finished=$(Get-Date -Format o)"
            exit 0
        }
    }

    $completed = 0
    if (Test-Path -LiteralPath $progressPath) {
        $progress = Get-Content -Raw -LiteralPath $progressPath | ConvertFrom-Json
        $completed = $progress.completed.Count
    }
    Add-Content -LiteralPath $logPath -Value "cycle=$cycle checkpointed=$completed; waiting=65s"
    Start-Sleep -Seconds 65
}

Add-Content -LiteralPath $logPath -Value "supervisor_exhausted_cycles=$(Get-Date -Format o)"
exit 2
