param(
    [switch]$SkipCrawl
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

if (-not $SkipCrawl) {
    python -m src.cli crawl-corporate-actions --from 2020-01-01 --to 2025-12-31 --tickers auto --max-workers 4 --pause-seconds 0.12
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python -m src.cli build-universe-pit-v2 --from 2020-01-01 --to 2025-12-31 --lookback-days 90 --minimum-observations 40
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m src.cli build-price-adjustment-v2
$adjustmentExit = $LASTEXITCODE
python -m src.cli adjustment-counterfactual
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/audit_research_v2.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m src.cli audit-data-sources
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/generate_research_v2_reports.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($adjustmentExit -ne 0) {
    Write-Host "Research V2 stopped safely: the price-adjustment gate is blocked."
    exit 2
}

python -m src.cli run-research-v2 --config configs/hose_research_v2.yaml
exit $LASTEXITCODE
