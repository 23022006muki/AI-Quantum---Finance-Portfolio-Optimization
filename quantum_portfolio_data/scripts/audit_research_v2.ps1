$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
python scripts/audit_research_v2.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m src.cli audit-data-sources
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/generate_research_v2_reports.py
