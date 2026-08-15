$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
& (Join-Path $PSScriptRoot "run_research_v2.ps1")
exit $LASTEXITCODE
