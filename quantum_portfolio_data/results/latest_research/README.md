# Latest research status: blocked

Research execution is intentionally blocked before training and backtesting. The available
real price panel contains 467,164 observations for 300 tickers from 2020-01-02 through
2025-12-31, but prices alone do not establish a valid historical investable universe,
adjustment contract, or total-return benchmark. The quality gate also identified 47
adjusted-return outliers that remain unresolved by a verified contract or corporate action.

No portfolio metrics, hypothesis conclusions, or quantum-advantage claims are published in
this directory. The blocker package is committed so GitHub users can distinguish an honest
incomplete research run from the separately available fixture demo.

Resolve every item in `blocker_manifest.json`, rebuild the point-in-time universe, rerun
`configs/hose300_real.yaml`, and require `scripts/audit_research_run.py` to return `pass`
before replacing this package with research results.

Latest blocked artifact: `20260806T181627-c5ef044e1b-blocked` (`blocked_valid`). The
software validation suite passes 42 tests, including a successful synthetic real-contract
research-mode integration run and artifact tamper detection.
