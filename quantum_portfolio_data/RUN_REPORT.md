# Run report

## Final implemented scope

The executable system now includes:

- immutable raw manifests, normalized/curated layers and Parquet contracts;
- deterministic Stage 1–3 fixtures with explicit `data_class=fixture`;
- official SSI FastConnect Stage 1 adapter with retry/backoff and fail-closed credentials;
- user-authorized CSV adapter and point-in-time importers for historical index membership,
  corporate actions, financial statements, macro releases and foreign flow;
- official FRED adapter with release-aware timestamps;
- point-in-time universe reconstruction and leakage/data-quality/coverage audits;
- technical, fundamental and macro features joined only after `available_at`;
- walk-forward folds, train-only imputer/scaler and XGBoost/EWMA baselines;
- correlation-aware adaptive universe reduction;
- QUBO, exact solver, simulated annealing and stochastic penalty baseline;
- full-space ideal statevector penalty-QAOA circuit simulation;
- fixed-Hamming-weight Dicke/XY-QAOA ideal statevector simulation;
- classical constrained weight optimization and transaction costs;
- eight ablation configurations;
- sensitivity over cardinality, depth, shots, noise proxy and transaction cost;
- block-bootstrap paired comparisons with Holm correction;
- trailing-information market-regime metrics;
- Streamlit artifact viewer and reproducible Markdown/HTML reports.

## Latest verified execution

Experiment:

`outputs/experiments/20260730T002029-1c4e58b47e`

Commands:

```powershell
python -m pytest -q
python -m compileall -q src app.py
python -m src.cli crawl --stage 1 --source fixture --from 2020-01-01 --to 2025-12-31
python -m src.cli validate --stage 1
python -m src.cli build-universe --rebalance monthly
python -m src.cli leakage-audit
python -m src.cli run-experiment --config configs/quick.yaml
python -m streamlit run app.py --server.headless=true --server.port=8511
```

Verified results:

- Tests: 10 passed.
- Compilation: passed.
- Records: 46,980; tickers: 30; period: 2020-01-01 through 2025-12-31.
- Universe rows: 2,160.
- Fixture data quality: pass.
- Fixture leakage contracts: 5/5 true, `pass_for_fixture_demo`.
- Walk-forward folds: 12/12.
- Ablation configurations: 8, producing 96 fold-level ablation rows.
- Sensitivity cases: 48.
- Solver runs: 108.
- Penalty-QAOA mean feasibility: approximately 0.781 in the 30-ticker comprehensive run.
- XY-QAOA/Dicke feasibility: 1.0 by fixed-weight subspace construction.
- Paired block-bootstrap/Holm results: no significant outperformance in fixture mode.
- UI: `http://localhost:8511`.
- Full research config preflight: correctly refused fixture data.

## Scientific status

All fixture artifacts are labeled **NOT RESEARCH RESULT**. The software implementation is
complete and tested, but a genuine HOSE 2015–2025 research execution cannot be fabricated.
The official SSI adapter requires credentials that are not present:

- `SSI_CONSUMER_ID`
- `SSI_CONSUMER_SECRET`

Historical listing/delisting dates, VN30 effective membership, corporate-action history and
publication-timestamp financial statements must also come from an authorized, reliable
source. The pipeline rejects incomplete point-in-time tables.

To execute official data after credentials are supplied:

```powershell
$env:SSI_CONSUMER_ID="..."
$env:SSI_CONSUMER_SECRET="..."
python -m src.cli crawl --stage 1 --source ssi --from 2015-01-01 --to 2025-12-31 `
  --tickers VNM,FPT,HPG,SSI
```
