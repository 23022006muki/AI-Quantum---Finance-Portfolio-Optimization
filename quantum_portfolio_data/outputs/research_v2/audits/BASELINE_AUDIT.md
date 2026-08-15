# Baseline audit — Data A

This document freezes the pre-remediation exploratory baseline. It is an audit reference, not a confirmatory research result. The source experiment remains unchanged at `outputs/Data A/outputs/experiments/20260813T164535-21c9b569ce`.

## Identity and reproducibility

| Field | Baseline value |
|---|---:|
| Experiment ID | `20260813T164535-21c9b569ce` |
| Status | `success` |
| Mode | `exploratory` |
| Label | `EXPLORATORY ONLY - COMPLETE-CASE REAL HOSE PANEL; NOT CONFIRMATORY RESEARCH` |
| Dataset SHA-256 | `6e046b509fef366681866328d5bd99ec63541c2de8597f0e7bebc101813baa05` |
| Config hash | `21c9b569ce` |
| Recorded Git commit | `19d9677ab49ebb3065d759ff12d0fe9555bb6dc2` |
| Packaged baseline commit | `c01b8eb8a1aae5023ea68248a4fa79675429edfc` |
| Rows | 565,471 |
| Tickers | 394 |
| Price interval | 2020-01-02 to 2025-12-31 |
| OOS interval | 2022-05-02 to 2025-12-30 |
| Completed folds | 12/12 |

## Quantitative baseline

| Metric | Value | Authoritative artifact |
|---|---:|---|
| Mean XGBoost walk-forward Rank IC | 0.079541 | `rankings.csv` |
| XY-QAOA feasibility rate | 100.00% | `solver_runs.csv` |
| XY-QAOA mean primary-solution optimality gap | 8.9266% | `solver_runs.csv` |
| XY-QAOA mean best-observed gap | 0.0000% | `solver_runs.csv` |
| Full-pipeline cumulative net return | -20.6929% | `strategy_metrics_summary.csv` |
| Full-pipeline annualized net return | -20.0563% | `strategy_metrics_summary.csv` |
| Full-pipeline Sharpe ratio | -0.669293 | `strategy_metrics_summary.csv` |
| Full-pipeline maximum drawdown | -34.9978% | `strategy_metrics_summary.csv` |
| Mean one-way turnover per rebalance | 1.850840 | `cost_ledger.csv` |
| Mean transaction cost per rebalance | 0.400971% | `cost_ledger.csv` |
| Sum of fold-level transaction-cost fractions | 4.811649% | `cost_ledger.csv` |
| Last selected basket | SMB, DXG, GAS, STB | `latest_selected_portfolio.csv` |

The feasibility rate is an ideal-statevector property of the feasible-subspace XY mixer and Dicke initialization. It is not evidence of quantum speedup or quantum advantage. Likewise, the zero best-observed gap must not be substituted for the non-zero primary-solution gap.

## Hypotheses

| Hypothesis | Baseline status |
|---|---|
| H1 | `not_statistically_supported` |
| H2 | `not_statistically_supported` |
| H3 | `statistically_supported_on_declared_tests` |
| H4 | `not_statistically_supported` |
| H5 | `not_statistically_supported` |
| H6 | `sensitivity_completed` |

## Baseline validity boundary

The baseline leakage audit passed only after explicitly accepting two exploratory limitations: the price-adjustment policy was unverified and no hash-bound adjustment contract matched the dataset. The corporate-actions table was empty. Therefore Data A is preserved for causal comparison but cannot be promoted to confirmatory research. All research-v2 results must use new dataset/config hashes and must fail closed if material adjustment events or the required total-return benchmark remain unresolved.
