# Methodology traceability

| Thành phần báo cáo | Triển khai | Artifact kiểm toán |
|---|---|---|
| Point-in-time universe | `build_universe`, `leakage_audit`, snapshot theo fold | `universe_monthly.parquet`, `leakage_audit.json`, `data_provenance.json` |
| Walk-forward purged/embargo | `make_folds`, `purged_fold_frames` | `fold_manifest.csv` |
| Feature/target | `build_features`, `attach_point_in_time_features` | `features.parquet`, `feature_coverage_by_fold.csv` |
| XGBoost tuning | `fit_ranker`, train-only preprocessing, validation Rank IC | `model_tuning.csv`, `rankings.csv` |
| EWMA đối chứng | EWMA đa biến cho mean/covariance; EWMA đơn chuỗi theo từng mã cho ranking baseline | `rankings.csv`, `statistical_tests.csv` |
| Adaptive universe reduction | signal, liquidity, risk, correlation và qubit budget; M trong biên cấu hình | `selected_universe.csv`, `aur_diagnostics.csv` |
| QUBO/Ising | `qubo_instance`, `qubo_to_ising` với quy ước `x=(1-z)/2` | `optimization_instances.json` |
| Solver | exact, SA, penalty-QAOA và Dicke/XY-QAOA | `solver_runs.csv`, `comparisons.csv` |
| Tỷ trọng | long-only, tổng tỷ trọng bằng một, lower/upper bound và turnover penalty | `weights.csv` |
| Backtest | monthly walk-forward, buy-and-hold giữa các kỳ, cùng cost policy | `portfolio_returns.csv`, `trades.csv`, `cost_ledger.csv` |
| Benchmark | universe 1/N, candidate 1/N, Markowitz, minimum variance, liquidity, EWMA, XGB, exact, SA, penalty, XY | `metrics_long.csv`, `ablation_results.csv` |
| H1-H6 | Rank IC, AUR diagnostics, feasibility, gap, net performance, sensitivity | `statistical_tests.csv`, `sensitivity_results.csv`, `RESEARCH_REPORT.md` |

Exact solver chỉ là oracle khi số biến đủ nhỏ để liệt kê. `full_pipeline_xy_qaoa` lấy nghiệm chính từ XY-QAOA; không lấy exact solution làm danh mục đề xuất.

