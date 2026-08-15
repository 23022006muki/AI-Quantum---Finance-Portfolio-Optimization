# REPORT SYSTEM TRACEABILITY

| Trường | Giá trị |
|---|---|
| Experiment ID | `BLOCKED_DATA_GATES` |
| Dataset hash (candidate) | `dd24d243f3e00d6516962326474ff243cdc200e9667dd1b4a7dd5cd3ed664b82` |
| Adjustment version | `corporate-actions-v2` |
| Config hash | `not_created_no_confirmatory_run` |
| Git commit tại thời điểm báo cáo | `c01b8eb8a1aae5023ea68248a4fa79675429edfc` |
| Created at | `2026-08-15T06:04:32.683165+00:00` |
| Mode | `research_v2_fail_closed` |
| Label | `BLOCKED — DIAGNOSTIC ARTIFACT, NOT CONFIRMATORY RESEARCH` |
| Folds | `0 (không huấn luyện/backtest khi gate chưa đạt)` |
| OOS start/end | `not_created` |
| Audit status | `blocked_valid` |

## Artifact map

| Nội dung | Artifact nguồn | Trạng thái |
|---|---|---|
| Baseline Data A | `outputs/research_v2/audits/BASELINE_AUDIT.md` | frozen/pass |
| Raw corporate actions | `outputs/research_v2/raw/corporate_actions/` | collected |
| Ledger chuẩn hóa | `outputs/research_v2/normalized/corporate_actions.parquet` | diagnostic |
| Corporate-action source audit | `outputs/research_v2/reports/corporate_action_source_audit.json` | unresolved |
| Adjustment contract | `docs/contracts/price_adjustment_contract.v2.json` | versioned |
| Total-return candidate | `outputs/research_v2/normalized/prices_total_return.parquet` | blocked candidate |
| Adjustment audit | `outputs/research_v2/reports/price_adjustment_audit.json` | blocked |
| Return-only comparison | `outputs/research_v2/reports/adjustment_return_only_summary.csv` | diagnostic complete |
| Historical security master | `outputs/research_v2/normalized/security_master_pit.parquet` | partial |
| Monthly PIT universe | `outputs/research_v2/curated/universe_monthly_pit.parquet` | partial |
| Survivorship audit | `outputs/research_v2/reports/survivorship_bias_audit.json` | partial_blocked |
| Verified total-return benchmark | `outputs/normalized/benchmark.parquet` | missing/blocked |
| Research V2 experiment | `outputs/research_v2/experiments/BLOCKED_DATA_GATES/manifest.json` | intentionally not run |
| Overall audit | `outputs/research_v2/audits/research_v2_audit.json` | blocked_valid |

## Reproducibility commands

```powershell
.\scripts\run_research_v2.ps1
.\scripts\audit_research_v2.ps1
.\scripts\reproduce_research_v2.ps1
```
