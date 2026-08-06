# Ma trận đối chiếu báo cáo – hệ thống

Ngày kiểm toán: 2026-08-06  
Nhánh: `fix/research-validity`

| Nội dung nghiên cứu | Trạng thái triển khai | Bằng chứng |
|---|---|---|
| Universe point-in-time | Hoàn chỉnh trong mã; dữ liệu thật còn bị chặn | `universe_contract.json`, `universe_monthly.parquet`, `leakage_audit.json` |
| Phân biệt toàn HOSE và thành phần chỉ số | Hoàn chỉnh | `universe.definition: hose_all_listed/index_membership` |
| Purging, embargo và walk-forward phủ toàn kỳ | Hoàn chỉnh | `fold_manifest.csv`; `max_folds: null` cho cấu hình HOSE |
| XGBoost tạo lợi nhuận kỳ vọng cho QUBO | Hoàn chỉnh | `signal_calibration.csv`, `optimization_instances.json` |
| EWMA đa biến cho hiệp phương sai | Hoàn chỉnh | `covariance.method: ewma`, `ewma_expected_return_reference` |
| Adaptive Universe Reduction và đối chứng Top-M | Hoàn chỉnh | `aur_diagnostics.csv`; kiểm định H2 trong `statistical_tests.csv` |
| QUBO–Ising, exact, SA, Penalty-QAOA, XY-QAOA | Hoàn chỉnh | `optimization_instances.json`, `solver_runs.csv` |
| Dicke/XY bảo toàn cardinality và noise stress test | Hoàn chỉnh ở simulator | ideal statevector; kênh depolarizing/readout chỉ là mô phỏng hiện tượng học |
| Tỷ trọng và ràng buộc thực tế | Hoàn chỉnh trong mã | long-only, full investment, bounds, sector cap, turnover, ADV capacity |
| Missing return, halt và delisting | Hoàn chỉnh, fail-closed | `missing_return_resolution.csv`; biến mất không giải thích sẽ chặn research |
| Phí, thuế, slippage và market impact | Hoàn chỉnh | `trades.csv`, `cost_ledger.csv` với từng thành phần |
| Benchmark thị trường total-return | Hợp đồng và cổng kiểm tra hoàn chỉnh; thiếu dữ liệu thật | `benchmark.parquet` bắt buộc khi cấu hình yêu cầu |
| Risk-free | Hoàn chỉnh | hỗ trợ `fixed_annual` và `pit_macro_series`; xuất `risk_free_series.csv` |
| H1–H6, bootstrap và Holm | Hoàn chỉnh trong mã | `statistical_tests.csv`, `sensitivity_results.csv` |
| Artifact integrity | Hoàn chỉnh | SHA-256 cho từng artifact; audit phát hiện chỉnh sửa |
| Dependency reproducibility | Hoàn chỉnh | `requirements.lock` chứa dependency bắc cầu và hash |
| Research-mode integration test dương tính | Hoàn chỉnh | 42 tests, gồm run thành công với hợp đồng dữ liệu tổng hợp hợp lệ |

## Các blocker dữ liệu thật còn lại

Đây không còn là thiếu sót logic có thể sửa bằng cách tự suy diễn dữ liệu. Research run thật
hiện bị chặn trước huấn luyện/backtest bởi năm điều kiện:

1. 47 outlier lợi nhuận điều chỉnh chưa được nối với corporate action hoặc hợp đồng điều chỉnh đã xác minh.
2. Security master vẫn dùng `first_price_observation_proxy`, không phải lịch sử niêm yết/hủy niêm yết chính thức.
3. Snapshot universe kế thừa nguồn proxy nên chưa đạt cổng tin cậy.
4. Chính sách điều chỉnh giá chưa có provenance được xác minh.
5. Chưa có chuỗi VN-Index total-return đáp ứng hợp đồng point-in-time.

Artifact mới nhất `20260806T181627-c5ef044e1b-blocked` được audit là `blocked_valid`; không
có metrics nghiên cứu hoặc tuyên bố quantum advantage. Các bảng fixture phụ trợ cũ đã được
chuyển có thể phục hồi sang `outputs/quarantine/fixture_auxiliary/20260806T180922`.
