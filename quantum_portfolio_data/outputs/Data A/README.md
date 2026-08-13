# Data A

`Data A` là tên ổn định của bộ dữ liệu complete-case HOSE được tạo từ panel giá thật
sau yêu cầu chỉ giữ phần dữ liệu còn đủ điều kiện sử dụng.

## Phạm vi

- Giai đoạn: 02/01/2020–31/12/2025.
- Số mã giữ lại: 394.
- Số bản ghi: 565.471.
- Universe động dùng tối đa 300 mã tại mỗi kỳ tái cân bằng.
- Mỗi mã phải có tối thiểu 40 quan sát hoàn chỉnh.
- Không chấp nhận gap đầu kỳ, cuối kỳ hoặc nội kỳ quá 5 phiên giao dịch.
- 51 mã không đạt điều kiện đã được loại và ghi trong
  `outputs/reports/complete_case_exclusions.csv`.

## Kết quả gắn với Data A

Experiment chính thức của gói này là `20260813T164535-21c9b569ce`, nằm tại
`outputs/experiments/20260813T164535-21c9b569ce`.

Experiment đã chạy đủ 12/12 walk-forward folds, gồm XGBoost/EWMA, adaptive universe
reduction, QUBO, exact solver, simulated annealing, penalty-QAOA, feasible-subspace
XY-QAOA, tối ưu tỷ trọng, backtest sau chi phí, ablation, sensitivity và kiểm định
block bootstrap/Holm.

## Nhãn diễn giải

Data A chỉ được sử dụng cho thực nghiệm khám phá. Bộ dữ liệu chưa có hợp đồng xác minh
điều chỉnh corporate actions, benchmark total-return và metadata ngành đầy đủ; do đó
không được trình bày như kiểm định confirmatory toàn HOSE hoặc bằng chứng quantum
advantage.
