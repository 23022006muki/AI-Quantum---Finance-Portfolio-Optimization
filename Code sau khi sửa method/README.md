# Hệ thống sau khi sửa phương pháp

Hệ thống này cô lập thí nghiệm mới khỏi implementation cũ. Đối tượng so sánh là
**Adaptive Universe Reduction (AUR)** và **Quantum-Assisted Universe Reduction
(QAUR)**. Hai phương pháp nhận cùng universe, tín hiệu XGBoost/EWMA và ma trận
rủi ro. Sau Top-K, cả hai gọi cùng một pipeline chọn danh mục và phân bổ tỷ trọng.

## Data flow

`shared forecasts -> {AUR, QAUR} -> Top-K -> shared cardinality QUBO -> shared XY-QAOA -> shared classical weights -> walk-forward backtest`

QAUR hiện dùng QUBO với cardinality-preserving local-search surrogate trên máy
cổ điển. Đây là backend kiểm chứng formulation, không phải bằng chứng quantum
advantage. XY-QAOA downstream là statevector feasible-subspace dùng chung.

## Chạy

```powershell
python run.py
python -m pytest tests -q
```

Đầu vào mặc định dùng forecast và fold manifest đã được sinh bởi experiment
`20260813T164535-21c9b569ce`, cùng dữ liệu feature point-in-time hiện có.
Kết quả được ghi vào `outputs/latest/`.

