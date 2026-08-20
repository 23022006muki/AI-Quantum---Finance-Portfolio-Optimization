# AI–Quantum Finance Portfolio Optimization

Hệ thống nghiên cứu tối ưu hóa danh mục cổ phiếu HOSE theo kiến trúc lai giữa học máy,
tối ưu hóa cổ điển và điện toán lượng tử biến phân.

> **Lưu ý:** đây là hệ thống nghiên cứu, không phải khuyến nghị đầu tư và hiện không có
> tuyên bố về quantum advantage.

## Phương pháp

```text
Dữ liệu point-in-time
        ↓
XGBoost ranking + EWMA covariance
        ↓
Adaptive Universe Reduction
        ↓
Cardinality-constrained QUBO
        ↓
Dicke-state feasible-subspace XY-QAOA
        ↓
Tối ưu tỷ trọng cổ điển có ràng buộc
        ↓
Walk-forward backtest sau chi phí
```

Khung nghiên cứu lựa chọn một tập ứng viên phù hợp với ngân sách qubit, dùng XY-QAOA để
chọn đúng số lượng cổ phiếu và xác định tỷ trọng bằng bộ tối ưu cổ điển. Các phương pháp
đối chứng gồm EWMA, Equal Weight, Markowitz, Minimum Variance, Exact Solver, Simulated
Annealing và Penalty-QAOA.

## Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Automated tests | **80/80 passed** |
| Data 17/8 panel | 157.826 dòng, 120 mã complete-case, 2020–2025 |
| Benchmark | VNAllShare TRI, 1.499/1.499 phiên |
| Walk-forward experiment | **33/33 folds completed** |
| Data-quality / leakage audit | `pass` / `pass_with_limitations` |
| Research classification | **Exploratory, không phải confirmatory full-HOSE** |

Run chuẩn `20260820T160429-4f2cfc123d` đạt lợi nhuận tích lũy sau chi phí 2,25%,
nhưng không vượt VNAllShare TRI và chưa có bằng chứng thống kê cho hiệu quả tài chính
vượt trội. XY-QAOA bảo toàn cardinality trên ideal simulator; kết quả không chứng minh
quantum advantage. Dữ liệu tài chính point-in-time và ngành lịch sử vẫn là giới hạn chính.

## Chạy trên Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/23022006muki/AI-Quantum---Finance-Portfolio-Optimization/blob/main/quantum_portfolio_data/colab/AI_Quantum_Portfolio_End_to_End.ipynb)

Notebook tự clone commit dữ liệu đã khóa, kiểm tra SHA-256, cài toàn bộ dependency, chạy
test suite và mặc định tái chạy đủ pipeline 33 folds. Nó xuất mô tả hệ thống, audit dữ liệu,
kết quả tín hiệu/AUR/solver, hiệu quả danh mục, rổ cuối và kết luận H1–H6. Đặt
`RUN_FULL_PIPELINE=False` nếu chỉ cần đọc toàn bộ artifact chuẩn mà không train lại.

## Cài đặt

Yêu cầu Python 3.11 trở lên.

```powershell
git clone https://github.com/23022006muki/AI-Quantum---Finance-Portfolio-Optimization.git
cd AI-Quantum---Finance-Portfolio-Optimization\quantum_portfolio_data
python -m pip install --require-hashes -r requirements.lock
```

## Chạy toàn bộ demo

```powershell
python -m src.cli run-full --config configs/full_demo.yaml
```

Demo sử dụng dữ liệu fixture xác định trước và luôn mang nhãn `NOT RESEARCH RESULT`.

## Chạy pipeline trên dữ liệu đã nhập

```powershell
python -m src.cli validate
python -m src.cli build-universe --definition hose_all_listed
python -m src.cli leakage-audit
python -m src.cli run-full --config configs/hose300_real.yaml
```

Nếu data contract chưa đạt, lệnh cuối tạo một blocked artifact có thể kiểm toán và dừng
trước khi huấn luyện mô hình.

## Kiểm thử và kiểm toán

```powershell
python -m pytest -q
python -m compileall -q src app.py scripts tests
python scripts/audit_research_run.py outputs\experiments\<experiment-id>
```

Mỗi successful run lưu config, dataset hash, universe hash, optimizer trace, weights,
trades, gross/net returns, thống kê H1–H6 và SHA-256 của từng artifact.

## Giao diện kết quả

```powershell
python -m streamlit run app.py
```

Dashboard hiển thị chất lượng dữ liệu, leakage audit, XGBoost calibration, Adaptive
Universe Reduction, solver comparison, danh mục, chi phí, constraint diagnostics,
backtest, ablation và sensitivity analysis.

## Cấu trúc repository

```text
quantum_portfolio_data/
├── colab/                   # Notebook chạy end-to-end trên Google Colab
├── colab_bundle/            # Runtime data + published experiment có SHA-256
├── configs/                 # Cấu hình demo và research
├── docs/                    # Phương pháp, governance và data contracts
├── results/latest_research/ # Trạng thái research công khai, không chứa dữ liệu thô
├── scripts/                 # Audit và lệnh hỗ trợ
├── src/                     # Data pipeline, source adapters và research engine
├── tests/                   # Kiểm thử phần mềm và research validity
└── app.py                   # Streamlit dashboard
```

## Tài liệu

- [Hướng dẫn chi tiết](quantum_portfolio_data/README.md)
- [Báo cáo trạng thái chạy](quantum_portfolio_data/RUN_REPORT.md)
- [Ma trận đối chiếu báo cáo–hệ thống](quantum_portfolio_data/docs/REPORT_SYSTEM_GAP_MATRIX.md)
- [Data governance và point-in-time](quantum_portfolio_data/docs/DATA_GOVERNANCE_AND_PIT.md)
- [Phương pháp backtest và thống kê](quantum_portfolio_data/docs/BACKTEST_AND_STATISTICS.md)
- [Mẫu data contract](quantum_portfolio_data/docs/contracts/)

## Giấy phép và dữ liệu

Repository không lưu cookies hoặc API secrets. Gói Colab chỉ chứa panel complete-case và
artifact cần để tái lập; kho raw disclosure/PDF cục bộ 54,6 GB không được sao chép lên
GitHub. Người sử dụng chịu trách nhiệm tuân thủ điều khoản của nhà cung cấp và không được
diễn giải run exploratory thành kết quả confirmatory toàn HOSE.
