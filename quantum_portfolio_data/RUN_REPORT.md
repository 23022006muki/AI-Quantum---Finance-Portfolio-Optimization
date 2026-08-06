# Run report

## Trạng thái triển khai

Nhánh `fix/research-validity` đã bổ sung:

- historical-universe/data-provenance contract và research fail-closed;
- observation time/availability time semantics;
- purged walk-forward với embargo và fold audit;
- feature coverage theo fold, validation tuning, XGBoost–EWMA Rank IC comparison;
- adaptive universe reduction có M trong biên qubit và diagnostics;
- QUBO–Ising mapping;
- COBYLA multi-start cho penalty-QAOA và Dicke/XY-QAOA;
- primary most-probable solution tách khỏi best-observed solution;
- buy-and-hold drift, gross/net returns, common costs và cost ledger;
- benchmark 1/N, Markowitz, minimum variance, liquidity, EWMA, XGBoost, exact, SA,
  penalty-QAOA và XY-QAOA;
- centered paired block bootstrap, Holm correction và sensitivity chạy lại thật;
- audit script, tài liệu phương pháp và 36 automated tests.

## Demo đã kiểm toán

Demo fixture gần nhất tại thời điểm sửa chạy 4/4 folds với 8 mã và 12.528 bản ghi.
Audit artifact trả về `pass`. Demo chỉ xác nhận luồng phần mềm và mang nhãn
**NOT RESEARCH RESULT**; mọi con số lợi nhuận/solver từ fixture không được dùng trong báo cáo
thực nghiệm HOSE.

Lệnh tái lập:

```powershell
python -m pytest -q
python -m compileall -q src app.py scripts
python -m src.cli run-full --config configs/quick.yaml
python scripts/audit_research_run.py outputs/experiments/<demo-id>
```

## Research gate

Panel giá hiện có 467.164 bản ghi, 300 mã, giai đoạn thực tế 2020-01-02 đến
2025-12-31 và vượt data-quality gate. Tuy nhiên research run bị chặn trước huấn luyện vì:

- security master được suy từ phiên giá đầu tiên (`first_price_observation_proxy`), không
  phải lịch sử niêm yết chính thức;
- universe snapshot/membership history chưa có provenance lịch sử đạt hợp đồng;
- corporate actions point-in-time và adjustment policy chưa được xác minh;
- các bảng phụ còn mang nguồn fixture từ demo.

Artifact blocker gần nhất được audit ở trạng thái `blocked_valid`. Hệ thống không tạo
`metrics_long.csv`, backtest hoặc kết luận H1–H6 cho run bị chặn.

```powershell
python -m src.cli run-experiment --config configs/hose300_real.yaml
python scripts/audit_research_run.py outputs/experiments/<blocked-id> --allow-blocked
```

## Ranh giới tuyên bố

Không có quantum advantage claim. XY-QAOA và penalty-QAOA là ideal statevector
simulations. Exact solver chỉ là oracle cho instance nhỏ. Chỉ một research run có leakage
audit hợp lệ, provenance đầy đủ và audit script trả `pass` mới được dùng cập nhật kết quả
nghiên cứu.
