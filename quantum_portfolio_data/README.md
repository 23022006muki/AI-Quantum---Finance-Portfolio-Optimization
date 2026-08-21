# AI–Quantum Portfolio Research Pipeline

Hệ thống point-in-time cho chuỗi:

`data → XGBoost/EWMA → adaptive universe reduction → cardinality QUBO →`
`XY-QAOA/Dicke → classical weights → walk-forward backtest → report`.

## Google Colab standalone

Notebook [`colab/AI_Quantum_Standalone_Complete_System.ipynb`](colab/AI_Quantum_Standalone_Complete_System.ipynb)
chứa toàn bộ mã nguồn của pipeline trong các cell, không `git clone` và không tải mã nguồn
từ GitHub/Google Drive. Người dùng chỉ cần chọn cấu hình `SMOKE` hoặc `FULL`, chạy lần lượt
các cell và tải lên một tệp dữ liệu CSV/ZIP. Notebook tự kiểm tra SHA-256, dựng môi trường
Python biệt lập, chạy kiểm thử, walk-forward backtest, các bộ giải đối chứng, H1–H6, bảng,
hình và đóng gói toàn bộ artifact để tải về.

[Mở notebook standalone bằng Google Colab](https://colab.research.google.com/github/23022006muki/AI-Quantum---Finance-Portfolio-Optimization/blob/main/quantum_portfolio_data/colab/AI_Quantum_Standalone_Complete_System.ipynb)

## Trạng thái dữ liệu

Workspace có panel giá thị trường 444/445 mã HOSE giao cắt giai đoạn 2020–2025 và
security master niêm yết/hủy niêm yết chính thức. Mã VPK không có phiên HOSE quan sát được
trong khoảng 01–13/01/2020 trên các nguồn công khai đã thử. Hệ thống vẫn thiếu hợp đồng
điều chỉnh giá và benchmark total-return đủ provenance. Sau đối soát CafeF và correction
có ledger/backup, data-quality hiện pass với 0 outlier lợi nhuận điều chỉnh chưa giải quyết.
Vì vậy research mode hiện **fail-closed**: hệ thống tạo artifact `blocked` rồi dừng trước
huấn luyện/backtest. Adapter `fixture` chỉ kiểm thử phần mềm và mọi artifact đều ghi
**NOT RESEARCH RESULT**. Hệ thống không suy ngày niêm yết từ phiên giá đầu tiên, không
đoán endpoint và không vượt paywall/CAPTCHA/robots.txt.

## Cài đặt và chạy trên Windows PowerShell

```powershell
cd "D:\NCKH 2026 - Thầy Dã\quantum_portfolio_data"
python -m pip install -e .
python -m src.cli crawl --stage 1 --source fixture --from 2020-01-01 --to 2025-12-31
python -m src.cli validate --stage 1
python -m src.cli build-universe --rebalance monthly
python -m src.cli leakage-audit
python -m src.cli run-experiment --config configs/quick.yaml
python -m streamlit run app.py
```

Cài đặt tái lập đúng toàn bộ dependency bắc cầu và hash:

```powershell
python -m pip install --require-hashes -r requirements.lock
```

Chạy một lệnh duy nhất để tạo dữ liệu fixture 30 mã, train/test 12 folds, chạy toàn bộ
solver/ablation/sensitivity/statistics và in báo cáo chi tiết ngay trong terminal:

```powershell
python -m src.cli run-full --config configs/full_demo.yaml
```

Lệnh demo chạy trong một workspace tạm biệt lập và chỉ sao chép artifact thí nghiệm về
`outputs/experiments`; nó không ghi đè `outputs/normalized`, `outputs/curated` hay các bảng thật.

Chạy một lệnh trên panel thật đã import (lệnh sẽ fail-closed và xuất blocker artifact nếu
hợp đồng point-in-time chưa đủ):

```powershell
python -m src.cli run-full --config configs/hose300_real.yaml
```

Nếu cần chạy ngay trên phần dữ liệu thật hiện đã đủ trường, dùng chế độ complete-case
khám phá. Lệnh này tạo workspace riêng, giữ các bản ghi đủ OHLCV/provenance và chỉ giữ
mã có tối thiểu 40 quan sát trong giai đoạn 2020–2025; panel chuẩn không bị thay đổi:

```powershell
python -m src.cli run-complete-case --config configs/hose300_complete_case_exploratory.yaml `
  --from 2020-01-01 --to 2025-12-31 --minimum-total-observations 40 `
  --maximum-calendar-gap-days 5
```

Kết quả được gắn nhãn **EXPLORATORY ONLY**. Việc lọc theo độ phủ toàn kỳ có thể gây
coverage/survivorship selection bias và không thay thế hợp đồng điều chỉnh corporate
actions; vì vậy kết quả này không được trình bày như kiểm định confirmatory toàn HOSE.

Chạy **Data B**, phiên bản tối ưu hóa trên panel Data A bất biến, với 44 fold
liên tục, blend XGBoost–technical chọn bằng validation, AUR có kiểm tra khả năng
mua lô, best-observed feasible XY-QAOA và market-regime exposure:

```powershell
python -m src.cli run-data-b
```

Run chuẩn `20260815T184821-a800b2584d` đạt lợi nhuận ngoài mẫu sau chi phí
16,13%, nhưng chỉ H3 được hỗ trợ thống kê và kết quả vẫn mang tính
khám phá. Xem `outputs/Data B/DATA_B_SYSTEM_AND_RESULTS_VI.md` để đọc phương
pháp, kết quả, rổ cuối và các hạn chế bắt buộc.

Tạo và thẩm định một panel **CafeF-only** hoàn toàn riêng; hệ thống chỉ train/backtest
khi còn tối thiểu 8 mã sau quality gate:

```powershell
python -m src.cli run-cafef --config configs/cafef_standalone_exploratory.yaml `
  --from 2020-01-01 --to 2025-12-31 `
  --tickers VCB,BID,CTG,MBB,HPG,FPT,VNM,VIC,GAS,MSN,MWG,SSI `
  --max-workers 3 --minimum-total-observations 40 --maximum-calendar-gap-days 5
```

CafeF được xem là nguồn tổng hợp tham chiếu, không phải feed chính thức của HOSE. Raw
response, checksum, lỗi theo mã, coverage và acceptance gate được lưu trong workspace
riêng. Panel chuẩn chỉ được đọc để lấy security master chính thức và không bị ghi đè.

Crawl toàn bộ security master, checkpoint có thể resume và đặt tên thư mục `data CafeF`:

```powershell
python -m src.cli run-cafef --config configs/cafef_all_exploratory.yaml `
  --from 2020-01-01 --to 2025-12-31 --tickers auto `
  --workspace-name "data CafeF" --max-workers 4 `
  --minimum-total-observations 40 --maximum-calendar-gap-days 5
```

Import CSV thật được cấp quyền:

```powershell
python -m src.cli crawl --stage 1 --source csv --input C:\path\prices.csv `
  --source-name "authorized_dataset" --source-url "https://documented-source"
```

Nguồn chính thức SSI FastConnect:

```powershell
$env:SSI_CONSUMER_ID="..."
$env:SSI_CONSUMER_SECRET="..."
python -m src.cli crawl --stage 1 --source ssi --from 2015-01-01 --to 2025-12-31 `
  --tickers VNM,FPT,HPG,SSI
```

Lấy security master và lịch sử hủy niêm yết trực tiếp từ dịch vụ công khai chính thức của HOSE:

```powershell
python -m src.cli crawl-hose-security-master --from-year 2015 --to-year 2025
```

Lệnh này gắn ISIN ổn định vào panel giá, cách ly các dòng nằm ngoài khoảng niêm yết HOSE và
lưu bản cũ có thể phục hồi. Nó không thay thế nguồn OHLC, corporate actions hay benchmark.

Hoàn thiện checkpoint giá cho toàn bộ security master lịch sử rồi merge an toàn:

```powershell
python -m src.cli crawl-historical-price-gaps --from 2020-01-01 --to 2025-12-31
python -m src.cli merge-historical-price-checkpoints --from 2020-01-01 --to 2025-12-31
```

Collector dùng FDR/Yahoo trước, vnstock/KBS và CafeF làm fallback; checkpoint không tự
promote. Chỉ lệnh merge mới kiểm tra OHLC, khoảng niêm yết và stable security ID trước khi
thay panel, đồng thời lưu bản cũ có thể phục hồi.

Lấy snapshot vĩ mô Việt Nam từ World Bank API v2 không cần API key và kiểm toán nguồn:

```powershell
python -m src.cli crawl-world-bank --from-year 2015 --to-year 2025
python -m src.cli audit-data-sources
```

World Bank snapshot được lưu riêng dưới tên `macro_world_bank_snapshot.parquet` với
`pit_eligible=false`; nó không được dùng trong walk-forward backtest vì API không cung cấp
release vintage cho từng quan sát lịch sử.

Đối chiếu OHLC bằng API chính thức của Trading Economics:

```powershell
$env:TRADING_ECONOMICS_API_KEY="..."
python -m src.cli crawl --stage 1 --source tradingeconomics `
  --from 2020-01-01 --to 2025-12-31 --tickers VCB,FPT,HPG,VNM
```

Kết quả Trading Economics chỉ nằm trong staging và `outputs/reports/trading_economics_crosscheck.csv`;
nó không được promote thành panel giá chính vì endpoint không có volume và không chứng nhận điều chỉnh giá.

Import các bảng point-in-time Stage 1–3:

```powershell
python -m src.cli import-pit-table --table index_membership --input data\vn30_history.csv
python -m src.cli import-pit-table --table security_master --input data\hose_listing_history.csv
python -m src.cli import-pit-table --table corporate_actions --input data\actions.csv
python -m src.cli import-pit-table --table financial_statements --input data\financials.csv
python -m src.cli import-pit-table --table macro --input data\macro_release_calendar.csv
python -m src.cli import-pit-table --table foreign_flow --input data\foreign_flow.csv
python -m src.cli import-pit-table --table benchmark --input data\vnindex_total_return.csv
python -m src.cli apply-adjustment-contract --input data\price_adjustment_contract.json
```

Các mẫu hợp đồng nằm trong `docs/contracts/`. Hợp đồng điều chỉnh giá bị ràng buộc bằng
SHA-256 với đúng file `prices.parquet`; hệ thống từ chối áp dụng nếu dữ liệu đã thay đổi.

Mọi bảng bị từ chối nếu thiếu `available_at`, `source_url` và các timestamp hiệu lực/
công bố theo data contract. `fetched_at` và `raw_checksum` được ghi khi import.

## Kiến trúc

- `outputs/raw`: response/fixture gốc và immutable manifest.
- `outputs/normalized`: prices, security master và corporate actions dạng Parquet.
- `outputs/quarantine`: fixture phụ trợ cũ được chuyển có thể phục hồi khi chuyển sang dữ liệu thật.
- `outputs/curated`: historical universe/features.
- `outputs/reports`: coverage, quality và leakage audit.
- `outputs/experiments/<id>`: config, hash, folds, rankings, instances, solver logs,
  weights, trades, returns, metrics, figures và report.

## Quantum implementation

XY-QAOA dùng ideal statevector trong không gian fixed-Hamming-weight. Trạng thái đầu là
Dicke state; all-to-all XY exchange mixer chỉ nối các bitstring khả thi. Góc biến phân
được tối ưu bằng COBYLA đa khởi tạo và lưu đầy đủ trace. Nghiệm chính là bitstring khả thi
có xác suất cao nhất; best-observed được báo riêng. Đây là simulator nội bộ, không phải
phần cứng lượng tử và không phải bằng chứng quantum advantage.

## Recovery và reproducibility

Raw fixture/CSV import là idempotent theo checksum. Mỗi experiment có config hash,
dataset hash, seeds, environment và artifact index. Có thể đọc UI từ artifact mà không
chạy lại solver. Full config không được chạy như nghiên cứu cho đến khi leakage audit
trên dữ liệu thật pass.

Xem thêm [data governance](docs/DATA_GOVERNANCE_AND_PIT.md),
[methodology traceability](docs/METHODOLOGY_TRACEABILITY.md),
[quantum implementation](docs/QUANTUM_IMPLEMENTATION.md),
[backtest and statistics](docs/BACKTEST_AND_STATISTICS.md) và
[research limitations](docs/RESEARCH_LIMITATIONS.md).

## Risk, constraints and trading costs

- The QUBO expected-return vector is calibrated from XGBoost ranks using only purged
  validation data; multivariate EWMA estimates covariance and remains a baseline mean.
- The quantum layer enforces fixed cardinality; the reference configuration
  selects exactly 4 assets from an 8-asset candidate universe.
- Classical allocation enforces full investment, no short selling and the
  configured per-asset lower and upper bounds.
- Every strategy uses buy-and-hold weight drift between monthly rebalances.
- Turnover is calculated over the union of old/new holdings, including full exits.
- The same transaction-cost policy applies to every strategy, with commission, sell tax,
  slippage and market impact exported separately.
- Portfolio constraints support long-only/full investment, bounds, sector caps, turnover
  limits and ADV capacity. Missing returns and verified delistings have an explicit audit log.

## Audit

```powershell
python scripts/audit_research_run.py outputs/experiments/<experiment-id>
python scripts/audit_research_run.py outputs/experiments/<blocked-id> --allow-blocked
```

Lệnh thứ hai xác nhận một blocked run là trung thực và không chứa metrics; nó không biến
blocked run thành research result.
