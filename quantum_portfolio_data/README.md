# AI–Quantum Portfolio Research Pipeline

Hệ thống point-in-time cho chuỗi:

`data → XGBoost/EWMA → adaptive universe reduction → cardinality QUBO →`
`XY-QAOA/Dicke → classical weights → walk-forward backtest → report`.

## Trạng thái dữ liệu

Workspace chưa có nguồn HOSE 2015–2025 được cấp quyền và kiểm chứng. Adapter `fixture`
chỉ dùng kiểm thử phần mềm và mọi artifact đều ghi **NOT RESEARCH RESULT**. Adapter
`csv` nhận file thật do người dùng cấp, kèm tên và URL/dataset nguồn; hệ thống không
đoán endpoint, không vượt paywall/CAPTCHA/robots.txt.

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

Chạy một lệnh duy nhất để tạo dữ liệu fixture 30 mã, train/test 12 folds, chạy toàn bộ
solver/ablation/sensitivity/statistics và in báo cáo chi tiết ngay trong terminal:

```powershell
python -m src.cli run-full --config configs/full_demo.yaml
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

Import các bảng point-in-time Stage 1–3:

```powershell
python -m src.cli import-pit-table --table index_membership --input data\vn30_history.csv
python -m src.cli import-pit-table --table corporate_actions --input data\actions.csv
python -m src.cli import-pit-table --table financial_statements --input data\financials.csv
python -m src.cli import-pit-table --table macro --input data\macro_release_calendar.csv
python -m src.cli import-pit-table --table foreign_flow --input data\foreign_flow.csv
```

Mọi bảng bị từ chối nếu thiếu `available_at` và các timestamp hiệu lực/công bố theo
data contract.

## Kiến trúc

- `outputs/raw`: response/fixture gốc và immutable manifest.
- `outputs/normalized`: prices, security master và corporate actions dạng Parquet.
- `outputs/curated`: historical universe/features.
- `outputs/reports`: coverage, quality và leakage audit.
- `outputs/experiments/<id>`: config, hash, folds, rankings, instances, solver logs,
  weights, trades, returns, metrics, figures và report.

## Quantum implementation

XY-QAOA dùng ideal statevector trong không gian fixed-Hamming-weight. Trạng thái đầu là
Dicke state; all-to-all XY exchange mixer chỉ nối các bitstring khả thi. Đây là simulator
nội bộ tái lập được, không phải phần cứng lượng tử. Penalty-QAOA dùng full-Hilbert-space
statevector, `|+>` initialization, cost Hamiltonian có cardinality penalty và transverse-X
mixer. Baseline stochastic cũ vẫn được giữ và ghi nhãn rõ để đối chiếu.

## Recovery và reproducibility

Raw fixture/CSV import là idempotent theo checksum. Mỗi experiment có config hash,
dataset hash, seeds, environment và artifact index. Có thể đọc UI từ artifact mà không
chạy lại solver. Full config không được chạy như nghiên cứu cho đến khi leakage audit
trên dữ liệu thật pass.

Xem thêm [data dictionary](docs/DATA_DICTIONARY.md) và
[source registry](docs/SOURCE_REGISTRY.md).

## Risk, constraints and trading costs

- The default estimator is a multivariate EWMA covariance matrix calculated only
  with observations available before each rebalance.
- The quantum layer enforces fixed cardinality; the reference configuration
  selects exactly 4 assets from an 8-asset candidate universe.
- Classical allocation enforces full investment, no short selling and the
  configured per-asset lower and upper bounds.
- Each strategy carries its drifted holdings into the next fold. Turnover is
  calculated over the union of old and new holdings as the sum of absolute
  pre-trade-to-target weight changes.
- Transaction costs are charged on actual turnover, including full exits.
