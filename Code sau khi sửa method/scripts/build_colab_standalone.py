"""Build the hand-written, no-GitHub standalone Colab notebook."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab" / "AUR_QAUR_XYQAOA_Standalone_Full_Colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(text).strip() + "\n"}


cells = [
md(r"""
# AUR vs QAUR Portfolio Research — Complete Executable Colab

Notebook này vừa là **mã nguồn có thể chạy**, vừa là **tài liệu phương pháp luận và báo cáo kết quả**. Toàn bộ code được viết trực tiếp trong từng cell, không `git clone`, không tải source code từ GitHub và không phụ thuộc package nội bộ.

\[
\mathcal U_t \xrightarrow{\text{XGBoost/EWMA}}
\{\text{AUR},\text{QAUR}\}
\xrightarrow{\text{Top-}K}
Q^{PO}\xrightarrow{\text{XY-QAOA}}
k_p\text{ tài sản}\xrightarrow{\text{classical weights}}
\text{walk-forward OOS}.
\]

**Cách chạy:** upload `data_sau_khi_sua_method.zip` (hoặc CSV tương ứng), chọn `EXECUTION_PROFILE`, sau đó bấm **Runtime → Run all**. Mỗi phần kết quả được hiển thị ngay trong notebook; cell cuối tạo `AUR_QAUR_XYQAOA_RESULTS.zip` trong bảng Files.

> **Phạm vi tuyên bố.** QAUR hiện dùng classical cardinality-preserving surrogate cho QUBO reduction; XY-QAOA là ideal statevector simulation trong fixed-Hamming-weight feasible subspace. Notebook không tuyên bố quantum advantage, quantum speedup hay chạy trên QPU thật.
"""),
md(r"""
## 0. Kiến trúc hệ thống, đối tượng so sánh và ký hiệu

Đối tượng được so sánh là **hai phương pháp giảm vũ trụ AUR và QAUR**, không phải AUR với QAOA. Cả hai nhánh bắt đầu từ cùng point-in-time universe \(\mathcal U_t\), nhận cùng dự báo XGBoost \(\hat\mu_{i,t}\), cùng rủi ro/hiệp phương sai EWMA và tạo cùng loại đầu ra là candidate set có đúng \(K\) tài sản. Sau đó, cả hai candidate set đi qua **cùng một downstream implementation**.

| Giai đoạn | Thuật toán | Đầu vào | Đầu ra | Vai trò trong thí nghiệm |
|---|---|---|---|---|
| 1 | XGBoost + EWMA | Dữ liệu chỉ khả dụng đến ngày quyết định | Signal, volatility, covariance | Dùng chung cho cả hai nhánh |
| 2 | AUR | \(\mathcal U_t\), unary score, correlation | \(\mathrm{Top}\text{-}K_A\) | Reducer thứ nhất |
| 3 | QAUR / \(Q^{UR}\) | Cùng đầu vào như AUR | \(\mathrm{Top}\text{-}K_{QA}\) | Reducer thứ hai |
| 4 | Paired comparison | Hai candidate set cùng fold | Jaccard, objective, stability | Đo universe-reduction effect |
| 5 | \(Q^{PO}\) + XY-QAOA + SLSQP | Từng candidate set | \(k_p\) tài sản và tỷ trọng | Downstream dùng chung |
| 6 | Walk-forward backtest | Danh mục của từng fold | Return, risk, turnover, drawdown | Đánh giá ngoài mẫu |

Hai QUBO có mục đích khác nhau:

\[
Q^{UR}:\quad \max_{z\in\{0,1\}^{N_t}}
\left(\sum_i s_{i,t}z_i-\lambda_c\sum_{i<j}|\rho_{ij,t}|z_iz_j\right),
\qquad \sum_i z_i=K,
\]

\[
Q^{PO}:\quad \min_{x\in\{0,1\}^{K}}
x^\top\left(\lambda_p\widetilde\Sigma_t-\operatorname{diag}(\widetilde\mu_t)\right)x,
\qquad \sum_i x_i=k_p.
\]

Trong đó \(z_i\) quyết định tài sản có vào candidate set hay không; \(x_i\) quyết định tài sản có được chọn vào danh mục cuối hay không. Việc tách \(Q^{UR}\) và \(Q^{PO}\) ngăn nhầm lẫn giữa universe reduction và portfolio selection.
"""),
md(r"""
## 1. Cài thư viện và cấu hình thí nghiệm

Đặt `EXECUTION_PROFILE = "SMOKE"` để kiểm tra kỹ thuật nhanh trên bốn fold đầu; đặt bằng `"FULL"` để chạy toàn bộ walk-forward experiment. Seed, kích thước candidate set \(K\), cardinality danh mục \(k_p\), tham số reducer, XY-QAOA, transaction cost và XGBoost đều nằm trong một `CONFIG` duy nhất. Vì hai nhánh đọc cùng object cấu hình nên không có tham số downstream riêng cho AUR hoặc QAUR.

Cell code bên dưới sẽ in toàn bộ cấu hình để mỗi lần chạy có thể được kiểm tra và tái lập.
"""),
code(r"""
!pip -q install "numpy>=2.0" "pandas>=2.2" "scipy>=1.13" "scikit-learn>=1.5" "xgboost>=2.1" "matplotlib>=3.8" "seaborn>=0.13"

from pathlib import Path
from itertools import combinations
import hashlib, json, math, os, time, warnings, zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
sns.set_theme(style="whitegrid")

EXECUTION_PROFILE = "FULL"  # Đổi thành "SMOKE" để chạy thử nhanh 4 folds.

CONFIG = {
    "seed": 42,
    "train_months": 24,
    "validation_months": 3,
    "test_months": 1,
    "embargo_days": 20,
    "target_horizon_days": 20,
    "max_folds_smoke": 4,
    "candidate_size": 8,
    "portfolio_cardinality": 4,
    "covariance_span": 60,
    "minimum_history_days": 126,
    "transaction_cost_bps": 25,
    "risk_aversion_qubo": 0.55,
    "risk_aversion_weights": 1.25,
    "weight_lower": 0.05,
    "weight_upper": 0.40,
    "qaoa_depth": 2,
    "qaoa_shots": 1024,
    "qaoa_parameter_budget": 45,
    "qa_restarts": 10,
    "qa_max_swap_rounds": 80,
    "reduction_weights": {
        "signal": 0.40,
        "liquidity": 0.30,
        "risk": 0.15,
        "stability": 0.15,
        "correlation_penalty": 0.10,
    },
    "xgboost": {
        "n_estimators": 120,
        "max_depth": 3,
        "learning_rate": 0.035,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    },
}

WORKDIR = Path("/content/aur_qaur_standalone")
RESULTS = WORKDIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
np.random.seed(CONFIG["seed"])
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))

experiment_design = pd.DataFrame([
    ("Execution profile", EXECUTION_PROFILE, "SMOKE = kiểm tra kỹ thuật; FULL = thí nghiệm hoàn chỉnh"),
    ("Candidate size K", CONFIG["candidate_size"], "Giống nhau cho AUR và QAUR"),
    ("Portfolio cardinality k_p", CONFIG["portfolio_cardinality"], "Được bảo toàn bởi feasible-subspace XY-QAOA"),
    ("Forecast horizon", CONFIG["target_horizon_days"], "Số phiên của forward-return target"),
    ("Train / validation / test", f'{CONFIG["train_months"]}/{CONFIG["validation_months"]}/{CONFIG["test_months"]} tháng', "Walk-forward monthly step"),
    ("Transaction cost", f'{CONFIG["transaction_cost_bps"]} bps', "Trừ tại ngày đầu test window"),
    ("QAUR backend", "Classical surrogate", "Quantum-ready QUBO; không phải QPU"),
    ("Shared portfolio solver", "Fixed-Hamming-weight XY-QAOA", "Giống hệt cho cả hai reducer"),
], columns=["Thành phần", "Giá trị", "Ý nghĩa"])
display(Markdown("### Thiết kế thí nghiệm đang chạy"))
display(experiment_design)
"""),
md("""
## 2. Upload dữ liệu trực tiếp

Notebook không âm thầm tải dữ liệu từ Internet. Hãy upload đúng một CSV hoặc ZIP chứa duy nhất một CSV. Cell thực hiện bốn việc: ưu tiên file đã có trong `/content`, kiểm tra định dạng, giải nén an toàn và tính SHA-256. Hash này được ghi vào run manifest để chứng minh kết quả gắn với đúng phiên bản dữ liệu.
"""),
code(r"""
from google.colab import files

# Ưu tiên file đã được đưa vào Colab session qua bảng Files. Cách này ổn định
# hơn khi chạy lại toàn bộ notebook vì không mở lại hộp chọn file ở mỗi lần Run all.
preuploaded_candidates = [
    Path("/content/data_sau_khi_sua_method.zip"),
    Path("/content/data_sau_khi_sua_method.csv"),
]
existing_uploads = [path for path in preuploaded_candidates if path.exists()]

if existing_uploads:
    upload_path = existing_uploads[0]
    print("Dùng file đã có trong Colab session:", upload_path)
else:
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Hãy upload đúng một file CSV hoặc ZIP.")

    upload_name = next(iter(uploaded))
    upload_path = WORKDIR / Path(upload_name).name
    upload_path.write_bytes(uploaded[upload_name])

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

if upload_path.suffix.lower() == ".zip":
    extract_dir = WORKDIR / "uploaded_data"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(upload_path) as archive:
        unsafe = [n for n in archive.namelist() if Path(n).is_absolute() or ".." in Path(n).parts]
        if unsafe:
            raise ValueError(f"ZIP chứa đường dẫn không an toàn: {unsafe[:3]}")
        archive.extractall(extract_dir)
    candidates = list(extract_dir.rglob("*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"ZIP phải chứa đúng một CSV; tìm thấy {len(candidates)} file.")
    csv_path = candidates[0]
elif upload_path.suffix.lower() == ".csv":
    csv_path = upload_path
else:
    raise ValueError("Định dạng được hỗ trợ: .csv hoặc .zip")

DATASET_SHA256 = sha256_file(csv_path)
print("CSV:", csv_path)
print("SHA-256:", DATASET_SHA256)
"""),
md(r"""
## 3. Đọc và kiểm tra hợp đồng dữ liệu

Dataset sử dụng long-table contract với bốn nhóm dữ liệu chính: `PRICE`, `SECURITY`, `BENCHMARK` và `CORPORATE_ACTION`. Cell tách từng nhóm, chuẩn hóa kiểu ngày/số, loại bản ghi giá không hợp lệ và kiểm tra uniqueness của khóa `(ticker, date)`. Security master được giữ lại để xây point-in-time universe; benchmark total-return index được chuyển thành daily return.

**Kết quả hiển thị:** số dòng theo record type và bảng chất lượng gồm số quan sát giá, số mã, giai đoạn dữ liệu, duplicate và missing adjusted close. Notebook dừng ngay nếu dữ liệu không đủ ít nhất \(K\) tài sản.
"""),
code(r"""
raw = pd.read_csv(csv_path, low_memory=False)
required = {"record_type", "date", "ticker", "adjusted_close", "volume", "trading_value"}
missing = required - set(raw.columns)
if missing:
    raise ValueError(f"Dataset thiếu cột bắt buộc: {sorted(missing)}")

record_counts = raw["record_type"].value_counts(dropna=False).rename_axis("record_type").reset_index(name="rows")
display(record_counts)

prices = raw.loc[raw["record_type"].eq("PRICE")].copy()
security = raw.loc[raw["record_type"].eq("SECURITY")].copy()
benchmark = raw.loc[raw["record_type"].eq("BENCHMARK")].copy()
corporate_actions = raw.loc[raw["record_type"].eq("CORPORATE_ACTION")].copy()

prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
for col in ["adjusted_close", "close", "volume", "trading_value"]:
    prices[col] = pd.to_numeric(prices[col], errors="coerce")
prices = prices.dropna(subset=["date", "ticker", "adjusted_close"])
prices = prices[prices["adjusted_close"] > 0].sort_values(["ticker", "date"])
prices = prices.drop_duplicates(["ticker", "date"], keep="last")

if not benchmark.empty:
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["total_return_index"] = pd.to_numeric(benchmark["total_return_index"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date", "total_return_index"]).sort_values("date")
    benchmark = benchmark.drop_duplicates("date", keep="last")
    benchmark["benchmark_return"] = benchmark["total_return_index"].pct_change(fill_method=None)

if not security.empty:
    for col in ["listing_date", "delisting_date", "effective_from", "effective_to"]:
        if col in security:
            security[col] = pd.to_datetime(security[col], errors="coerce")

quality = pd.DataFrame({
    "metric": ["price_rows", "tickers", "start", "end", "duplicate_ticker_dates", "missing_adjusted_close"],
    "value": [len(prices), prices["ticker"].nunique(), prices["date"].min(), prices["date"].max(), prices.duplicated(["ticker", "date"]).sum(), prices["adjusted_close"].isna().sum()],
})
display(quality)
if len(prices) == 0 or prices["ticker"].nunique() < CONFIG["candidate_size"]:
    raise ValueError("Dữ liệu không đủ để tạo candidate universe.")
"""),
md(r"""
## Giai đoạn 1 — Dự báo lợi suất và rủi ro

### 4. Feature engineering không dùng dữ liệu tương lai

Với mỗi tài sản, notebook xây momentum, moving-average ratio, RSI, MACD, realized volatility, downside volatility, drawdown và liquidity chỉ từ dữ liệu tại hoặc trước ngày \(t\). Nhãn dự báo là forward return sau \(h\) phiên:

\[
y_{i,t}^{(h)}=\frac{P_{i,t+h}}{P_{i,t}}-1,
\qquad
\widetilde y_{i,t}=\operatorname{rank}_{cs}\!\left(y_{i,t}^{(h)}\right).
\]

`target_available_at` lưu ngày \(t+h\) mà nhãn thực sự quan sát được. Một dòng chỉ được vào tập huấn luyện khi `target_available_at < train_end`; đây là purge rule chống label leakage. Nhãn tuyệt đối không được dùng làm feature tại ngày quyết định.

**Kết quả hiển thị:** coverage của từng feature và quy mô feature panel sau xử lý.
"""),
code(r"""
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def build_features(price_frame: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    parts = []
    for ticker, group in price_frame.groupby("ticker", sort=False):
        g = group.sort_values("date").copy()
        px = g["adjusted_close"].astype(float)
        ret1 = px.pct_change(fill_method=None)
        g["return_1d"] = ret1
        for window in [5, 20, 60, 120]:
            g[f"return_{window}d"] = px.pct_change(window, fill_method=None)
        g["sma_ratio_20"] = px / px.rolling(20).mean() - 1
        g["ema_ratio_20"] = px / px.ewm(span=20, adjust=False).mean() - 1
        g["rsi_14"] = rsi(px, 14) / 100.0
        ema12 = px.ewm(span=12, adjust=False).mean()
        ema26 = px.ewm(span=26, adjust=False).mean()
        g["macd_scaled"] = (ema12 - ema26) / px
        g["volatility_20d"] = ret1.rolling(20).std(ddof=1)
        g["downside_volatility_20d"] = ret1.where(ret1 < 0, 0).rolling(20).std(ddof=1)
        g["drawdown_60d"] = px / px.rolling(60).max() - 1
        liquidity = g["trading_value"].where(g["trading_value"] > 0, g["volume"] * px)
        g["liquidity_20d"] = liquidity.rolling(20).mean()
        g["target_return_20d"] = px.shift(-horizon) / px - 1
        # Ngày target thực sự trở nên quan sát được; dùng cột này để purge nhãn.
        g["target_available_at"] = g["date"].shift(-horizon)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["target_rank"] = out.groupby("date")["target_return_20d"].rank(pct=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)

features = build_features(prices, CONFIG["target_horizon_days"])
FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_60d", "return_120d",
    "sma_ratio_20", "ema_ratio_20", "rsi_14", "macd_scaled",
    "volatility_20d", "downside_volatility_20d", "drawdown_60d",
    "liquidity_20d",
]
coverage = features[FEATURE_COLUMNS].notna().mean().sort_values(ascending=False).rename("coverage")
display(coverage.to_frame())
print("Feature rows:", len(features), "| dates:", features["date"].nunique())
"""),
md("""
### 5. Thiết kế walk-forward folds

Mỗi fold gồm cửa sổ train 24 tháng, validation 3 tháng và test 1 tháng; các mốc dịch tiến một tháng sau mỗi fold. XGBoost được fit lại và danh mục được tái cân bằng ở từng fold. Validation chỉ dùng để chẩn đoán forecast; mọi performance metric chính được tính trên test window ngoài mẫu.

Trong chế độ `FULL`, notebook chạy toàn bộ fold khả dụng. `SMOKE` chỉ giới hạn bốn fold đầu và không được dùng để kết luận nghiên cứu. Bảng ngay dưới cell code cho biết chính xác train/validation/test boundaries của từng fold.
"""),
code(r"""
def make_folds(dates: pd.Series, train_months: int, validation_months: int, test_months: int) -> list[dict]:
    start = pd.Timestamp(dates.min()).normalize()
    end = pd.Timestamp(dates.max()).normalize()
    train_start = start
    train_end = train_start + pd.DateOffset(months=train_months)
    folds = []
    fold_id = 0
    while True:
        validation_start = train_end
        validation_end = validation_start + pd.DateOffset(months=validation_months)
        test_start = validation_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end + pd.Timedelta(days=1):
            break
        folds.append({
            "fold": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        fold_id += 1
        train_start += pd.DateOffset(months=1)
        train_end += pd.DateOffset(months=1)
    return folds

folds = make_folds(features["date"], CONFIG["train_months"], CONFIG["validation_months"], CONFIG["test_months"])
if EXECUTION_PROFILE.upper() == "SMOKE":
    folds = folds[:CONFIG["max_folds_smoke"]]
if not folds:
    raise ValueError("Khoảng thời gian dữ liệu không đủ để tạo walk-forward fold.")
fold_table = pd.DataFrame(folds)
display(fold_table)
print("Số fold sẽ chạy:", len(folds))
"""),
md(r"""
### 6. XGBoost, EWMA và point-in-time eligibility dùng chung

XGBoost học quan hệ phi tuyến giữa feature vector và cross-sectional target rank. Forecast thô được chuyển về percentile rank để tạo signal có cùng thang đo giữa các fold. Chất lượng dự báo được báo cáo bằng daily Spearman Rank IC và RMSE trên validation window.

Rủi ro được ước lượng bằng EWMA. Với decay \(\alpha=2/(L+1)\), trọng số của quan sát cũ giảm theo cấp số nhân:

\[
w_\tau\propto(1-\alpha)^{T-\tau},\qquad
\widehat\Sigma_t^{EWMA}=\sum_\tau w_\tau(r_\tau-\bar r_w)(r_\tau-\bar r_w)^\top.
\]

Một mã chỉ hợp lệ nếu đã niêm yết, chưa hủy niêm yết tại ngày quyết định và có đủ minimum history. **Cùng model, forecast snapshot, history và covariance estimator được dùng cho AUR và QAUR.**
"""),
code(r"""
def prepare_matrix(frame: pd.DataFrame, medians: pd.Series | None = None):
    X = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = X.median().fillna(0.0)
    return X.fillna(medians).to_numpy(float), medians

def fit_xgboost(train: pd.DataFrame, seed: int):
    usable = train.dropna(subset=["target_rank"]).copy()
    X, medians = prepare_matrix(usable)
    y = usable["target_rank"].to_numpy(float)
    params = dict(CONFIG["xgboost"])
    model = XGBRegressor(random_state=seed, **params)
    model.fit(X, y)
    return model, medians

def predict_signal(model, medians, snapshot: pd.DataFrame) -> np.ndarray:
    X, _ = prepare_matrix(snapshot, medians)
    prediction = model.predict(X)
    return pd.Series(prediction).rank(pct=True).to_numpy(float)

def validation_daily_rank_ic(model, medians, validation: pd.DataFrame) -> float:
    if validation.empty:
        return np.nan
    X, _ = prepare_matrix(validation, medians)
    scored = validation[["date", "target_rank"]].copy()
    scored["prediction"] = model.predict(X)
    daily = scored.groupby("date").apply(
        lambda group: stats.spearmanr(group["prediction"], group["target_rank"], nan_policy="omit").statistic,
        include_groups=False,
    )
    return float(daily.replace([np.inf, -np.inf], np.nan).mean())

def ewma_covariance(history: pd.DataFrame, tickers: list[str], span: int) -> np.ndarray:
    panel = history.pivot(index="date", columns="ticker", values="return_1d").reindex(columns=tickers)
    panel = panel.tail(max(span * 3, 60)).fillna(0.0)
    if len(panel) < 20:
        return np.eye(len(tickers)) * 1e-4
    decay = 2.0 / (span + 1.0)
    weights = (1.0 - decay) ** np.arange(len(panel) - 1, -1, -1)
    weights = weights / weights.sum()
    values = panel.to_numpy(float)
    mean = weights @ values
    centered = values - mean
    cov = (centered * weights[:, None]).T @ centered
    cov = (cov + cov.T) / 2
    cov += np.eye(len(tickers)) * 1e-8
    return cov

def point_in_time_eligible(ticker: str, decision_time: pd.Timestamp) -> bool:
    if security.empty or "ticker" not in security:
        return True
    rows = security[security["ticker"].eq(ticker)]
    if rows.empty:
        return True
    row = rows.iloc[-1]
    listing = row.get("listing_date", pd.NaT)
    delisting = row.get("delisting_date", pd.NaT)
    if pd.notna(listing) and listing > decision_time:
        return False
    if pd.notna(delisting) and delisting <= decision_time:
        return False
    return True
"""),
md(r"""
## Giai đoạn 2 — Adaptive Universe Reduction (AUR)

Trước tiên, mỗi tài sản nhận một unary score dùng chung:

\[
s_{i,t}=w_sS_{i,t}+w_lL_{i,t}+w_rR_{i,t}+w_hH_{i,t},
\]

trong đó \(S\) là forecast signal, \(L\) là liquidity rank, \(R\) là inverse-volatility rank và \(H\) phản ánh membership stability so với fold trước. AUR bắt đầu với tập rỗng rồi lặp lại:

\[
i^*=\arg\max_{i\notin A}
\left[s_{i,t}-\lambda_c\sum_{j\in A}|\rho_{ij,t}|\right].
\]

Mỗi bước thêm một tài sản có marginal score cao nhất cho đến khi \(|A|=K\). Thuật toán nhanh và dễ giải thích, nhưng quyết định greedy ở bước trước không được quay lại tối ưu toàn cục.
"""),
md(r"""
## Giai đoạn 3 — Quantum-Assisted Universe Reduction (QAUR)

QAUR giải joint selection problem \(Q^{UR}\), trong đó quality của toàn bộ tập và pairwise redundancy được đánh giá đồng thời. Cardinality \(\sum_i z_i=K\) luôn được bảo toàn bằng swap move: loại một tài sản đang chọn và thêm một tài sản bên ngoài. Notebook dùng nhiều điểm khởi tạo rồi lặp best-improving swap đến local optimum.

Backend hiện tại là **classical cardinality-preserving surrogate for a quantum-ready QUBO**. Cách gọi *quantum-assisted* mô tả formulation có thể chuyển sang QAOA/annealing backend; nó không có nghĩa cell này đang chạy QPU.
"""),
md(r"""
## Giai đoạn 4 — So sánh hai phương pháp giảm vũ trụ

AUR và QAUR phải có cùng \(K\), cùng input snapshot và cùng unary score. Candidate-set similarity tại fold \(f\) được đo bằng:

\[
J_f=\frac{|A_f\cap Q_f|}{|A_f\cup Q_f|}.
\]

Ngoài Jaccard, notebook lưu reduction objective, candidate membership, final XY-QAOA membership và selection stability theo từng fold. Các metric danh mục chỉ được diễn giải sau khi đã tách rõ candidate-level effect và downstream portfolio effect.

Cell code dưới đây định nghĩa score dùng chung, AUR và QAUR. Kết quả thực nghiệm được hiển thị sau vòng walk-forward để bảo đảm so sánh theo cặp trên cùng fold.
"""),
code(r"""
def rank01(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = series.rank(method="average", pct=True)
    return ranked if higher_is_better else 1.0 - ranked

def common_reduction_scores(snapshot: pd.DataFrame, previous: set[str]) -> pd.DataFrame:
    weights = CONFIG["reduction_weights"]
    x = snapshot.copy().sort_values("ticker").reset_index(drop=True)
    x["signal_component"] = rank01(x["signal"].fillna(x["signal"].median()))
    x["liquidity_component"] = rank01(x["liquidity_20d"].fillna(0.0))
    x["risk_component"] = rank01(x["volatility_20d"].fillna(np.inf), False)
    x["stability_component"] = x["ticker"].isin(previous).astype(float)
    x["unary_score"] = (
        weights["signal"] * x["signal_component"]
        + weights["liquidity"] * x["liquidity_component"]
        + weights["risk"] * x["risk_component"]
        + weights["stability"] * x["stability_component"]
    )
    return x

def absolute_correlation(history: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    panel = history.pivot(index="date", columns="ticker", values="return_1d").reindex(columns=tickers)
    corr = panel.corr(min_periods=20).fillna(0.0).abs().to_numpy(float)
    np.fill_diagonal(corr, 0.0)
    return corr

def reduction_objective(bits, unary, corr, penalty):
    return float(unary @ bits - penalty * 0.5 * bits @ corr @ bits)

def adaptive_universe_reduction(snapshot, history, previous, k):
    x = common_reduction_scores(snapshot, previous)
    tickers = x["ticker"].tolist()
    corr = absolute_correlation(history, tickers)
    penalty = CONFIG["reduction_weights"]["correlation_penalty"]
    selected, remaining = [], set(range(len(x)))
    while len(selected) < min(k, len(x)):
        best = max(remaining, key=lambda i: (x.loc[i, "unary_score"] - penalty * sum(corr[i, j] for j in selected), tickers[i]))
        selected.append(best)
        remaining.remove(best)
    bits = np.zeros(len(x), dtype=int); bits[selected] = 1
    return {
        "method": "AUR", "tickers": sorted(tickers[i] for i in selected),
        "objective": reduction_objective(bits, x["unary_score"].to_numpy(), corr, penalty),
        "backend": "adaptive_greedy",
    }

def quantum_assisted_universe_reduction(snapshot, history, previous, k, seed):
    x = common_reduction_scores(snapshot, previous)
    tickers = x["ticker"].tolist()
    unary = x["unary_score"].to_numpy(float)
    corr = absolute_correlation(history, tickers)
    penalty = CONFIG["reduction_weights"]["correlation_penalty"]
    rng = np.random.default_rng(seed)
    starts = [np.argsort(unary)[-k:]]
    starts += [rng.choice(len(x), size=k, replace=False) for _ in range(CONFIG["qa_restarts"] - 1)]
    best_bits, best_value = None, -np.inf
    for start in starts:
        bits = np.zeros(len(x), dtype=int); bits[np.asarray(start)] = 1
        value = reduction_objective(bits, unary, corr, penalty)
        for _ in range(CONFIG["qa_max_swap_rounds"]):
            selected = np.flatnonzero(bits); outside = np.flatnonzero(1 - bits)
            move, best_delta = None, 0.0
            for i in selected:
                selected_without_i = selected[selected != i]
                old_pair = corr[i, selected_without_i].sum()
                for j in outside:
                    delta = unary[j] - unary[i] - penalty * (corr[j, selected_without_i].sum() - old_pair)
                    if delta > best_delta + 1e-12:
                        best_delta, move = float(delta), (i, j)
            if move is None:
                break
            bits[move[0]] = 0; bits[move[1]] = 1; value += best_delta
        if value > best_value:
            best_bits, best_value = bits.copy(), value
    return {
        "method": "QAUR", "tickers": sorted(x.loc[best_bits.astype(bool), "ticker"].tolist()),
        "objective": float(best_value),
        "backend": "classical_cardinality_preserving_surrogate_for_quantum_ready_QUBO",
    }
"""),
md(r"""
## Giai đoạn 5 — Portfolio Optimization dùng chung

### 8. Cardinality-Constrained QUBO, XY-QAOA và phân bổ tỷ trọng

Với mỗi candidate set, expected-return vector và EWMA covariance được chuẩn hóa để tạo portfolio-selection QUBO:

\[
E(x)=x^\top Q^{PO}x,\qquad
Q^{PO}=\lambda_p\widetilde\Sigma-\operatorname{diag}(\widetilde\mu),
\qquad \sum_i x_i=k_p.
\]

XY-QAOA downstream là **một implementation duy nhất cho cả AUR và QAUR**. Statevector chỉ biểu diễn các bitstring có đúng \(k_p\) bit bằng 1; trạng thái đầu là Dicke state đều trên feasible basis. Mỗi layer áp dụng cost unitary \(e^{-i\gamma H_C}\) và XY exchange mixer \(e^{-i\beta H_M}\). COBYLA tối ưu \((\gamma,\beta)\), sau đó notebook sample state và chọn nghiệm khả thi tốt nhất đã quan sát.

Sau asset selection, SLSQP giải bài toán tỷ trọng liên tục:

\[
\min_w\ \lambda_w w^\top\Sigma w-\mu^\top w,
\quad \mathbf1^\top w=1,\quad \ell\le w_i\le u.
\]

XY-QAOA quyết định **chọn mã nào**; SLSQP quyết định **phân bổ bao nhiêu**. Feasibility 100% chỉ chứng minh nghiệm luôn thỏa cardinality, không chứng minh quantum advantage hay optimality tuyệt đối.
"""),
code(r"""
def portfolio_qubo(mu: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    mu_scale = max(float(np.max(np.abs(mu))), 1e-9)
    cov_scale = max(float(np.max(np.abs(cov))), 1e-9)
    return risk_aversion * cov / cov_scale - np.diag(mu / mu_scale)

def energy(bits: np.ndarray, q: np.ndarray) -> float:
    return float(bits @ q @ bits)

def feasible_states(n: int, k: int) -> np.ndarray:
    combos = list(combinations(range(n), k))
    states = np.zeros((len(combos), n), dtype=int)
    for row, idx in enumerate(combos):
        states[row, list(idx)] = 1
    return states

def optimize_qaoa_angles(evaluate, depth: int, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    starts = max(1, min(3, budget // 8))
    per_start = max(6, budget // starts)
    bounds = [(0.0, 2 * np.pi)] * depth + [(0.0, np.pi)] * depth
    best = None
    for _ in range(starts):
        x0 = np.r_[rng.uniform(0, 2 * np.pi, depth), rng.uniform(0, np.pi, depth)]
        result = minimize(lambda params: evaluate(params)[0], x0, method="COBYLA", bounds=bounds, options={"maxiter": per_start})
        expected, probabilities = evaluate(result.x)
        if best is None or expected < best["expected_energy"]:
            best = {"expected_energy": float(expected), "probabilities": probabilities, "parameters": result.x.tolist(), "success": bool(result.success)}
    return best

def xy_qaoa_statevector(q: np.ndarray, k: int, depth: int, budget: int, shots: int, seed: int):
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    states = feasible_states(len(q), k)
    costs = np.array([energy(state, q) for state in states])
    dimension = len(states)
    mixer = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i + 1, dimension):
            if np.abs(states[i] - states[j]).sum() == 2:
                mixer[i, j] = mixer[j, i] = 1.0
    eigenvalues, eigenvectors = np.linalg.eigh(mixer)
    initial = np.ones(dimension, dtype=complex) / np.sqrt(dimension)  # Dicke state trong feasible subspace.
    scaled_costs = costs / max(float(np.max(np.abs(costs))), 1e-12)

    def evaluate(params):
        gammas, betas = params[:depth], params[depth:]
        psi = initial.copy()
        for gamma, beta in zip(gammas, betas):
            psi *= np.exp(-1j * gamma * scaled_costs)
            coefficients = eigenvectors.T.conj() @ psi
            psi = eigenvectors @ (np.exp(-1j * beta * eigenvalues) * coefficients)
        probabilities = np.abs(psi) ** 2
        probabilities /= probabilities.sum()
        return float(probabilities @ costs), probabilities

    optimized = optimize_qaoa_angles(evaluate, depth, budget, seed)
    sampled_indices = rng.choice(dimension, size=shots, p=optimized["probabilities"])
    sampled_states = states[sampled_indices]
    unique_states = np.unique(sampled_states, axis=0)
    best_observed = min(unique_states, key=lambda state: energy(state, q))
    exact_best = states[int(np.argmin(costs))]
    return {
        "bits": best_observed,
        "energy": energy(best_observed, q),
        "exact_energy": energy(exact_best, q),
        "optimality_gap": float((energy(best_observed, q) - energy(exact_best, q)) / max(abs(energy(exact_best, q)), 1e-12)),
        "expected_energy": optimized["expected_energy"],
        "feasibility_rate": float(np.mean(sampled_states.sum(axis=1) == k)),
        "success_probability": float(optimized["probabilities"][np.isclose(costs, costs.min())].sum()),
        "parameters": optimized["parameters"],
        "runtime_seconds": time.perf_counter() - start_time,
        "backend": "ideal_statevector_fixed_hamming_weight_XY_QAOA",
    }

def optimize_weights(mu, cov, lower, upper, risk_aversion):
    n = len(mu)
    initial = np.ones(n) / n
    result = minimize(
        lambda w: float(risk_aversion * w @ cov @ w - mu @ w),
        initial, method="SLSQP", bounds=[(lower, upper)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        return initial
    return np.asarray(result.x, float)
"""),
md(r"""
## Giai đoạn 6 — Walk-Forward Backtest ngoài mẫu

### 9. Hàm đánh giá tài chính và chi phí giao dịch

Tỷ trọng được giữ cố định trong từng test window. Tại ngày tái cân bằng, one-way turnover và transaction cost được tính theo:

\[
\mathrm{TO}_t=\frac12\sum_i|w_{i,t}-w_{i,t^-}|,
\qquad r^{net}_{t,1}=r^{gross}_{t,1}-\mathrm{TO}_t\frac{c_{bps}}{10^4}.
\]

Notebook báo cáo cumulative return, annualized return, annualized volatility, Sharpe, Sortino và maximum drawdown trên chuỗi return ròng. `FULL_UNIVERSE_EW` và `VNALLSHARE_TRI` được giữ làm baseline, nhưng paired statistical test chính vẫn là QAUR trừ AUR.
"""),
code(r"""
def portfolio_turnover(previous: dict[str, float], target: dict[str, float]) -> float:
    names = set(previous) | set(target)
    return 0.5 * sum(abs(target.get(name, 0.0) - previous.get(name, 0.0)) for name in names)

def financial_metrics(returns: pd.Series) -> dict:
    r = pd.Series(returns).dropna()
    if r.empty:
        return {}
    wealth = (1 + r).cumprod()
    annualized_return = float(wealth.iloc[-1] ** (252 / len(r)) - 1)
    annualized_volatility = float(r.std(ddof=1) * np.sqrt(252))
    drawdown = wealth / wealth.cummax() - 1
    downside = r[r < 0].std(ddof=1) * np.sqrt(252)
    return {
        "observations": len(r),
        "cumulative_return": float(wealth.iloc[-1] - 1),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_rf": annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan,
        "sortino_zero_rf": annualized_return / downside if pd.notna(downside) and downside > 0 else np.nan,
        "maximum_drawdown": float(drawdown.min()),
    }
"""),
md(r"""
### 10. Chạy pipeline end-to-end

Đây là vòng lặp nghiên cứu chính. Thứ tự trong mỗi fold được cố định: purge training labels → fit XGBoost → tạo point-in-time snapshot → ước lượng EWMA → chạy AUR và QAUR → đưa từng Top-\(K\) vào cùng \(Q^{PO}\)/XY-QAOA/SLSQP → ghi nhận test return. `FULL_UNIVERSE_EW` được tính độc lập như baseline cổ điển.

Cell chỉ in tiến độ ngắn gọn trong lúc chạy. Tất cả bảng kết quả, biểu đồ và phần giải thích tự động xuất hiện ở các cell ngay sau đó.
"""),
code(r"""
return_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()
benchmark_returns = benchmark.set_index("date")["benchmark_return"].sort_index() if not benchmark.empty else pd.Series(dtype=float)
selection_rows, solver_rows, return_rows, fold_rows, model_rows = [], [], [], [], []
previous_universe = {"AUR": set(), "QAUR": set()}
previous_weights = {"AUR": {}, "QAUR": {}}
previous_full_weights = {}

for fold in folds:
    fold_id = fold["fold"]
    print(f"Fold {fold_id + 1}/{len(folds)}", end=" ... ")
    # Một dòng train chỉ hợp lệ nếu target 20 phiên đã quan sát được trước train_end.
    train = features[
        (features["date"] >= fold["train_start"])
        & (features["date"] < fold["train_end"])
        & (features["target_available_at"] < fold["train_end"])
    ].copy()
    model, medians = fit_xgboost(train, CONFIG["seed"] + fold_id)

    decision_candidates = features[features["date"] < fold["test_start"]]
    decision_time = decision_candidates["date"].max()
    snapshot = decision_candidates[decision_candidates["date"].eq(decision_time)].copy()
    snapshot = snapshot[snapshot["ticker"].map(lambda ticker: point_in_time_eligible(ticker, decision_time))]
    history_start = decision_time - pd.Timedelta(days=CONFIG["minimum_history_days"] * 2)
    history = features[(features["date"] <= decision_time) & (features["date"] >= history_start)].copy()
    history_counts = history.groupby("ticker")["return_1d"].count()
    eligible = history_counts[history_counts >= CONFIG["minimum_history_days"]].index
    snapshot = snapshot[snapshot["ticker"].isin(eligible)].copy()
    if len(snapshot) < CONFIG["candidate_size"]:
        raise RuntimeError(f"Fold {fold_id}: chỉ có {len(snapshot)} tài sản đủ điều kiện.")
    snapshot["signal"] = predict_signal(model, medians, snapshot)

    validation = features[
        (features["date"] >= fold["validation_start"])
        & (features["date"] < fold["validation_end"])
        & (features["target_available_at"] < fold["validation_end"])
    ].dropna(subset=["target_rank"])
    if len(validation):
        X_validation, _ = prepare_matrix(validation, medians)
        val_pred = model.predict(X_validation)
        rank_ic = validation_daily_rank_ic(model, medians, validation)
        rmse = mean_squared_error(validation["target_rank"], val_pred) ** 0.5
    else:
        rank_ic, rmse = np.nan, np.nan
    model_rows.append({
        "fold": fold_id, "decision_time": decision_time,
        "validation_rank_ic": rank_ic, "validation_rmse": rmse,
        "train_rows": len(train), "universe_size": len(snapshot),
        "max_train_target_available_at": train["target_available_at"].max(),
        "train_end": fold["train_end"],
    })

    reducers = {
        "AUR": adaptive_universe_reduction(snapshot, history, previous_universe["AUR"], CONFIG["candidate_size"]),
        "QAUR": quantum_assisted_universe_reduction(snapshot, history, previous_universe["QAUR"], CONFIG["candidate_size"], CONFIG["seed"] + fold_id),
    }
    for method, reduced in reducers.items():
        candidates = reduced["tickers"]
        candidate_snapshot = snapshot.set_index("ticker").reindex(candidates)
        mu = candidate_snapshot["signal"].to_numpy(float)
        cov = ewma_covariance(history, candidates, CONFIG["covariance_span"])
        q = portfolio_qubo(mu, cov, CONFIG["risk_aversion_qubo"])
        solved = xy_qaoa_statevector(q, CONFIG["portfolio_cardinality"], CONFIG["qaoa_depth"], CONFIG["qaoa_parameter_budget"], CONFIG["qaoa_shots"], CONFIG["seed"] + fold_id)
        chosen_idx = np.flatnonzero(solved["bits"])
        chosen = [candidates[i] for i in chosen_idx]
        weights = optimize_weights(mu[chosen_idx], cov[np.ix_(chosen_idx, chosen_idx)], CONFIG["weight_lower"], CONFIG["weight_upper"], CONFIG["risk_aversion_weights"])
        target_weights = dict(zip(chosen, weights))
        turnover = portfolio_turnover(previous_weights[method], target_weights)
        test_returns = return_panel.loc[(return_panel.index >= fold["test_start"]) & (return_panel.index < fold["test_end"]), chosen].fillna(0.0)
        daily_returns = test_returns.to_numpy(float) @ weights
        if len(daily_returns):
            daily_returns[0] -= turnover * CONFIG["transaction_cost_bps"] / 10000.0
        for date, value in zip(test_returns.index, daily_returns):
            return_rows.append({"fold": fold_id, "date": date, "method": method, "return": float(value)})
        for ticker in candidates:
            selection_rows.append({
                "fold": fold_id,
                "decision_time": decision_time,
                "method": method,
                "ticker": ticker,
                "selected_by_xy_qaoa": ticker in chosen,
                "portfolio_weight": float(target_weights.get(ticker, 0.0)),
                "forecast_signal": float(candidate_snapshot.loc[ticker, "signal"]),
                "volatility_20d": float(candidate_snapshot.loc[ticker, "volatility_20d"]),
                "liquidity_20d": float(candidate_snapshot.loc[ticker, "liquidity_20d"]),
                "reduction_objective": reduced["objective"],
                "reduction_backend": reduced["backend"],
            })
        solver_rows.append({"fold": fold_id, "method": method, "energy": solved["energy"], "exact_energy": solved["exact_energy"], "optimality_gap": solved["optimality_gap"], "feasibility_rate": solved["feasibility_rate"], "success_probability": solved["success_probability"], "runtime_seconds": solved["runtime_seconds"], "turnover": turnover, "backend": solved["backend"]})
        previous_universe[method] = set(candidates)
        previous_weights[method] = target_weights

    full_names = snapshot["ticker"].tolist()
    full_target = {ticker: 1.0 / len(full_names) for ticker in full_names}
    full_turnover = portfolio_turnover(previous_full_weights, full_target)
    full_test = return_panel.loc[(return_panel.index >= fold["test_start"]) & (return_panel.index < fold["test_end"]), full_names].fillna(0.0)
    full_daily = full_test.mean(axis=1).to_numpy(float)
    if len(full_daily):
        full_daily[0] -= full_turnover * CONFIG["transaction_cost_bps"] / 10000.0
    for date, value in zip(full_test.index, full_daily):
        return_rows.append({"fold": fold_id, "date": date, "method": "FULL_UNIVERSE_EW", "return": float(value)})
    previous_full_weights = full_target
    if not benchmark_returns.empty:
        benchmark_test = benchmark_returns[(benchmark_returns.index >= fold["test_start"]) & (benchmark_returns.index < fold["test_end"])].dropna()
        for date, value in benchmark_test.items():
            return_rows.append({"fold": fold_id, "date": date, "method": "VNALLSHARE_TRI", "return": float(value)})
    fold_rows.append({**fold, "decision_time": decision_time, "eligible_universe_size": len(snapshot)})
    print("done")

selections = pd.DataFrame(selection_rows)
solver_diagnostics = pd.DataFrame(solver_rows)
portfolio_returns = pd.DataFrame(return_rows)
fold_manifest = pd.DataFrame(fold_rows)
model_diagnostics = pd.DataFrame(model_rows)
print("Hoàn tất", len(folds), "folds và", portfolio_returns["date"].nunique(), "phiên ngoài mẫu.")
"""),
md("""
## 11. Kết quả hiển thị trực tiếp trên Colab

Phần này không đọc bảng kết quả bên ngoài và không dùng số hard-code. Tất cả bảng được tạo từ object của lần chạy hiện tại. Trình tự đọc kết quả là: forecast quality → candidate-set comparison → final asset selection → shared solver diagnostics → portfolio performance → paired statistical tests.
"""),
code(r"""
performance_summary = pd.DataFrame([
    {"method": method, **financial_metrics(group.sort_values("date")["return"])}
    for method, group in portfolio_returns.groupby("method")
])
method_order = ["AUR", "QAUR", "FULL_UNIVERSE_EW", "VNALLSHARE_TRI"]
performance_summary["method"] = pd.Categorical(performance_summary["method"], method_order, ordered=True)
performance_summary = performance_summary.sort_values("method").reset_index(drop=True)

overlap_rows = []
for fold_id, group in selections.groupby("fold"):
    a = set(group[group["method"].eq("AUR")]["ticker"])
    q = set(group[group["method"].eq("QAUR")]["ticker"])
    a_final = set(group[group["method"].eq("AUR") & group["selected_by_xy_qaoa"]]["ticker"])
    q_final = set(group[group["method"].eq("QAUR") & group["selected_by_xy_qaoa"]]["ticker"])
    overlap_rows.append({
        "fold": fold_id,
        "candidate_intersection": len(a & q),
        "candidate_union": len(a | q),
        "candidate_jaccard": len(a & q) / len(a | q),
        "candidate_sets_identical": a == q,
        "portfolio_intersection": len(a_final & q_final),
        "portfolio_union": len(a_final | q_final),
        "portfolio_jaccard": len(a_final & q_final) / len(a_final | q_final),
        "portfolio_sets_identical": a_final == q_final,
    })
reduction_comparison = pd.DataFrame(overlap_rows)

stability_rows = []
for method, method_frame in selections.groupby("method"):
    previous_set = None
    for fold_id, fold_frame in method_frame.groupby("fold"):
        current_set = set(fold_frame["ticker"])
        stability = np.nan if previous_set is None else len(previous_set & current_set) / len(previous_set | current_set)
        stability_rows.append({"method": method, "fold": fold_id, "candidate_stability_jaccard": stability})
        previous_set = current_set
reduction_stability = pd.DataFrame(stability_rows)

reduction_summary = pd.DataFrame({
    "metric": [
        "Mean candidate Jaccard", "Minimum candidate Jaccard",
        "Identical candidate folds", "Mean final-portfolio Jaccard",
        "Identical final-portfolio folds",
    ],
    "value": [
        reduction_comparison["candidate_jaccard"].mean(),
        reduction_comparison["candidate_jaccard"].min(),
        int(reduction_comparison["candidate_sets_identical"].sum()),
        reduction_comparison["portfolio_jaccard"].mean(),
        int(reduction_comparison["portfolio_sets_identical"].sum()),
    ],
})

candidate_sets_by_fold = (
    selections.groupby(["fold", "decision_time", "method"])["ticker"]
    .apply(lambda values: ", ".join(sorted(values)))
    .unstack("method").reset_index()
)
final_portfolios_by_fold = (
    selections[selections["selected_by_xy_qaoa"]]
    .sort_values(["fold", "method", "portfolio_weight"], ascending=[True, True, False])
    .groupby(["fold", "decision_time", "method"])
    .apply(
        lambda frame: ", ".join(
            f'{row.ticker} ({row.portfolio_weight:.1%})' for row in frame.itertuples()
        ),
        include_groups=False,
    ).rename("assets_and_weights").unstack("method").reset_index()
)

wide = portfolio_returns.pivot(index="date", columns="method", values="return").sort_index()
paired = wide[["AUR", "QAUR"]].dropna()
difference = paired["QAUR"] - paired["AUR"]
t_result = stats.ttest_rel(paired["QAUR"], paired["AUR"])
wilcoxon_result = stats.wilcoxon(difference) if (difference != 0).any() else None
statistical_tests = pd.DataFrame([{
    "comparison": "QAUR_minus_AUR",
    "n": len(difference),
    "mean_daily_difference": difference.mean(),
    "paired_t_statistic": t_result.statistic,
    "paired_t_pvalue": t_result.pvalue,
    "wilcoxon_statistic": wilcoxon_result.statistic if wilcoxon_result else 0.0,
    "wilcoxon_pvalue": wilcoxon_result.pvalue if wilcoxon_result else 1.0,
}])

model_summary = model_diagnostics[["validation_rank_ic", "validation_rmse", "train_rows", "universe_size"]].agg(["mean", "std", "min", "max"]).T.reset_index(names="metric")
solver_summary = solver_diagnostics.groupby("method", as_index=False)[["feasibility_rate", "optimality_gap", "success_probability", "runtime_seconds", "turnover"]].mean()

display(Markdown("### 11.1. Chất lượng forecast XGBoost và quy mô universe"))
display(model_summary.style.format(precision=6))
display(Markdown("**Cách đọc:** Rank IC dương nghĩa là forecast có xu hướng xếp hạng đúng return tương lai; RMSE thấp hơn nghĩa là predicted target rank gần observed target rank hơn."))

display(Markdown("### 11.2. So sánh candidate set AUR và QAUR"))
display(reduction_summary.style.format({"value": "{:.4f}"}))
display(reduction_comparison.style.format({"candidate_jaccard": "{:.4f}", "portfolio_jaccard": "{:.4f}"}))
display(Markdown("#### Candidate set của từng fold"))
display(candidate_sets_by_fold)

display(Markdown("### 11.3. Danh mục sau shared XY-QAOA và tỷ trọng SLSQP"))
display(final_portfolios_by_fold)
display(Markdown("**Cách đọc:** mỗi ô liệt kê đúng $k_p$ tài sản; số trong ngoặc là tỷ trọng do classical optimizer phân bổ, không phải xác suất đo của QAOA."))

display(Markdown("### 11.4. Shared XY-QAOA diagnostics"))
display(solver_summary.style.format(precision=6))
display(Markdown("**Cách đọc:** feasibility đo tỷ lệ sample có đúng cardinality; optimality gap so với exact enumeration trong candidate set; success probability là xác suất của optimal state dưới statevector sau tối ưu góc."))

display(Markdown("### 11.5. Hiệu quả tài chính ngoài mẫu"))
display(performance_summary.style.format({
    "cumulative_return": "{:.2%}", "annualized_return": "{:.2%}",
    "annualized_volatility": "{:.2%}", "sharpe_zero_rf": "{:.4f}",
    "sortino_zero_rf": "{:.4f}", "maximum_drawdown": "{:.2%}",
}))

display(Markdown("### 11.6. Paired statistical tests: QAUR minus AUR"))
display(statistical_tests.style.format(precision=6))
display(Markdown("**Cách đọc:** p-value lớn không chứng minh hai phương pháp tương đương; nó chỉ cho biết dữ liệu hiện tại chưa đủ bằng chứng bác bỏ giả thuyết chênh lệch trung bình bằng 0."))
"""),
md("""
## 12. Diễn giải kết quả tự động

Cell này chuyển các output định lượng thành narrative có kiểm soát. Mọi con số được lấy trực tiếp từ bảng phía trên; câu kết luận thay đổi theo dấu của chênh lệch, p-value, Jaccard và feasibility. Phần diễn giải tách riêng bốn tầng: forecast, universe reduction, shared portfolio selection và investment performance.
"""),
code(r"""
performance_index = performance_summary.set_index("method")
aur_perf = performance_index.loc["AUR"]
qaur_perf = performance_index.loc["QAUR"]
mean_candidate_jaccard = float(reduction_comparison["candidate_jaccard"].mean())
mean_portfolio_jaccard = float(reduction_comparison["portfolio_jaccard"].mean())
identical_candidate_folds = int(reduction_comparison["candidate_sets_identical"].sum())
identical_portfolio_folds = int(reduction_comparison["portfolio_sets_identical"].sum())
mean_rank_ic = float(model_diagnostics["validation_rank_ic"].mean())
mean_rmse = float(model_diagnostics["validation_rmse"].mean())
minimum_feasibility = float(solver_diagnostics["feasibility_rate"].min())
mean_gap = float(solver_diagnostics["optimality_gap"].mean())
mean_daily_difference = float(statistical_tests.loc[0, "mean_daily_difference"])
paired_pvalue = float(statistical_tests.loc[0, "paired_t_pvalue"])

forecast_assessment = (
    "dương, cho thấy model có khả năng xếp hạng cùng chiều với return tương lai ở mức trung bình"
    if mean_rank_ic > 0 else
    "không dương, vì vậy forecast signal chưa cho thấy khả năng xếp hạng ổn định trong validation"
)
similarity_assessment = (
    "rất cao" if mean_candidate_jaccard >= 0.80 else
    "tương đối cao" if mean_candidate_jaccard >= 0.60 else
    "trung bình" if mean_candidate_jaccard >= 0.40 else "thấp"
)
statistical_assessment = (
    "chênh lệch có ý nghĩa thống kê ở ngưỡng 5%"
    if paired_pvalue < 0.05 else
    "chưa có ý nghĩa thống kê ở ngưỡng 5%"
)
daily_direction = "cao hơn" if mean_daily_difference > 0 else "thấp hơn" if mean_daily_difference < 0 else "bằng"
profile_warning = (
    "**Lưu ý:** đây là SMOKE run nên các con số chỉ dùng để kiểm tra kỹ thuật, không dùng làm kết luận nghiên cứu."
    if EXECUTION_PROFILE.upper() == "SMOKE" else
    "Đây là FULL run trên toàn bộ walk-forward folds khả dụng."
)

key_findings = pd.DataFrame([
    ("Validation Rank IC trung bình", mean_rank_ic),
    ("Validation RMSE trung bình", mean_rmse),
    ("Candidate Jaccard trung bình", mean_candidate_jaccard),
    ("Final-portfolio Jaccard trung bình", mean_portfolio_jaccard),
    ("AUR annualized return", float(aur_perf["annualized_return"])),
    ("QAUR annualized return", float(qaur_perf["annualized_return"])),
    ("QAUR − AUR mean daily return", mean_daily_difference),
    ("Paired t-test p-value", paired_pvalue),
    ("Minimum XY-QAOA feasibility", minimum_feasibility),
    ("Mean XY-QAOA optimality gap", mean_gap),
], columns=["Kết quả chính", "Giá trị"])
display(key_findings.style.format({"Giá trị": "{:.6f}"}))

interpretation_text = fr'''
### 12.1. Forecast và risk estimation

Validation Rank IC trung bình là **{mean_rank_ic:.4f}** và RMSE trung bình là **{mean_rmse:.4f}**. Rank IC {forecast_assessment}. EWMA được ước lượng lại tại từng decision date và dùng chung cho cả hai reducer, do đó forecast/risk model không phải là biến khác biệt giữa AUR và QAUR.

### 12.2. Universe-reduction effect

Candidate-set Jaccard trung bình đạt **{mean_candidate_jaccard:.4f}**, thể hiện mức tương đồng {similarity_assessment}. Hai reducer tạo candidate set hoàn toàn giống nhau ở **{identical_candidate_folds}/{len(folds)} fold**. Sự khác biệt còn lại đến từ search mechanism: AUR thực hiện greedy marginal selection, trong khi QAUR tối ưu pairwise redundancy bằng cardinality-preserving swap search trên cùng \(Q^{{UR}}\) objective.

### 12.3. Downstream portfolio effect

Sau shared XY-QAOA, Jaccard của final asset sets trung bình là **{mean_portfolio_jaccard:.4f}** và có **{identical_portfolio_folds}/{len(folds)} fold** tạo cùng tập tài sản cuối. Điều này giải thích vì sao khác biệt ở Top-\(K\) có thể bị hấp thụ tại asset-selection layer. Minimum feasibility rate đạt **{minimum_feasibility:.2%}**; mean optimality gap là **{mean_gap:.6f}**. Feasibility cao chỉ cho thấy cardinality constraint được bảo toàn, không phải bằng chứng quantum advantage.

### 12.4. Investment performance và statistical evidence

AUR có annualized return **{aur_perf['annualized_return']:.2%}**, Sharpe **{aur_perf['sharpe_zero_rf']:.4f}** và maximum drawdown **{aur_perf['maximum_drawdown']:.2%}**. QAUR có annualized return **{qaur_perf['annualized_return']:.2%}**, Sharpe **{qaur_perf['sharpe_zero_rf']:.4f}** và maximum drawdown **{qaur_perf['maximum_drawdown']:.2%}**. Mean daily return của QAUR {daily_direction} AUR một lượng **{abs(mean_daily_difference):.6f}**; paired t-test cho \(p={paired_pvalue:.4f}\), tức {statistical_assessment}. Kết luận này không đồng nghĩa với chứng minh equivalence khi p-value lớn.

### 12.5. Kết luận hợp lệ và giới hạn

{profile_warning} Đối tượng so sánh hợp lệ là AUR với QAUR; XY-QAOA và SLSQP là downstream dùng chung. QAUR hiện chạy classical surrogate và XY-QAOA chạy ideal statevector trên candidate set nhỏ. Vì vậy, notebook chỉ đánh giá formulation, candidate selection và portfolio outcome trong môi trường mô phỏng; không kết luận quantum speedup, hardware superiority hoặc lợi thế lượng tử.
'''
display(Markdown(interpretation_text))
"""),
md("""
## 13. Fail-fast research audit

Cell này dừng notebook nếu phát hiện leakage, sai cardinality, khác decision date giữa hai reducer, solver backend không dùng chung, nghiệm XY-QAOA không khả thi, tỷ trọng sai ràng buộc hoặc duplicate strategy return. Audit phải qua trước khi tạo biểu đồ và export kết quả.
"""),
code(r"""
audit_checks = []
def audit(name: str, passed: bool, detail: str):
    audit_checks.append({"check": name, "passed": bool(passed), "detail": detail})

candidate_counts = selections.groupby(["fold", "method"])["ticker"].nunique()
selected_counts = selections[selections["selected_by_xy_qaoa"]].groupby(["fold", "method"])["ticker"].nunique()
audit("top_k_cardinality", bool((candidate_counts == CONFIG["candidate_size"]).all()), candidate_counts.to_dict().__str__())
audit("portfolio_cardinality", bool((selected_counts == CONFIG["portfolio_cardinality"]).all()), selected_counts.to_dict().__str__())
audit("xy_qaoa_feasibility", bool((solver_diagnostics["feasibility_rate"] == 1.0).all()), f"minimum={solver_diagnostics['feasibility_rate'].min()}")
audit("no_duplicate_strategy_returns", not portfolio_returns.duplicated(["date", "method"]).any(), f"duplicates={portfolio_returns.duplicated(['date', 'method']).sum()}")
audit("training_labels_available_before_train_end", bool((model_diagnostics["max_train_target_available_at"] < model_diagnostics["train_end"]).all()), "target_available_at < train_end for every fold")
audit("same_reducer_fold_coverage", selections.groupby("method")["fold"].nunique().nunique() == 1, selections.groupby("method")["fold"].nunique().to_dict().__str__())
audit("finite_portfolio_returns", bool(np.isfinite(portfolio_returns["return"]).all()), "all strategy returns finite")
audit("same_decision_time_per_fold", bool((selections.groupby("fold")["decision_time"].nunique() == 1).all()), "AUR and QAUR use the same decision snapshot in every fold")
audit("shared_xy_qaoa_backend", solver_diagnostics["backend"].nunique() == 1, solver_diagnostics["backend"].unique().__str__())
weight_sums = selections[selections["selected_by_xy_qaoa"]].groupby(["fold", "method"])["portfolio_weight"].sum()
audit("portfolio_weights_sum_to_one", bool(np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-7)), f"min={weight_sums.min():.8f}, max={weight_sums.max():.8f}")
selected_weights = selections.loc[selections["selected_by_xy_qaoa"], "portfolio_weight"]
within_bounds = selected_weights.between(CONFIG["weight_lower"] - 1e-7, CONFIG["weight_upper"] + 1e-7).all()
audit("portfolio_weights_within_bounds", bool(within_bounds), f"min={selected_weights.min():.6f}, max={selected_weights.max():.6f}")
audit("result_tables_nonempty", all(len(frame) > 0 for frame in [performance_summary, reduction_comparison, solver_summary, statistical_tests]), "all required result tables contain rows")

audit_results = pd.DataFrame(audit_checks)
audit_results.insert(0, "status", np.where(audit_results["passed"], "PASS", "FAIL"))
display(audit_results.style.map(lambda value: "color: #0a7f35; font-weight: bold" if value == "PASS" else "color: #b42318; font-weight: bold" if value == "FAIL" else "", subset=["status"]))
failed = audit_results[~audit_results["passed"]]
if len(failed):
    raise AssertionError("Research audit failed:\n" + failed.to_string(index=False))
print("RESEARCH_AUDIT_OK — tất cả kiểm tra bắt buộc đã qua.")
"""),
md("""
## 14. Biểu đồ và cách đọc kết quả

Các figure được tạo trực tiếp từ output của lần chạy hiện tại, không sử dụng số liệu hard-code. Equity curve và drawdown phản ánh performance ngoài mẫu; Jaccard và candidate stability phản ánh universe-reduction effect; Rank IC phản ánh forecast layer; feasibility/gap phản ánh shared solver layer.
"""),
code(r"""
figures_dir = RESULTS / "figures"
figures_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.lineplot(data=model_diagnostics, x="fold", y="validation_rank_ic", marker="o", ax=axes[0], color="#2563eb")
axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
axes[0].set_title("Validation Rank IC by fold")
sns.lineplot(data=model_diagnostics, x="fold", y="universe_size", marker="o", ax=axes[1], color="#0f766e")
axes[1].set_title("Point-in-time eligible universe size")
plt.tight_layout(); plt.savefig(figures_dir / "forecast_and_universe_diagnostics.png", dpi=180); plt.show()
display(Markdown(f"**Diễn giải:** Rank IC dao động quanh mức trung bình **{mean_rank_ic:.4f}**. Universe size được tính tại từng decision date, vì vậy thay đổi theo fold là hợp lệ và không đồng nghĩa với missing data."))

wide_returns = portfolio_returns.pivot(index="date", columns="method", values="return").sort_index().fillna(0.0)
equity = (1 + wide_returns).cumprod()
ax = equity.plot(figsize=(12, 6), linewidth=2, title="Out-of-sample equity curves")
ax.set_ylabel("Growth of 1 unit"); ax.set_xlabel("Date")
plt.tight_layout(); plt.savefig(figures_dir / "equity_curve.png", dpi=180); plt.show()
display(Markdown("**Diễn giải:** đường cao hơn tại một thời điểm thể hiện giá trị tích lũy lớn hơn đến thời điểm đó; hình dạng đường không tự chứng minh statistical significance."))

drawdown = equity / equity.cummax() - 1
ax = drawdown.plot(figsize=(12, 5), linewidth=1.8, title="Out-of-sample drawdown")
ax.set_ylabel("Drawdown"); ax.set_xlabel("Date")
plt.tight_layout(); plt.savefig(figures_dir / "drawdown.png", dpi=180); plt.show()
display(Markdown("**Diễn giải:** drawdown đo mức giảm từ đỉnh wealth gần nhất; giá trị âm lớn hơn về độ lớn thể hiện tổn thất sâu hơn."))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.barplot(data=performance_summary, x="method", y="annualized_return", ax=axes[0])
axes[0].set_title("Annualized return"); axes[0].tick_params(axis="x", rotation=20)
sns.barplot(data=performance_summary, x="method", y="sharpe_zero_rf", ax=axes[1])
axes[1].set_title("Sharpe ratio (rf = 0)"); axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(figures_dir / "performance_comparison.png", dpi=180); plt.show()
display(Markdown("**Diễn giải:** annualized return đo tăng trưởng, còn Sharpe chuẩn hóa return theo volatility. Hai metric phải được đọc cùng maximum drawdown và paired tests."))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(reduction_comparison["fold"], reduction_comparison["candidate_jaccard"], marker="o", label="Candidate Top-K")
axes[0].plot(reduction_comparison["fold"], reduction_comparison["portfolio_jaccard"], marker="s", label="Final portfolio")
axes[0].set_ylim(0, 1.05); axes[0].set_title("AUR–QAUR Jaccard by fold"); axes[0].legend()
sns.lineplot(data=solver_diagnostics, x="fold", y="turnover", hue="method", marker="o", ax=axes[1])
axes[1].set_title("Portfolio turnover by fold")
plt.tight_layout(); plt.savefig(figures_dir / "reduction_and_turnover.png", dpi=180); plt.show()
display(Markdown(f"**Diễn giải:** candidate Jaccard trung bình **{mean_candidate_jaccard:.4f}** cho biết reducer khác nhau đến mức nào; final-portfolio Jaccard trung bình **{mean_portfolio_jaccard:.4f}** cho biết bao nhiêu khác biệt còn tồn tại sau shared XY-QAOA."))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.lineplot(data=solver_diagnostics, x="fold", y="feasibility_rate", hue="method", marker="o", ax=axes[0])
axes[0].set_ylim(0, 1.05); axes[0].set_title("XY-QAOA feasibility rate")
sns.lineplot(data=solver_diagnostics, x="fold", y="optimality_gap", hue="method", marker="o", ax=axes[1])
axes[1].axhline(0, color="black", linewidth=1, linestyle="--"); axes[1].set_title("Observed optimality gap")
plt.tight_layout(); plt.savefig(figures_dir / "xy_qaoa_diagnostics.png", dpi=180); plt.show()
display(Markdown("**Diễn giải:** feasibility bằng 1 xác nhận mọi sample có đúng $k_p$ tài sản. Optimality gap bằng 0 nghĩa là best sampled state trùng exact best state trong feasible basis; đây vẫn là simulator-level evidence."))

selection_frequency = (
    selections[selections["selected_by_xy_qaoa"]]
    .groupby(["method", "ticker"]).size().rename("selected_folds").reset_index()
)
top_names = selection_frequency.groupby("ticker")["selected_folds"].sum().nlargest(15).index
plt.figure(figsize=(12, 5))
sns.barplot(data=selection_frequency[selection_frequency["ticker"].isin(top_names)], x="ticker", y="selected_folds", hue="method")
plt.title("Most frequently selected assets after XY-QAOA"); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(figures_dir / "selection_frequency.png", dpi=180); plt.show()
display(Markdown("**Diễn giải:** biểu đồ tần suất cho biết stability của final asset selection qua các fold; tần suất cao không đồng nghĩa với future return cao."))
"""),
md("""
## 15. Lưu artifact và tạo gói kết quả

Cell cuối lưu config, dataset hash, fold manifest, forecast diagnostics, Top-K, asset selection, solver diagnostics, returns, metrics, statistical tests và figures. Sau đó toàn bộ thư mục được nén để tải về.
"""),
code(r"""
artifacts = {
    "fold_manifest.csv": fold_manifest,
    "model_diagnostics.csv": model_diagnostics,
    "universe_and_asset_selections.csv": selections,
    "solver_diagnostics.csv": solver_diagnostics,
    "portfolio_returns.csv": portfolio_returns,
    "performance_summary.csv": performance_summary,
    "reduction_comparison.csv": reduction_comparison,
    "reduction_stability.csv": reduction_stability,
    "candidate_sets_by_fold.csv": candidate_sets_by_fold,
    "final_portfolios_by_fold.csv": final_portfolios_by_fold,
    "key_findings.csv": key_findings,
    "selection_frequency.csv": selection_frequency,
    "statistical_tests.csv": statistical_tests,
    "audit_results.csv": audit_results,
}
for filename, frame in artifacts.items():
    frame.to_csv(RESULTS / filename, index=False)

manifest = {
    "framework": "AUR_vs_QAUR_shared_cardinality_QUBO_XY_QAOA",
    "execution_profile": EXECUTION_PROFILE,
    "dataset_filename": csv_path.name,
    "dataset_sha256": DATASET_SHA256,
    "folds_completed": len(folds),
    "out_of_sample_days": int(portfolio_returns["date"].nunique()),
    "mean_candidate_jaccard": mean_candidate_jaccard,
    "mean_final_portfolio_jaccard": mean_portfolio_jaccard,
    "minimum_xy_qaoa_feasibility": minimum_feasibility,
    "mean_xy_qaoa_optimality_gap": mean_gap,
    "paired_ttest_pvalue": paired_pvalue,
    "config": CONFIG,
    "qa_backend_disclosure": "classical cardinality-preserving surrogate for quantum-ready universe-reduction QUBO",
    "xy_qaoa_backend_disclosure": "ideal statevector simulation restricted to fixed-Hamming-weight feasible subspace",
    "quantum_advantage_claimed": False,
}
(RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
(RESULTS / "research_interpretation.md").write_text(interpretation_text, encoding="utf-8")

readme = f'''# Kết quả AUR vs QAUR standalone Colab

- Profile: {EXECUTION_PROFILE}
- Dataset SHA-256: {DATASET_SHA256}
- Folds: {len(folds)}
- Out-of-sample days: {portfolio_returns['date'].nunique()}
- Mean candidate Jaccard: {mean_candidate_jaccard:.6f}
- Mean final-portfolio Jaccard: {mean_portfolio_jaccard:.6f}
- Minimum XY-QAOA feasibility: {minimum_feasibility:.6f}
- Mean XY-QAOA optimality gap: {mean_gap:.6f}
- Paired t-test p-value: {paired_pvalue:.6f}
- QAUR backend: classical surrogate for quantum-ready QUBO
- Shared downstream: fixed-Hamming-weight XY-QAOA statevector
- Quantum advantage claimed: No
'''
(RESULTS / "README.md").write_text(readme, encoding="utf-8")

archive_path = Path("/content/AUR_QAUR_XYQAOA_RESULTS.zip")
if archive_path.exists():
    archive_path.unlink()
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in RESULTS.rglob("*"):
        if path.is_file():
            archive.write(path, path.relative_to(RESULTS.parent))

print("Đã tạo:", archive_path, "| size:", archive_path.stat().st_size, "bytes")
print("Tải file kết quả từ bảng Files của Colab: AUR_QAUR_XYQAOA_RESULTS.zip")
artifact_inventory = pd.DataFrame([
    {"artifact": path.relative_to(RESULTS).as_posix(), "bytes": path.stat().st_size}
    for path in sorted(RESULTS.rglob("*")) if path.is_file()
])
display(Markdown("### Danh sách artifact đã xuất"))
display(artifact_inventory)
display(Markdown(f"**Hoàn tất:** `{archive_path.name}` đã được tạo với dung lượng **{archive_path.stat().st_size:,} bytes**. Mở bảng **Files** ở cạnh trái Colab để tải file."))
"""),
md("""
## 16. Quy tắc diễn giải và checklist bàn giao

1. Đối tượng so sánh là **AUR và QAUR**; XY-QAOA là shared downstream solver, không phải đối thủ trực tiếp của AUR.
2. Candidate-level, final-selection-level và portfolio-level results phải được đọc tách biệt.
3. Feasibility 100% không đồng nghĩa với optimality, financial superiority hoặc quantum advantage.
4. QAUR backend hiện là classical surrogate; XY-QAOA backend hiện là ideal statevector simulator.
5. P-value lớn không chứng minh equivalence; p-value nhỏ cũng không tự chứng minh economic significance.
6. `SMOKE` chỉ dùng để kiểm tra code/schema; chỉ `FULL` run mới được cân nhắc cho báo cáo nghiên cứu.
7. Mọi bảng, biểu đồ và câu diễn giải phía trên được sinh từ lần chạy hiện tại; file ZIP kết quả chứa manifest, hash dữ liệu và toàn bộ artifact phục vụ tái lập.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT.relative_to(ROOT).as_posix())
