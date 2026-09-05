from __future__ import annotations

"""Build the final, self-contained research-submission Google Colab."""

import json
import copy
from pathlib import Path

import build_colab_29_8_complete as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab" / "AUR_QAUR_RESEARCH_SUBMISSION_FULL_CODE.ipynb"
LIGHT_OUTPUT = ROOT / "colab" / "AUR_QAUR_RESEARCH_SUBMISSION_COLAB.ipynb"
DATA_DRIVE_ID = "18oKxxmGSJA01o6MDYH7tBTEg23P3O6d9"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.rstrip() + "\n",
    }


def main() -> None:
    # Regenerate the source notebook first so the embedded implementation is
    # always byte-for-byte aligned with the current research engine.
    base.main()
    source_notebook = json.loads(base.WEB_OUTPUT.read_text(encoding="utf-8"))
    install_cell = source_notebook["cells"][2]
    embedded_data_cell = source_notebook["cells"][4]
    core_source_cell = source_notebook["cells"][6]
    orchestrator_source_cell = source_notebook["cells"][8]
    run_cell = source_notebook["cells"][10]

    cells = [
        markdown(r"""
# AUR–QAUR Portfolio Research — Full Executable Submission Colab

Notebook này là **bằng chứng code và thực nghiệm có thể tái lập** của toàn bộ
hệ thống. Dữ liệu thật đã được nén, nhúng trực tiếp và khóa bằng SHA-256; toàn
bộ source code được viết trong notebook, không clone GitHub, không mount Drive,
không yêu cầu upload thủ công và không tải ra file kết quả. Chỉ cần chọn
**Runtime → Run all** trên Google Colab CPU.

Đối tượng so sánh là **Adaptive Universe Reduction (AUR)** và
**Quantum-Assisted Universe Reduction (QAUR)**. XY-QAOA là thuật toán lựa chọn
danh mục dùng chung cho hai nhánh, không phải đối tượng so sánh với AUR.

Notebook phân biệt hai lớp bằng chứng:

1. **Confirmatory historical:** chọn cấu hình trên development folds 0–28 và
   đánh giá cấu hình đã khóa trên untouched holdout folds 29–43.
2. **Practical method design:** đánh giá 24 cấu hình với bốn market gates trên
   dữ liệu đã quan sát đến 28/08/2026. Đây là phân tích post-hoc, không phải
   bằng chứng prospective.
"""),
        markdown(r"""
## Kiến trúc phương pháp luận và ký hiệu

\[
\mathcal U_t
\xrightarrow{\text{XGBoost/EWMA}}
(\hat\mu_t,\hat\Sigma_t)
\begin{cases}
\xrightarrow{\mathrm{AUR}}\mathrm{Top}\text{-}K_A\\
\xrightarrow{\mathrm{QAUR}\;(Q^{UR})}\mathrm{Top}\text{-}K_{QA}
\end{cases}
\xrightarrow{\text{cùng }Q^{PO}\text{ và XY-QAOA}}
k_p\text{ tài sản}
\xrightarrow{\text{classical weights}}
w_t
\xrightarrow{\text{walk-forward}}
r^{OOS}_{t+1}.
\]

Hai reducer nhận cùng point-in-time universe \(\mathcal U_t\), cùng forecast
\(\hat\mu_t\), cùng EWMA covariance \(\hat\Sigma_t\), cùng lịch folds và cùng
các giả định downstream. Vì vậy, chênh lệch giữa hai nhánh được quy chủ yếu cho
cơ chế universe reduction. Framework gồm đúng sáu giai đoạn:

1. Dự báo lợi suất và rủi ro;
2. Adaptive Universe Reduction;
3. Quantum-Assisted Universe Reduction;
4. So sánh hai phương pháp giảm vũ trụ;
5. Portfolio Optimization dùng chung;
6. Walk-Forward Backtest ngoài mẫu.
"""),
        markdown("## 0. Cài đặt thư viện và môi trường tái lập"),
        install_cell,
        markdown(r"""
## 1. Dữ liệu thực, checksum và kiểm tra chất lượng

Bộ dữ liệu được nhúng trực tiếp nhằm bảo đảm `Run all` không phụ thuộc một file
bên ngoài. Cell đầu tiên xác minh SHA-256 trước khi giải nén; cell tiếp theo báo
cáo quy mô mẫu, thời gian quan sát, missing values và duplicate price keys.
Dữ liệu synthetic không được sử dụng trong thực nghiệm chính.
"""),
        embedded_data_cell,
        code(r"""
import platform
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from IPython.display import display, Markdown

raw_data = pd.read_csv(DATASET, low_memory=False)
price_raw = raw_data[raw_data["record_type"].eq("PRICE")].copy()
price_raw["date"] = pd.to_datetime(price_raw["date"], errors="coerce")
security_raw = raw_data[raw_data["record_type"].eq("SECURITY")].copy()
benchmark_raw = raw_data[raw_data["record_type"].eq("BENCHMARK")].copy()

duplicate_price_keys = int(price_raw.duplicated(["ticker", "date"]).sum())
required_fields = ["record_type", "date", "ticker", "adjusted_close", "volume", "trading_value"]
missing_required = raw_data[required_fields].isna().sum().rename("missing_values").to_frame()

data_overview = pd.DataFrame([
    {"indicator": "Total records", "value": len(raw_data)},
    {"indicator": "Price records", "value": len(price_raw)},
    {"indicator": "Point-in-time security records", "value": len(security_raw)},
    {"indicator": "Benchmark records", "value": len(benchmark_raw)},
    {"indicator": "Unique price tickers", "value": price_raw["ticker"].nunique()},
    {"indicator": "First price date", "value": str(price_raw["date"].min().date())},
    {"indicator": "Last observed price date", "value": str(price_raw["date"].max().date())},
    {"indicator": "Duplicate (ticker, date) price keys", "value": duplicate_price_keys},
    {"indicator": "Dataset SHA-256", "value": digest},
])
environment = pd.DataFrame([
    {"component": "Python", "version": platform.python_version()},
    {"component": "NumPy", "version": np.__version__},
    {"component": "pandas", "version": pd.__version__},
    {"component": "SciPy", "version": scipy.__version__},
    {"component": "scikit-learn", "version": sklearn.__version__},
    {"component": "XGBoost", "version": xgboost.__version__},
])
display(Markdown("### Quy mô và phạm vi dữ liệu"))
display(data_overview)
display(Markdown("### Missing values trong các trường hợp đồng dữ liệu"))
display(missing_required)
display(Markdown("### Môi trường thực thi"))
display(environment)
"""),
        markdown(r"""
## 2. Toàn bộ source code của phương pháp

Cell dưới đây viết trực tiếp experimental engine vào runtime. Source bao gồm
feature engineering, purged walk-forward forecast, XGBoost, EWMA, point-in-time
eligibility, AUR, QAUR, \(Q^{UR}\), \(Q^{PO}\), exact fixed-cardinality
reference, XY-QAOA statevector audit, bounded weight allocation, turnover,
transaction costs, market gate và các kiểm định thống kê.

### Giai đoạn 1 — Dự báo lợi suất và rủi ro

\[
y_{i,t}^{(h)}=\frac{P_{i,t+h}}{P_{i,t}}-1,\qquad
\hat\Sigma_t=(1-\lambda)\sum_{\tau\le t}\lambda^{t-\tau}
(r_\tau-\bar r_t)(r_\tau-\bar r_t)^\top.
\]

Training labels được purge trước decision time. Validation Rank IC chỉ là chẩn
đoán năng lực xếp hạng và không được dùng như một quan sát độc lập về alpha.

### Giai đoạn 2 — Adaptive Universe Reduction

\[
s_{i,t}=w_sS_{i,t}+w_lL_{i,t}+w_rR_{i,t}+w_hH_{i,t}.
\]

AUR xây candidate set tuần tự bằng cách cân bằng unary quality với incremental
correlation redundancy.

### Giai đoạn 3 — Quantum-Assisted Universe Reduction

\[
\max_{z}\;Q^{UR}(z)=\sum_i s_{i,t}z_i-lambda_c
\sum_{i<j}|\rho_{ij,t}|z_iz_j,qquad \sum_i z_i=K.
\]

QAUR đánh giá đồng thời các quan hệ cặp trong feasible cardinality subspace.
Backend hiện tại là classical cardinality-preserving surrogate cho
quantum-ready QUBO; notebook không tuyên bố quantum advantage.

### Giai đoạn 4 — So sánh reducer

\[
J_t=\frac{|\mathrm{Top}\text{-}K_A\cap\mathrm{Top}\text{-}K_{QA}|}
{|\mathrm{Top}\text{-}K_A\cup\mathrm{Top}\text{-}K_{QA}|}.
\]

Ngoài Jaccard, so sánh còn sử dụng \(Q^{UR}\), mean absolute correlation và
candidate turnover.

### Giai đoạn 5 — Portfolio Optimization dùng chung

\[
\min_x\;x^\top Q^{PO}x,\qquad
Q^{PO}=\lambda_p\widetilde\Sigma-\operatorname{diag}(\widetilde\mu),
\qquad \sum_i x_i=k_p.
\]

XY-QAOA dùng fixed-Hamming-weight initialization và constraint-preserving XY
mixer. Cùng depth, budget và shots được áp dụng cho hai reducer.

### Giai đoạn 6 — Walk-Forward Backtest

\[
\mathrm{TO}_t=\frac12\sum_i|w_{i,t}-w_{i,t^-}|,\qquad
r^{net}_{t,1}=r^{gross}_{t,1}-\mathrm{TO}_t\frac{c_{bps}}{10^4}.
\]
"""),
        core_source_cell,
        markdown(r"""
## 3. Mã điều phối confirmatory và practical experiments

Lớp confirmatory sàng lọc 43 cấu hình trên development, khóa cấu hình rồi mới
đánh giá 15 historical holdout folds. Lớp practical đánh giá 24 cấu hình nhân
với bốn market gates, tương ứng 96 phương án. Quy tắc practical là lợi nhuận
dương ở cả sáu ô giai đoạn–reducer, maximum drawdown không vượt 20% về độ lớn,
sau đó tối đa hóa worst-case Sharpe.
"""),
        orchestrator_source_cell,
        markdown(r"""
## 4. Chạy toàn bộ hệ thống

Cell này fit lại XGBoost theo từng fold, tạo cả hai candidate sets, chạy shared
portfolio pipeline, backtest, statistical tests và final-basket decision. Thời
gian chạy Colab CPU dự kiến khoảng 20–30 phút. Các bảng trung gian chỉ tồn tại
trong runtime để mô-đun hóa pipeline; mọi kết quả nghiên cứu được hiển thị trực
tiếp bên dưới và không có bước tải file.
"""),
        run_cell,
        code(r"""
import json

def read_result(name, **kwargs):
    return pd.read_csv(RESULTS / name, **kwargs)

manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
fold_manifest = read_result("walk_forward_fold_manifest.csv", parse_dates=[
    "train_start", "train_end", "validation_start", "validation_end",
    "test_start", "test_end",
])
forecast_diagnostics = read_result("forecast_diagnostics.csv", parse_dates=["decision_time"])
forecast_snapshots = read_result("forecast_snapshots.csv", parse_dates=["decision_time"])
confirmatory_definitions = read_result("confirmatory_configuration_definitions.csv")
practical_definitions = read_result("practical_configuration_definitions.csv")
confirmatory_configurations = read_result("confirmatory_configuration_results.csv")
confirmatory_tests = read_result("confirmatory_hypothesis_tests.csv")
confirmatory_seeds = read_result("confirmatory_seed_robustness.csv")
confirmatory_folds = read_result("confirmatory_best_fold_diagnostics.csv")
confirmatory_selections = read_result("confirmatory_best_selections.csv")
confirmatory_returns = read_result("confirmatory_best_returns.csv", parse_dates=["date"])
confirmatory_baselines = read_result("confirmatory_holdout_baselines.csv")
xy_audit = read_result("confirmatory_xy_qaoa_holdout_audit.csv")
practical_all = read_result("practical_configuration_period_results.csv")
practical_ranking = read_result("practical_robust_ranking.csv")
selected_periods = read_result("selected_practical_period_results.csv")
selected_returns = read_result("selected_practical_returns.csv", parse_dates=["date"])
selected_folds = read_result("selected_practical_fold_diagnostics.csv")
selected_selections = read_result("selected_practical_selections.csv")
positive_evidence = read_result("selected_practical_positive_return_evidence.csv")
practical_h4 = read_result("selected_practical_h4_by_period.csv")
practical_seeds = read_result("selected_practical_seed_robustness.csv")
market_exposures = read_result("market_gate_exposures.csv", parse_dates=["decision_time"])
final_candidates = read_result("september_2026_candidate_audit.csv", parse_dates=["decision_time"])
final_basket = read_result("september_2026_shadow_and_executable_basket.csv")

display(Markdown("### Run manifest"))
display(pd.DataFrame({"field": list(manifest), "value": [str(manifest[k]) for k in manifest]}))
display(Markdown("### Walk-forward fold manifest"))
display(fold_manifest)
"""),
        markdown(r"""
## 5. Kết quả Giai đoạn 1 — XGBoost và EWMA

Rank IC đo tương quan thứ hạng giữa forecast và lợi suất quan sát. Giá trị dương
cho thấy mô hình có xu hướng xếp tài sản sinh lợi cao lên trên, nhưng không trực
tiếp chứng minh alpha có ý nghĩa thống kê. EWMA được dùng thống nhất cho cả AUR
và QAUR nên không tạo confounding effect giữa hai nhánh.
"""),
        code(r"""
def evidence_period(fold):
    fold = int(fold)
    if fold <= 28:
        return "development_0_28"
    if fold <= 43:
        return "historical_holdout_29_43"
    if fold <= 52:
        return "observed_extension_44_52"
    return "prospective_basket_53"

forecast_view = forecast_diagnostics.copy()
forecast_view["evidence_period"] = forecast_view["fold"].map(evidence_period)
forecast_summary = forecast_view.groupby("evidence_period", sort=False).agg(
    folds=("fold", "nunique"),
    mean_rank_ic=("validation_rank_ic", "mean"),
    median_rank_ic=("validation_rank_ic", "median"),
    positive_rank_ic_folds=("validation_rank_ic", lambda x: int((x > 0).sum())),
    mean_rmse=("validation_rmse", "mean"),
    mean_universe_size=("universe_size", "mean"),
    total_runtime_seconds=("runtime_seconds", "sum"),
).reset_index()

display(Markdown("### Forecast summary theo lớp bằng chứng"))
display(forecast_summary)
display(Markdown("### Chẩn đoán của toàn bộ 54 folds"))
display(forecast_view)

observed_forecast = forecast_view[forecast_view["fold"].le(52)]
mean_ic = observed_forecast["validation_rank_ic"].mean()
positive_ic = int((observed_forecast["validation_rank_ic"] > 0).sum())
display(Markdown(f'''
**Diễn giải.** Có **{len(forecast_view)} forecast snapshots**; 53 folds đầu đã
có test window quan sát, còn fold 53 tạo quyết định prospective. Trên 53 folds
đã quan sát, validation Rank IC trung bình bằng **{mean_ic:.4f}** và dương tại
**{positive_ic}/53 folds**. Kết quả cho thấy năng lực xếp hạng dương ở mức mô
tả, nhưng độ biến thiên giữa folds và sự chồng lấn cửa sổ không cho phép coi
đây là 53 bằng chứng thống kê độc lập về alpha.
'''))
"""),
        markdown(r"""
## 6. Kết quả Giai đoạn 2 — Adaptive Universe Reduction

AUR sử dụng cùng unary score với QAUR nhưng xây Top-\(K\) theo cơ chế greedy
tuần tự. Bảng dưới đây hiển thị đầy đủ candidate set và tập tài sản được shared
portfolio pipeline lựa chọn tại từng historical holdout fold.
"""),
        code(r"""
locked_id = manifest["confirmatory_best_config"]
confirmatory_holdout_selections = confirmatory_selections[
    confirmatory_selections["fold"].between(29, 43)
].copy()

def selection_by_fold(frame, method):
    chosen = frame[frame["method"].eq(method)].copy()
    candidates = chosen.groupby("fold")["ticker"].agg(lambda x: ", ".join(x))
    portfolio = chosen[chosen["selected_downstream"]].groupby("fold")["ticker"].agg(lambda x: ", ".join(x))
    weights = chosen[chosen["selected_downstream"]].groupby("fold").apply(
        lambda x: ", ".join(f"{t}:{w:.2%}" for t, w in zip(x["ticker"], x["weight"])),
        include_groups=False,
    )
    return pd.concat([candidates.rename("Top-K"), portfolio.rename("selected_kp"), weights.rename("weights")], axis=1).reset_index()

aur_topk = selection_by_fold(confirmatory_holdout_selections, "AUR")
display(Markdown(f"**Cấu hình đã khóa:** `{locked_id}`"))
display(aur_topk)
aur_reduction = confirmatory_folds[
    confirmatory_folds["fold"].between(29, 43) & confirmatory_folds["method"].eq("AUR")
]
display(aur_reduction)
"""),
        markdown(r"""
## 7. Kết quả Giai đoạn 3 — Quantum-Assisted Universe Reduction

QAUR tối ưu joint quality–redundancy objective trong fixed-cardinality
subspace. Đây là quantum-assisted formulation, nhưng reducer đang được giải
bằng classical cardinality-preserving search; kết quả không được diễn giải là
quantum speedup.
"""),
        code(r"""
qaur_topk = selection_by_fold(confirmatory_holdout_selections, "QAUR")
display(qaur_topk)
qaur_reduction = confirmatory_folds[
    confirmatory_folds["fold"].between(29, 43) & confirmatory_folds["method"].eq("QAUR")
]
display(qaur_reduction)
"""),
        markdown(r"""
## 8. Kết quả Giai đoạn 4 — So sánh AUR và QAUR; kiểm định H1–H5

H1–H4 sử dụng paired one-sided tests trên untouched historical holdout và được
hiệu chỉnh Holm. H3 dùng non-inferiority margin 2 điểm phần trăm. H5 là điều
kiện robustness về hướng Sharpe qua các seed, không gán p-value giả tạo.
"""),
        code(r"""
holdout_fold_metrics = confirmatory_folds[confirmatory_folds["fold"].between(29, 43)]
metric_wide = holdout_fold_metrics.pivot(index="fold", columns="method", values=[
    "reduction_objective", "candidate_mean_abs_correlation", "candidate_turnover"
])
comparison_by_fold = pd.DataFrame({
    "fold": metric_wide.index,
    "QUR_AUR": metric_wide[("reduction_objective", "AUR")],
    "QUR_QAUR": metric_wide[("reduction_objective", "QAUR")],
    "correlation_AUR": metric_wide[("candidate_mean_abs_correlation", "AUR")],
    "correlation_QAUR": metric_wide[("candidate_mean_abs_correlation", "QAUR")],
    "turnover_AUR": metric_wide[("candidate_turnover", "AUR")],
    "turnover_QAUR": metric_wide[("candidate_turnover", "QAUR")],
})
comparison_by_fold["candidate_jaccard"] = holdout_fold_metrics.groupby("fold")["candidate_jaccard"].first()
display(Markdown("### So sánh theo từng holdout fold"))
display(comparison_by_fold.reset_index(drop=True))

hypothesis_text = {
    "H1_QAUR_higher_QUR_objective": "QAUR có Q^UR cao hơn AUR",
    "H2_QAUR_lower_candidate_correlation": "QAUR có candidate correlation thấp hơn AUR",
    "H3_QAUR_turnover_noninferior_margin_2pp": "QAUR không kém hơn AUR về turnover (margin 2pp)",
    "H4_QAUR_higher_mean_daily_return": "QAUR có mean daily return cao hơn AUR",
    "H5_QAUR_financial_direction_robust_across_seeds": "Ưu thế Sharpe của QAUR ổn định qua seed",
}
hypothesis_table = confirmatory_tests.copy()
hypothesis_table.insert(1, "statement", hypothesis_table["hypothesis"].map(hypothesis_text))
hypothesis_table["conclusion"] = np.where(
    hypothesis_table["supported_holm_5pct"].fillna(False),
    "Được ủng hộ", "Không được ủng hộ"
)
display(Markdown("### Bảng kiểm định giả thuyết"))
display(hypothesis_table)
display(Markdown("### Seed robustness của confirmatory configuration"))
display(confirmatory_seeds)

supported = hypothesis_table.loc[hypothesis_table["conclusion"].eq("Được ủng hộ"), "hypothesis"].str[:2].tolist()
unsupported = hypothesis_table.loc[hypothesis_table["conclusion"].ne("Được ủng hộ"), "hypothesis"].str[:2].tolist()
mean_jaccard = comparison_by_fold["candidate_jaccard"].mean()
display(Markdown(f'''
**Kết luận thống kê.** Candidate-set Jaccard trung bình bằng
**{mean_jaccard:.4f}**. Các giả thuyết được ủng hộ là **{', '.join(supported)}**;
các giả thuyết chưa được ủng hộ là **{', '.join(unsupported)}**. Vì vậy, bằng
chứng xác nhận ưu thế của QAUR nằm ở tầng universe reduction; chưa có bằng
chứng rằng ưu thế này chuyển thành mean daily return hoặc Sharpe cao hơn.
'''))
"""),
        markdown(r"""
## 9. Kết quả Giai đoạn 5 — Shared Portfolio QUBO và XY-QAOA

Feasibility rate kiểm tra tỷ lệ samples thỏa \(\sum_i x_i=k_p\). Optimality gap
so sánh nghiệm tốt nhất quan sát với exact fixed-cardinality reference. Success
probability là xác suất single-shot của nghiệm tối ưu, không phải xác suất sinh
lợi của danh mục.
"""),
        code(r"""
display(Markdown("### Toàn bộ 30 XY-QAOA holdout audit instances"))
display(xy_audit)
xy_summary = xy_audit.groupby("method").agg(
    instances=("fold", "size"),
    feasibility_rate=("feasibility_rate", "mean"),
    mean_optimality_gap=("optimality_gap", "mean"),
    mean_success_probability=("success_probability", "mean"),
    min_success_probability=("success_probability", "min"),
).reset_index()
display(Markdown("### Tổng hợp solver audit"))
display(xy_summary)
display(Markdown("### Sáu tài sản được chọn từ mỗi Top-10 tại từng holdout fold"))
selected_holdout = confirmatory_holdout_selections[
    confirmatory_holdout_selections["selected_downstream"]
].pivot_table(index="fold", columns="method", values="ticker", aggfunc=lambda x: ", ".join(x)).reset_index()
display(selected_holdout)

mean_feasible = xy_audit["feasibility_rate"].mean()
mean_gap = xy_audit["optimality_gap"].mean()
mean_success = xy_audit["success_probability"].mean()
display(Markdown(f'''
**Diễn giải.** Feasibility trung bình đạt **{mean_feasible:.2%}**, mean
optimality gap bằng **{mean_gap:.6f}** và single-shot success probability trung
bình bằng **{mean_success:.2%}**. Kết quả chứng minh formulation giữ đúng
cardinality và có thể quan sát nghiệm exact reference trên các instances nhỏ
với 1.024 shots. Nó không chứng minh quantum advantage vì statevector simulator
chạy trên phần cứng cổ điển và exact enumeration vẫn khả thi với candidate set
nhỏ.
'''))
"""),
        markdown(r"""
## 10. Kết quả Giai đoạn 6 — Walk-Forward Backtest ngoài mẫu

Các metrics được tính trên return ròng sau transaction-cost assumptions.
Maximum drawdown là mức giảm sâu nhất của tài sản từ một đỉnh lịch sử xuống đáy
tiếp theo; ví dụ −15% nghĩa là tại thời điểm tệ nhất, giá trị danh mục thấp hơn
đỉnh gần nhất 15%.
"""),
        code(r"""
confirmatory_holdout = confirmatory_configurations[
    confirmatory_configurations["config_id"].eq(locked_id)
    & confirmatory_configurations["sample"].eq("holdout")
].drop(columns=["config_id", "sample"])
confirmatory_performance = pd.concat([
    confirmatory_holdout,
    confirmatory_baselines,
], ignore_index=True)
confirmatory_performance["final_wealth"] = 1.0 + confirmatory_performance["cumulative_return"]
confirmatory_performance["calmar_zero_rf"] = (
    confirmatory_performance["annualized_return"]
    / confirmatory_performance["maximum_drawdown"].abs().replace(0, np.nan)
)
display(Markdown("### Untouched historical holdout: reducers và baselines"))
display(confirmatory_performance)

best_return_method = confirmatory_performance.loc[
    confirmatory_performance["cumulative_return"].idxmax(), "method"
]
best_drawdown_method = confirmatory_performance.loc[
    confirmatory_performance["maximum_drawdown"].idxmax(), "method"
]
h4_row = hypothesis_table[hypothesis_table["hypothesis"].str.startswith("H4")].iloc[0]
display(Markdown(f'''
**Diễn giải.** Phương pháp có cumulative return cao nhất trên holdout là
**{best_return_method}**; phương pháp có maximum drawdown nông nhất là
**{best_drawdown_method}**. Chênh lệch QAUR–AUR về mean daily return bằng
**{h4_row['estimate']:.8f}**, với Holm-adjusted p-value
**{h4_row['holm_adjusted_pvalue']:.4f}**. Do đó, khác biệt tài chính giữa hai
reducer không có ý nghĩa thống kê ở mức 5%. Candidate set tốt hơn chưa chắc tạo
lợi nhuận cao hơn vì shared Q^PO chỉ giữ k_p tài sản và classical optimizer tiếp
tục làm tương đồng risk exposures.
'''))
"""),
        markdown(r"""
## 11. Practical method design, ràng buộc và độ bền lợi nhuận

Phần này là post-hoc method design. Mục tiêu là tìm cấu hình có hiệu quả kinh tế
ổn định trong không gian đã khai báo, không thay thế confirmatory evidence.
"""),
        code(r"""
selected_id = manifest["practical_best_config"]["config_id"]
selected_gate = int(manifest["practical_best_market_gate_lookback"])
selected_config = pd.DataFrame([manifest["practical_best_config"]])
selected_periods_view = selected_periods.copy()
selected_periods_view["final_wealth"] = 1.0 + selected_periods_view["cumulative_return"]
selected_periods_view["calmar_zero_rf"] = (
    selected_periods_view["annualized_return"]
    / selected_periods_view["maximum_drawdown"].abs().replace(0, np.nan)
)

selected_exposures = market_exposures[
    market_exposures["config_id"].eq(selected_id)
    & market_exposures["lookback"].eq(selected_gate)
].sort_values("fold")
exposure_days = selected_returns.groupby("method").agg(
    observations=("date", "size"),
    invested_days=("market_gate_exposure", lambda x: int((x > 0).sum())),
    cash_days=("market_gate_exposure", lambda x: int((x == 0).sum())),
).reset_index()
transaction_cost_bps = float(manifest["practical_best_config"]["transaction_cost_bps"])
cost_summary = selected_folds.groupby("method").agg(
    total_one_way_turnover=("portfolio_turnover", "sum"),
    mean_rebalance_turnover=("portfolio_turnover", "mean"),
).reset_index()
cost_summary["modeled_rebalance_cost_rate"] = (
    cost_summary["total_one_way_turnover"] * transaction_cost_bps / 10000.0
)

display(Markdown("### Cấu hình practical được lựa chọn"))
display(selected_config.T.rename(columns={0: "value"}))
display(Markdown(f"**Common market gate:** {selected_gate} phiên"))
display(Markdown("### Kết quả theo ba giai đoạn và hai reducer"))
display(selected_periods_view)
display(Markdown("### Số ngày có exposure và giữ tiền mặt"))
display(exposure_days)
display(Markdown("### Turnover và chi phí mô phỏng"))
display(cost_summary)
display(Markdown("### Bằng chứng lợi nhuận dương"))
display(positive_evidence)
display(Markdown("### H4 exploratory theo từng giai đoạn"))
display(practical_h4)
display(Markdown("### Practical seed robustness"))
display(practical_seeds)
display(Markdown("### Top 10 trong xếp hạng 96 phương án"))
display(practical_ranking.head(10))

economic_positive = int(positive_evidence["positive_economically"].sum())
statistical_positive = int(positive_evidence["positive_mean_supported_holm_5pct"].sum())
worst_sharpe = selected_periods["sharpe_zero_rf"].min()
worst_drawdown = selected_periods["maximum_drawdown"].min()
display(Markdown(f'''
**Diễn giải.** Cấu hình `{selected_id} + gate {selected_gate}` có lợi nhuận
quan sát dương tại **{economic_positive}/6** ô, worst-case Sharpe bằng
**{worst_sharpe:.4f}** và maximum drawdown tệ nhất bằng
**{worst_drawdown:.2%}**. Tuy nhiên, chỉ **{statistical_positive}/6** ô có mean
daily return dương sau Holm ở mức 5%. Vì cấu hình được chọn sau khi quan sát 96
phương án, kết quả này là post-hoc và chỉ đủ làm cơ sở cho paper trading.
'''))
"""),
        markdown(r"""
## 12. Final Stock Basket and Execution Status

Top-K cuối cùng và rổ bốn cổ phiếu bên dưới được pipeline tính trực tiếp từ
fold prospective; không có mã cổ phiếu hoặc tỷ trọng nào được nhập thủ công.
Signal/liquidity/risk ranks chỉ giải thích vị trí tương đối trong candidate set,
không phải khuyến nghị mua bán.
"""),
        code(r"""
final_candidates = final_candidates.copy()
final_candidates["signal_rank_in_candidate"] = final_candidates.groupby("method")["xgb_signal"].rank(pct=True)
final_candidates["liquidity_rank_in_candidate"] = final_candidates.groupby("method")["liquidity_20d"].rank(pct=True)
final_candidates["low_risk_rank_in_candidate"] = final_candidates.groupby("method")["volatility_20d"].rank(pct=True, ascending=False)
final_candidates = final_candidates.sort_values(["method", "selected_downstream", "shadow_weight"], ascending=[True, False, False])

display(Markdown("### Top-K_A và Top-K_QA, kèm quyết định shared portfolio pipeline"))
display(final_candidates[[
    "decision_time", "method", "ticker", "selected_downstream", "xgb_signal",
    "momentum_signal", "liquidity_20d", "volatility_20d",
    "signal_rank_in_candidate", "liquidity_rank_in_candidate",
    "low_risk_rank_in_candidate", "shadow_weight", "executable_weight",
]])

final_selected = final_candidates[final_candidates["selected_downstream"]].copy()
display(Markdown("### Rổ cổ phiếu cuối cùng"))
display(final_selected[[
    "method", "ticker", "shadow_weight", "executable_weight", "cash_weight",
    "market_growth", "market_gate_exposure",
]])

aur_set = set(final_selected.loc[final_selected["method"].eq("AUR"), "ticker"])
qaur_set = set(final_selected.loc[final_selected["method"].eq("QAUR"), "ticker"])
same_final_set = aur_set == qaur_set
gate_exposure = float(final_selected["market_gate_exposure"].iloc[0])
cash_weight = float(final_selected["cash_weight"].iloc[0])
market_growth = float(final_selected["market_growth"].iloc[0])

representative = final_selected[final_selected["method"].eq("AUR")]
asset_lines = []
for row in representative.itertuples():
    asset_lines.append(
        f"- **{row.ticker}** — shadow weight {row.shadow_weight:.2%}; "
        f"signal rank {row.signal_rank_in_candidate:.0%}, liquidity rank "
        f"{row.liquidity_rank_in_candidate:.0%}, low-risk rank "
        f"{row.low_risk_rank_in_candidate:.0%} trong Top-K."
    )

status = "được phép có exposure" if gate_exposure > 0 else "bị market gate chặn"
convergence = "hội tụ về cùng một tập tài sản" if same_final_set else "tạo hai tập tài sản cuối khác nhau"
display(Markdown(
    f'''**Giải thích rổ cuối.** AUR và QAUR **{convergence}**. Market proxy 30
phiên tăng trưởng **{market_growth:.2%}**, vì vậy quyết định hiện tại **{status}**.
Cash weight bằng **{cash_weight:.2%}**. Các shadow weights mô tả danh mục sẽ
được theo dõi khi risk-on; executable weights mới là quyết định thực thi.

''' + "\n".join(asset_lines) + "\n\nRổ này phục vụ paper trading, không phải khuyến nghị đầu tư."
))
"""),
        markdown(r"""
## 13. Biểu đồ kết quả

Các biểu đồ dưới đây được tạo từ object của lần chạy hiện tại và hiển thị trực
tiếp trong notebook.
"""),
        code(r"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 2, figsize=(16, 22))

# Forecast Rank IC
axes[0, 0].plot(forecast_diagnostics["fold"], forecast_diagnostics["validation_rank_ic"], marker="o", ms=3)
axes[0, 0].axhline(0, color="black", lw=1, ls="--")
axes[0, 0].set(title="Validation Rank IC by fold", xlabel="Fold", ylabel="Rank IC")

# Candidate similarity
axes[0, 1].plot(comparison_by_fold["fold"], comparison_by_fold["candidate_jaccard"], marker="o", color="#6a1b9a")
axes[0, 1].set(title="AUR–QAUR candidate-set similarity", xlabel="Holdout fold", ylabel="Jaccard")
axes[0, 1].set_ylim(0, 1.05)

# QUR and correlation
axes[1, 0].plot(comparison_by_fold["fold"], comparison_by_fold["QUR_AUR"], label="AUR")
axes[1, 0].plot(comparison_by_fold["fold"], comparison_by_fold["QUR_QAUR"], label="QAUR")
axes[1, 0].set(title="Quality–redundancy objective", xlabel="Holdout fold", ylabel="Q^UR")
axes[1, 0].legend()
axes[1, 1].plot(comparison_by_fold["fold"], comparison_by_fold["correlation_AUR"], label="AUR")
axes[1, 1].plot(comparison_by_fold["fold"], comparison_by_fold["correlation_QAUR"], label="QAUR")
axes[1, 1].set(title="Candidate mean absolute correlation", xlabel="Holdout fold", ylabel="Mean |correlation|")
axes[1, 1].legend()

# Practical cumulative wealth and drawdown
for method, group in selected_returns.groupby("method"):
    group = group.sort_values("date")
    wealth = (1 + group["return"]).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    axes[2, 0].plot(group["date"], wealth, label=method)
    axes[2, 1].plot(group["date"], drawdown, label=method)
axes[2, 0].set(title="Practical cumulative wealth", xlabel="Date", ylabel="Growth of 1")
axes[2, 1].set(title="Practical drawdown", xlabel="Date", ylabel="Drawdown")
axes[2, 0].legend(); axes[2, 1].legend()

# Holdout risk-return comparison
for row in confirmatory_performance.itertuples():
    axes[3, 0].scatter(row.annualized_volatility, row.annualized_return, s=70)
    axes[3, 0].annotate(row.method, (row.annualized_volatility, row.annualized_return), xytext=(4, 4), textcoords="offset points")
axes[3, 0].set(title="Historical holdout risk–return", xlabel="Annualized volatility", ylabel="Annualized return")

# Final basket and market gate
basket_plot = representative.set_index("ticker")["shadow_weight"].sort_values(ascending=False)
axes[3, 1].bar(basket_plot.index, basket_plot.values, color="#ef6c00")
axes[3, 1].set(title=f"Final shadow basket; executable exposure={gate_exposure:.0%}", xlabel="Ticker", ylabel="Weight")

for ax in axes.flat:
    ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()
figure_created = True

display(Markdown('''
**Cách đọc.** Hai biểu đồ đầu mô tả forecast và mức trùng lặp reducer; hai biểu
đồ tiếp theo tách quality objective khỏi correlation redundancy; equity curve
và drawdown phản ánh hiệu quả kinh tế sau chi phí; risk–return plot đặt AUR và
QAUR cạnh các baseline; biểu đồ cuối là shadow weights, không phải executable
weights khi market gate đóng.
'''))
"""),
        markdown(r"""
## 14. Tổng hợp học thuật và mức độ sẵn sàng ứng dụng

Phần dưới được sinh từ kết quả của lần chạy hiện tại, không chèn sẵn số liệu.
"""),
        code(r"""
best_practical = selected_periods_view.loc[selected_periods_view["sharpe_zero_rf"].idxmax()]
supported_labels = ", ".join(supported) if supported else "không có"
unsupported_labels = ", ".join(unsupported) if unsupported else "không có"

display(Markdown(f'''
### Kết luận kết quả

Bộ dữ liệu gồm **{len(raw_data):,} bản ghi**, **{price_raw['ticker'].nunique()} mã
cổ phiếu**, bao phủ từ **{price_raw['date'].min().date()}** đến
**{price_raw['date'].max().date()}**. Pipeline tạo **{len(forecast_diagnostics)}
walk-forward snapshots**. Năng lực xếp hạng của XGBoost mang dấu dương ở mức mô
tả nhưng biến động theo fold; EWMA cung cấp cùng risk input cho hai reducer.

Ở tầng universe reduction, các giả thuyết được ủng hộ là
**{supported_labels}**; các giả thuyết chưa được ủng hộ là
**{unsupported_labels}**. Điều này xác định phạm vi ưu thế của QAUR ở candidate
quality, correlation và turnover, nhưng không cho phép suy ra ưu thế lợi nhuận.

XY-QAOA đạt feasibility **{mean_feasible:.2%}** và mean optimality gap
**{mean_gap:.6f}** trên {len(xy_audit)} instances. Đây là bằng chứng về tính
đúng đắn của feasible-subspace implementation trên simulator, không phải bằng
chứng quantum advantage.

Cấu hình practical tốt nhất là **{selected_id} + gate {selected_gate}**. Lợi
nhuận quan sát dương tại **{economic_positive}/6** ô nhưng lợi nhuận dương có ý
nghĩa thống kê sau Holm chỉ đạt **{statistical_positive}/6** ô. Maximum drawdown
tệ nhất là **{worst_drawdown:.2%}**. Do đó, kết quả kinh tế tương đối ổn định
trong mẫu đã xét nhưng bằng chứng thống kê vẫn chưa đủ mạnh.

Rổ prospective gồm **{', '.join(sorted(aur_set | qaur_set))}**. Market gate
hiện đặt exposure bằng **{gate_exposure:.0%}**, nên executable portfolio là
**{cash_weight:.0%} tiền mặt**; shadow basket chỉ được theo dõi trong paper
trading.

### Mức độ áp dụng

Framework hiện phù hợp cho **paper trading không sử dụng vốn thật**. Chưa nên
triển khai live capital vì practical selection là post-hoc, positive-return
tests chưa đạt ý nghĩa sau multiple-testing correction, dữ liệu 2026 còn mang
tính provisional và chưa có prospective record đủ dài với slippage, market
impact cùng sự cố dữ liệu thực tế. Bước tiếp theo là khóa toàn bộ tham số, chạy
forward 6–12 tháng, công bố cả kỳ giữ tiền mặt và tiếp tục so sánh với
Full-Universe equal-weight cùng VNAllShare TRI.
'''))
"""),
        markdown("## 15. Research audit"),
        code(r"""
audit_rows = []
def audit(name, condition, evidence):
    audit_rows.append({"check": name, "passed": bool(condition), "evidence": str(evidence)})

folds_chronological = (
    (fold_manifest["train_end"] <= fold_manifest["validation_start"]).all()
    and (fold_manifest["validation_end"] <= fold_manifest["test_start"]).all()
)
locked_definition = confirmatory_definitions[confirmatory_definitions["config_id"].eq(locked_id)].iloc[0]
practical_definition = practical_definitions[practical_definitions["config_id"].eq(selected_id)].iloc[0]
confirm_candidate_counts = confirmatory_selections.groupby(["fold", "method"])["ticker"].nunique()
confirm_selected_counts = confirmatory_selections[confirmatory_selections["selected_downstream"]].groupby(["fold", "method"])["ticker"].nunique()
practical_candidate_counts = selected_selections.groupby(["fold", "method"])["ticker"].nunique()
practical_selected = selected_selections[selected_selections["selected_downstream"]]
practical_selected_counts = practical_selected.groupby(["fold", "method"])["ticker"].nunique()
practical_weight_sums = practical_selected.groupby(["fold", "method"])["weight"].sum()

audit("Dataset SHA-256", digest == manifest["dataset_sha256"], digest)
audit("No duplicate price keys", duplicate_price_keys == 0, duplicate_price_keys)
audit("Chronological train-validation-test folds", folds_chronological, len(fold_manifest))
audit("All 54 forecast snapshots present", len(forecast_diagnostics) == 54, len(forecast_diagnostics))
audit("AUR/QAUR share the same fold set", confirmatory_selections.groupby("fold")["method"].nunique().eq(2).all(), "two reducers per fold")
audit("Confirmatory Top-K cardinality", confirm_candidate_counts.eq(int(locked_definition["candidate_size"])).all(), confirm_candidate_counts.unique())
audit("Confirmatory portfolio cardinality", confirm_selected_counts.eq(int(locked_definition["portfolio_cardinality"])).all(), confirm_selected_counts.unique())
audit("Practical Top-K cardinality", practical_candidate_counts.eq(int(practical_definition["candidate_size"])).all(), practical_candidate_counts.unique())
audit("Practical portfolio cardinality", practical_selected_counts.eq(int(practical_definition["portfolio_cardinality"])).all(), practical_selected_counts.unique())
audit("Practical weights sum to one before common gate", np.allclose(practical_weight_sums, 1.0, atol=1e-7), practical_weight_sums.min())
audit("Practical weight lower bound", practical_selected["weight"].ge(float(practical_definition["weight_lower"]) - 1e-8).all(), practical_selected["weight"].min())
audit("Practical weight upper bound", practical_selected["weight"].le(float(practical_definition["weight_upper"]) + 1e-8).all(), practical_selected["weight"].max())
audit("Transaction-cost inputs are non-negative", selected_folds["portfolio_turnover"].ge(0).all() and transaction_cost_bps >= 0, transaction_cost_bps)
audit("XY-QAOA fixed-cardinality feasibility", xy_audit["feasibility_rate"].eq(1.0).all(), xy_audit["feasibility_rate"].mean())
audit("Five hypotheses reported", set(confirmatory_tests["hypothesis"]) == set(hypothesis_text), len(confirmatory_tests))
audit("Holm p-values available for H1-H4", confirmatory_tests["holm_adjusted_pvalue"].notna().sum() == 4, confirmatory_tests["holm_adjusted_pvalue"].notna().sum())
audit("Final shadow weights sum to one per reducer", final_selected.groupby("method")["shadow_weight"].sum().between(0.999999, 1.000001).all(), final_selected.groupby("method")["shadow_weight"].sum().to_dict())
expected_executable_sum = gate_exposure
audit("Final executable weights obey common gate", np.allclose(final_selected.groupby("method")["executable_weight"].sum(), expected_executable_sum, atol=1e-8), expected_executable_sum)
audit("No live-capital or quantum-advantage claim", not manifest["live_capital_authorized"] and not manifest["quantum_advantage_claimed"], "both false")
audit("All principal figures rendered", figure_created, figure_created)
audit("No NaN in principal performance metrics", not selected_periods[["cumulative_return", "annualized_return", "annualized_volatility", "sharpe_zero_rf", "maximum_drawdown"]].isna().any().any(), "checked")

audit_table = pd.DataFrame(audit_rows)
display(audit_table)
display(Markdown(f"**Passed:** {int(audit_table['passed'].sum())}/{len(audit_table)} checks."))
"""),
        code(r"""
assert audit_table["passed"].all(), audit_table.loc[~audit_table["passed"]].to_dict("records")
print("RESEARCH_AUDIT_FINAL_OK")
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
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT.name)

    # Submission-light release: code and every research output remain in the
    # notebook, while the hash-locked dataset is fetched automatically from a
    # public read-only Drive object.  This avoids a 6 MB notebook document that
    # is unnecessarily slow to render in Colab while preserving one-click
    # reproducibility and the exact same data checksum.
    light_notebook = copy.deepcopy(notebook)
    light_notebook["metadata"]["colab"]["name"] = LIGHT_OUTPUT.name
    light_notebook["cells"][0] = markdown(r"""
# AUR–QAUR Portfolio Research — Full Executable Submission Colab

Notebook này là **bằng chứng code và thực nghiệm có thể tái lập** của toàn bộ
hệ thống. Toàn bộ source code được viết trực tiếp trong notebook, không clone
GitHub, không mount Drive và không yêu cầu upload thủ công. Bộ dữ liệu thật được
nạp tự động từ một Drive object chỉ-đọc, sau đó bắt buộc khớp SHA-256 trước khi
được sử dụng. Chỉ cần chọn **Runtime → Run all** trên Google Colab CPU.

Mọi bảng kết quả, biểu đồ, kiểm định H1–H5 và rổ cổ phiếu cuối cùng được sinh
trực tiếp từ code của lần chạy hiện tại; notebook không tải ZIP/CSV/HTML kết quả.
""")
    light_notebook["cells"][5] = code(f'''
!pip -q install "gdown>=5.2"
from pathlib import Path
import hashlib
import zipfile
import gdown

# Use the same runtime root as the two subsequent %%writefile cells.  Colab's
# writefile magic does not create parent directories, so this directory must
# exist before the embedded research engine is materialized.
WORKDIR = Path("/content/aur_qaur_29_8")
WORKDIR.mkdir(parents=True, exist_ok=True)
DATA_ZIP = WORKDIR / "data_29_8.zip"
gdown.download(id="{DATA_DRIVE_ID}", output=str(DATA_ZIP), quiet=False)

EXPECTED_ZIP_SHA256 = "0025fd746a9060755f2c05b4cae317bb1f3fa4f517f7308529d0e1d8ab3bca0c"
zip_digest = hashlib.sha256(DATA_ZIP.read_bytes()).hexdigest()
assert zip_digest == EXPECTED_ZIP_SHA256, f"Dataset ZIP SHA256 mismatch: {{zip_digest}}"
with zipfile.ZipFile(DATA_ZIP) as archive:
    archive.extractall(WORKDIR)

candidates = list(WORKDIR.rglob("data_29_8.csv"))
if len(candidates) != 1:
    raise RuntimeError(f"Cần đúng một data_29_8.csv, tìm thấy {{len(candidates)}}")
DATASET = candidates[0]
EXPECTED_SHA256 = "b0a16d9f8c31a2a5d4e1ba8f00d49b50f112f149d4fae23b3529df085a45ccb2"
digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
assert digest == EXPECTED_SHA256, f"Dataset CSV SHA256 mismatch: {{digest}}"
print("Dataset:", DATASET)
print("Bytes:", DATASET.stat().st_size)
print("ZIP SHA256 verified:", zip_digest)
print("CSV SHA256 verified:", digest)
''')
    LIGHT_OUTPUT.write_text(
        json.dumps(light_notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(LIGHT_OUTPUT.name)


if __name__ == "__main__":
    main()
