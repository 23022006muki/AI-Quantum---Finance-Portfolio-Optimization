from __future__ import annotations

"""Generate standalone Google Colab releases for the complete AUR/QAUR system."""

import base64
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab" / "AUR_QAUR_Practical_Optimization_29_8_Full_Colab.ipynb"
WEB_OUTPUT = ROOT / "colab" / "AUR_QAUR_Standalone_Full_Web_Colab_30_8.ipynb"
CORE = ROOT / "scripts" / "run_constraint_strategy_search.py"
ORCHESTRATOR = ROOT / "scripts" / "run_colab_29_8_complete.py"
DATA_ARCHIVE = ROOT / "data 29_8" / "data_29_8.zip"


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
    core_source = CORE.read_text(encoding="utf-8")
    orchestrator_source = ORCHESTRATOR.read_text(encoding="utf-8")
    cells = [
        markdown(r"""
# AUR–QAUR Practical Portfolio Optimization — Full Web Colab ngày 30/8

Notebook này chứa **toàn bộ code hệ thống**, không clone hoặc tải code từ
GitHub. Người dùng chỉ upload `data_29_8.zip` hoặc `data_29_8.csv` rồi chạy lần
lượt các cell.

Notebook tách hai lớp bằng chứng:

1. **Confirmatory historical:** chọn cấu hình trên folds 0–28 và kiểm định trên
   untouched historical holdout folds 29–43.
2. **Practical method design:** chạy grid ràng buộc, allocation và common market
   gate trên toàn bộ dữ liệu đã quan sát đến 29/8/2026, sau đó tạo rổ paper
   trading. Lớp này là post-hoc và không được gọi là prospective proof.

Đối tượng so sánh là AUR và QAUR. Cả hai dùng chung portfolio QUBO, shared
fixed-cardinality solver, weight allocation, transaction cost, market gate và
walk-forward backtest. QAUR hiện là classical cardinality-preserving surrogate;
notebook không tuyên bố quantum advantage hoặc cho phép dùng vốn thật.
"""),
        markdown("## 1. Cài thư viện"),
        code(r"""
!pip -q install "numpy>=2.0" "pandas>=2.2" "scipy>=1.13" "scikit-learn>=1.5" "xgboost>=2.1" "matplotlib>=3.8" "pyarrow>=16" "tabulate>=0.9"
"""),
        markdown("## 2. Upload bộ dữ liệu ngày 29/8"),
        code(r"""
from google.colab import files
from pathlib import Path
import hashlib
import zipfile

uploaded = files.upload()
if not uploaded:
    raise RuntimeError("Hãy upload data_29_8.zip hoặc data_29_8.csv")

WORKDIR = Path("/content/aur_qaur_29_8")
WORKDIR.mkdir(parents=True, exist_ok=True)
uploaded_path = Path(next(iter(uploaded)))
if uploaded_path.suffix.lower() == ".zip":
    with zipfile.ZipFile(uploaded_path) as archive:
        archive.extractall(WORKDIR)
    candidates = list(WORKDIR.rglob("data_29_8.csv"))
else:
    target = WORKDIR / "data_29_8.csv"
    target.write_bytes(uploaded_path.read_bytes())
    candidates = [target]
if len(candidates) != 1:
    raise RuntimeError(f"Cần đúng một data_29_8.csv, tìm thấy {len(candidates)}")
DATASET = candidates[0]
EXPECTED_SHA256 = "b0a16d9f8c31a2a5d4e1ba8f00d49b50f112f149d4fae23b3529df085a45ccb2"
digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
assert digest == EXPECTED_SHA256, (
    "Sai bộ dữ liệu. Hãy upload đúng data_29_8.zip/data_29_8.csv; "
    f"SHA256 nhận được: {digest}"
)
print("Dataset:", DATASET)
print("Bytes:", DATASET.stat().st_size)
print("SHA256 verified:", digest)
"""),
        markdown(r"""
## 3. Mã nguồn phương pháp và experimental engine

Cell dưới đây là toàn bộ implementation: feature engineering, XGBoost,
point-in-time eligibility, AUR, QAUR, fixed-cardinality portfolio QUBO,
bounded-simplex weight allocation, walk-forward folds, transaction costs,
financial metrics, hypothesis tests và XY-QAOA statevector audit.
"""),
        code("%%writefile /content/aur_qaur_29_8/run_constraint_strategy_search.py\n" + core_source),
        markdown(r"""
## 4. Mã nguồn điều phối thí nghiệm ngày 29/8

Experimental grid gồm 43 cấu hình confirmatory và 24 cấu hình thực tiễn. Mỗi
cấu hình thực tiễn được thử với common market gate 0, 20, 30 và 40 phiên. Tiêu
chí chọn trước là: lợi nhuận dương ở cả ba giai đoạn và cả hai reducers,
maximum drawdown không thấp hơn −20%, sau đó tối đa hóa worst-case Sharpe.
"""),
        code("%%writefile /content/aur_qaur_29_8/run_colab_29_8_complete.py\n" + orchestrator_source),
        markdown("## 5. Chạy toàn bộ hệ thống"),
        code(r"""
import subprocess, sys, time

RESULTS = WORKDIR / "results_29_8"
RESULTS.mkdir(exist_ok=True)
started = time.time()
command = [
    sys.executable,
    str(WORKDIR / "run_colab_29_8_complete.py"),
    "--dataset", str(DATASET),
    "--output", str(RESULTS),
]
print("Running:", " ".join(command))
subprocess.run(command, check=True, cwd=WORKDIR)
print(f"Hoàn tất sau {(time.time()-started)/60:.1f} phút")
"""),
        markdown("## 6. Xem cấu hình tối ưu, bằng chứng H1–H5 và rổ cổ phiếu"),
        code(r"""
import json
import pandas as pd
from IPython.display import display, Markdown

manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
display(Markdown("### Manifest và nhãn bằng chứng"))
display(pd.DataFrame({"field": manifest.keys(), "value": [str(v) for v in manifest.values()]}))

display(Markdown("### Chẩn đoán XGBoost theo walk-forward fold"))
forecast_diagnostics = pd.read_csv(RESULTS / "forecast_diagnostics.csv")
forecast_summary = pd.DataFrame([{
    "folds": len(forecast_diagnostics),
    "mean_validation_rank_ic": forecast_diagnostics["validation_rank_ic"].mean(),
    "median_validation_rank_ic": forecast_diagnostics["validation_rank_ic"].median(),
    "positive_rank_ic_folds": int((forecast_diagnostics["validation_rank_ic"] > 0).sum()),
    "mean_validation_rmse": forecast_diagnostics["validation_rmse"].mean(),
    "mean_universe_size": forecast_diagnostics["universe_size"].mean(),
}])
display(forecast_summary)
display(forecast_diagnostics)

display(Markdown("### Kết quả 43 cấu hình confirmatory"))
confirmatory_configurations = pd.read_csv(
    RESULTS / "confirmatory_configuration_results.csv"
)
display(confirmatory_configurations)

display(Markdown("### Confirmatory hypothesis tests"))
confirmatory_tests = pd.read_csv(RESULTS / "confirmatory_hypothesis_tests.csv")
display(confirmatory_tests)

display(Markdown("### Practical period results"))
period_results = pd.read_csv(RESULTS / "selected_practical_period_results.csv")
display(period_results)

display(Markdown("### Xếp hạng đầy đủ 96 phương án practical"))
practical_ranking = pd.read_csv(RESULTS / "practical_robust_ranking.csv")
display(practical_ranking)

display(Markdown("### Exploratory H4 by period"))
display(pd.read_csv(RESULTS / "selected_practical_h4_by_period.csv"))

display(Markdown("### Lợi nhuận dương và ý nghĩa thống kê"))
positive_evidence = pd.read_csv(
    RESULTS / "selected_practical_positive_return_evidence.csv"
)
display(positive_evidence)

display(Markdown("### QAUR seed robustness"))
practical_seed = pd.read_csv(RESULTS / "selected_practical_seed_robustness.csv")
confirmatory_seed = pd.read_csv(RESULTS / "confirmatory_seed_robustness.csv")
display(Markdown("**Confirmatory seeds**"))
display(confirmatory_seed)
display(Markdown("**Practical seeds**"))
display(practical_seed)

display(Markdown("### XY-QAOA audit trên holdout lịch sử"))
xy_audit = pd.read_csv(RESULTS / "confirmatory_xy_qaoa_holdout_audit.csv")
display(xy_audit)
display(xy_audit.groupby("method")[[
    "feasibility_rate", "optimality_gap", "success_probability"
]].agg(["mean", "min", "max"]))

display(Markdown("### Rổ shadow và tỷ trọng thực thi cho tháng 9/2026"))
basket = pd.read_csv(RESULTS / "september_2026_shadow_and_executable_basket.csv")
display(basket[["method", "ticker", "shadow_weight", "executable_weight", "cash_weight", "market_growth"]])

display(Markdown((RESULTS / "FINAL_RESULTS_29_8_VI.md").read_text(encoding="utf-8")))

display(Markdown("### Danh mục toàn bộ artifacts đã xuất"))
artifact_inventory = pd.DataFrame([
    {"file": path.name, "bytes": path.stat().st_size}
    for path in sorted(RESULTS.iterdir()) if path.is_file()
])
display(artifact_inventory)
"""),
        markdown("## 7. Biểu đồ kết quả và rổ shadow"),
        code(r"""
import matplotlib.pyplot as plt
import numpy as np

selected_daily = pd.read_csv(
    RESULTS / "selected_practical_returns.csv", parse_dates=["date"]
).sort_values(["method", "date"])
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for method, group in selected_daily.groupby("method"):
    wealth = (1.0 + group["return"]).cumprod()
    axes[0].plot(group["date"], wealth, label=method, linewidth=1.6)
axes[0].set_title("Cumulative wealth — selected practical protocol")
axes[0].set_ylabel("Growth of 1 unit")
axes[0].legend()
axes[0].grid(alpha=.25)

bar = period_results.pivot(
    index="sample", columns="method", values="cumulative_return"
)
bar.plot(kind="bar", ax=axes[1], color=["#2e7d32", "#1565c0"])
axes[1].axhline(0, color="black", linewidth=.8)
axes[1].set_title("Cumulative return by evidence period")
axes[1].set_ylabel("Return")
axes[1].tick_params(axis="x", rotation=20)
axes[1].grid(axis="y", alpha=.25)

shadow_plot = basket[basket["method"].eq("AUR")].set_index("ticker")["shadow_weight"]
shadow_plot.plot(kind="bar", ax=axes[2], color="#ef6c00")
axes[2].set_title("September 2026 shadow basket")
axes[2].set_ylabel("Target weight")
axes[2].tick_params(axis="x", rotation=0)
axes[2].grid(axis="y", alpha=.25)

fig.tight_layout()
figure_path = RESULTS / "selected_practical_results_and_basket.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", figure_path)
"""),
        markdown("## 8. Fail-fast audit"),
        code(r"""
required = [
    "run_manifest.json",
    "confirmatory_hypothesis_tests.csv",
    "confirmatory_seed_robustness.csv",
    "confirmatory_xy_qaoa_holdout_audit.csv",
    "practical_robust_ranking.csv",
    "selected_practical_period_results.csv",
    "selected_practical_h4_by_period.csv",
    "selected_practical_positive_return_evidence.csv",
    "selected_practical_seed_robustness.csv",
    "september_2026_shadow_and_executable_basket.csv",
    "FINAL_RESULTS_29_8_VI.md",
]
missing = [name for name in required if not (RESULTS / name).exists()]
assert not missing, f"Thiếu artifacts: {missing}"
assert set(confirmatory_tests["hypothesis"]) == {
    "H1_QAUR_higher_QUR_objective",
    "H2_QAUR_lower_candidate_correlation",
    "H3_QAUR_turnover_noninferior_margin_2pp",
    "H4_QAUR_higher_mean_daily_return",
    "H5_QAUR_financial_direction_robust_across_seeds",
}
assert confirmatory_tests["holm_adjusted_pvalue"].notna().sum() == 4
assert confirmatory_tests["supported_holm_5pct"].sum() == 3
assert len(forecast_diagnostics) == 54
assert len(confirmatory_configurations) == 43 * 2 * 2
assert len(practical_ranking) == 24 * 4
assert period_results.groupby(["sample", "method"]).size().size == 6
assert (period_results["cumulative_return"] > 0).all(), "Cấu hình được chọn không dương ở mọi cell"
assert positive_evidence["positive_economically"].all()
assert period_results["maximum_drawdown"].min() >= -0.20
assert basket.groupby("method")["shadow_weight"].sum().between(0.999999, 1.000001).all()
assert len(xy_audit) == 30
assert xy_audit["feasibility_rate"].eq(1.0).all()
assert not manifest["live_capital_authorized"]
assert not manifest["quantum_advantage_claimed"]
print("RESEARCH_AUDIT_29_8_OK")
"""),
        markdown("## 9. Nén và tải toàn bộ kết quả"),
        code(r"""
import shutil
archive = shutil.make_archive("/content/AUR_QAUR_RESULTS_29_8", "zip", RESULTS)
print("Đã tạo:", archive)
files.download(archive)
"""),
        markdown(r"""
## Quy tắc diễn giải

- H1–H4 trong file confirmatory sử dụng untouched historical holdout.
- H4 của practical method được báo cáo riêng theo từng giai đoạn và mang nhãn
  exploratory/post-hoc.
- Lợi nhuận dương trong backtest không phải cam kết lợi nhuận tương lai.
- Rổ tháng 9 là paper-trading target. Nếu common market gate ở trạng thái cash,
  shadow weights vẫn được ghi nhận nhưng executable weights bằng 0.
- Feasibility hoặc optimality trên simulator không chứng minh quantum advantage.
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
    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT.name)

    # The web release is deliberately self-contained. Embedding the compressed,
    # hash-locked dataset avoids a fragile browser upload widget and makes
    # "Runtime -> Run all" reproducible without GitHub, Drive mounting, or a
    # second local file.
    archive_bytes = DATA_ARCHIVE.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    encoded = base64.b64encode(archive_bytes).decode("ascii")
    encoded_literal = "\n".join(
        f'    "{encoded[offset:offset + 100]}"'
        for offset in range(0, len(encoded), 100)
    )
    embedded_data_cell = f'''from google.colab import files
from pathlib import Path
from io import BytesIO
import base64
import hashlib
import zipfile

WORKDIR = Path("/content/aur_qaur_29_8")
WORKDIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_DATA_ZIP_B64 = (
{encoded_literal}
)
archive_bytes = base64.b64decode(EMBEDDED_DATA_ZIP_B64)
assert hashlib.sha256(archive_bytes).hexdigest() == "{archive_sha256}"
with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
    archive.extractall(WORKDIR)

candidates = list(WORKDIR.rglob("data_29_8.csv"))
if len(candidates) != 1:
    raise RuntimeError(f"Cần đúng một data_29_8.csv, tìm thấy {{len(candidates)}}")
DATASET = candidates[0]
EXPECTED_SHA256 = "b0a16d9f8c31a2a5d4e1ba8f00d49b50f112f149d4fae23b3529df085a45ccb2"
digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
assert digest == EXPECTED_SHA256, f"Embedded dataset SHA256 mismatch: {{digest}}"
print("Standalone dataset:", DATASET)
print("Bytes:", DATASET.stat().st_size)
print("SHA256 verified:", digest)
'''
    web_notebook = copy.deepcopy(notebook)
    web_notebook["metadata"]["colab"]["name"] = WEB_OUTPUT.name
    web_notebook["cells"][0] = markdown(r'''
# AUR–QAUR Practical Portfolio Optimization — Standalone Full Web Colab ngày 30/8

Notebook chứa **toàn bộ code và bộ dữ liệu nén đã khóa SHA-256**. Không clone
GitHub, không mount Drive và không cần upload thêm tệp; chỉ chọn
**Runtime → Run all**.

Notebook tách hai lớp bằng chứng:

1. **Confirmatory historical:** chọn cấu hình trên folds 0–28 và kiểm định trên
   untouched historical holdout folds 29–43.
2. **Practical method design:** chạy grid ràng buộc, allocation và common market
   gate trên toàn bộ dữ liệu đã quan sát đến 29/8/2026, sau đó tạo rổ paper
   trading. Lớp này là post-hoc và không được gọi là prospective proof.

Đối tượng so sánh là AUR và QAUR. Cả hai dùng chung portfolio QUBO, shared
fixed-cardinality solver, weight allocation, transaction cost, market gate và
walk-forward backtest. QAUR hiện là classical cardinality-preserving surrogate;
notebook không tuyên bố quantum advantage hoặc cho phép dùng vốn thật.
''')
    web_notebook["cells"][3] = markdown("## 2. Khôi phục bộ dữ liệu standalone đã khóa SHA-256")
    web_notebook["cells"][4] = code(embedded_data_cell)
    WEB_OUTPUT.write_text(
        json.dumps(web_notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(WEB_OUTPUT.name)


if __name__ == "__main__":
    main()
