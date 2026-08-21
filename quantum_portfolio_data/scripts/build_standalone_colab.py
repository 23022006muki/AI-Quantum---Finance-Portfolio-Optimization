"""Build the self-contained Google Colab notebook without external source checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab" / "AI_Quantum_Standalone_Complete_System.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }


def writefile(path: str, content: str) -> dict:
    return code(f"%%writefile {path}\n{content.rstrip()}\n")


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def smoke_config() -> str:
    full = read("configs/data_17_8.yaml")
    replacements = {
        'label: "DATA 17/8 - OFFICIAL DISCLOSURE ENRICHED, FAIL-CLOSED EXPLORATORY HOSE PIPELINE"':
            'label: "STANDALONE COLAB SMOKE - REAL UPLOADED DATA"',
        "  max_folds: null": "  max_folds: 4",
        "  selection: all": "  selection: evenly_spaced",
        "  continuous_monthly: true": "  continuous_monthly: false",
        "  final_holdout_months: 12": "  final_holdout_months: 0",
        "  n_estimators: 180": "  n_estimators: 40",
        "  max_depth: 4": "  max_depth: 3",
        "  qaoa_depth: 2": "  qaoa_depth: 1",
        "  shots: 1024": "  shots: 256",
        "  seeds: [11, 23, 47]": "  seeds: [11]",
        "  parameter_trials: 45": "  parameter_trials: 24",
        "  representative_folds: 3": "  representative_folds: 1",
        "  seeds: [11, 23]": "  seeds: [11]",
    }
    for old, new in replacements.items():
        if old not in full:
            raise ValueError(f"Smoke config replacement target is missing: {old}")
        full = full.replace(old, new, 1)
    return full


FACADES = {
    "config.py": '''"""Central standalone configuration contract."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    training_months: int = 24
    validation_months: int = 3
    testing_months: int = 1
    rebalance_frequency: str = "monthly"
    maximum_universe_size: int = 300
    candidate_count: int = 8
    cardinality: int = 4
    minimum_history: int = 40
    maximum_weight: float = 0.40
    minimum_weight: float = 0.05
    risk_aversion: float = 1.25
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    turnover_penalty: float = 0.01
    qaoa_depth: int = 2
    shots: int = 1024
    optimizer_iterations: int = 45
    simulated_annealing_iterations: int = 800
    bootstrap_iterations: int = 500
    significance_level: float = 0.05
    execution_profile: str = "FULL"

    def to_dict(self) -> dict:
        return asdict(self)
''',
    "data_contracts.py": '''"""Required record types and schema for the uploaded research CSV."""
REQUIRED_RECORD_TYPES = {"METADATA", "PRICE", "BENCHMARK", "SECURITY", "CORPORATE_ACTION"}
PRICE_REQUIRED = {"ticker", "date", "open", "high", "low", "close", "volume", "available_at"}
BENCHMARK_REQUIRED = {"date", "total_return_index", "available_at"}
SECURITY_REQUIRED = {"ticker", "company_name", "exchange", "listing_date", "available_at"}
ACTION_REQUIRED = {"ticker", "event_type", "ex_date", "available_at"}
''',
    "data_loader.py": '''"""Public data-loading facade for the standalone notebook."""
from pathlib import Path
from scripts.import_colab_complete_csv import import_dataset

def load_complete_csv(csv_path: Path, workspace: Path) -> dict:
    return import_dataset(csv_path, workspace, replace=True)
''',
    "data_quality.py": '''"""Data quality facade."""
from src.data_pipeline import validate_data
__all__ = ["validate_data"]
''',
    "leakage_audit.py": '''"""Point-in-time leakage audit facade."""
from src.data_pipeline import leakage_audit
__all__ = ["leakage_audit"]
''',
    "features.py": '''"""Point-in-time feature engineering facade."""
from src.research import build_features, attach_point_in_time_features
__all__ = ["build_features", "attach_point_in_time_features"]
''',
    "walk_forward.py": '''"""Walk-forward split and purge facade."""
from src.research import make_folds, purged_fold_frames
__all__ = ["make_folds", "purged_fold_frames"]
''',
    "signals.py": '''"""XGBoost ranking and prediction facade."""
from src.research import fit_ranker, predict, calibrate_rank_signal_to_returns
__all__ = ["fit_ranker", "predict", "calibrate_rank_signal_to_returns"]
''',
    "adaptive_universe.py": '''"""Adaptive Universe Reduction facade."""
from src.research import adaptive_reduce
__all__ = ["adaptive_reduce"]
''',
    "covariance.py": '''"""Multivariate EWMA covariance facade."""
from src.research import ewma_mean_cov
__all__ = ["ewma_mean_cov"]
''',
    "qubo.py": '''"""Cardinality-constrained QUBO facade."""
from src.research import qubo_instance, energy, feasible_states
__all__ = ["qubo_instance", "energy", "feasible_states"]
''',
    "exact_solver.py": '''"""Exact combinatorial solver facade."""
from src.research import exact_solver
__all__ = ["exact_solver"]
''',
    "simulated_annealing.py": '''"""Simulated annealing solver facade."""
from src.research import simulated_annealing
__all__ = ["simulated_annealing"]
''',
    "penalty_qaoa.py": '''"""Full-Hilbert-space Penalty-QAOA facade."""
from src.research import penalty_qaoa_statevector
__all__ = ["penalty_qaoa_statevector"]
''',
    "xy_qaoa.py": '''"""Dicke-state feasible-subspace XY-QAOA facade."""
from src.research import xy_qaoa_statevector
__all__ = ["xy_qaoa_statevector"]
''',
    "weight_optimizer.py": '''"""Constrained classical weight optimizer facade."""
from src.research import optimize_weights
__all__ = ["optimize_weights"]
''',
    "backtest.py": '''"""Out-of-sample accounting facade."""
from src.research import record_rebalanced_strategy, financial_metrics, transaction_cost_breakdown
__all__ = ["record_rebalanced_strategy", "financial_metrics", "transaction_cost_breakdown"]
''',
    "statistics.py": '''"""Block bootstrap and Holm correction facade."""
from src.research import paired_block_bootstrap_test, holm_adjust
__all__ = ["paired_block_bootstrap_test", "holm_adjust"]
''',
    "reporting.py": '''"""Research reporting facade."""
from src.research import create_report
__all__ = ["create_report"]
''',
    "pipeline.py": '''"""End-to-end research pipeline facade."""
from src.research import run_experiment
__all__ = ["run_experiment"]
''',
}


def facade_cell() -> dict:
    mapping = json.dumps(FACADES, ensure_ascii=False, indent=2)
    return code(
        "# Materialize the documented standalone module boundaries.\n"
        "from pathlib import Path\n"
        f"FACADE_SOURCES = {mapping}\n"
        "for name, source in FACADE_SOURCES.items():\n"
        "    target = STANDALONE_ROOT / 'ai_quantum_system' / name\n"
        "    target.write_text(source, encoding='utf-8')\n"
        "print('Facade modules written:', len(FACADE_SOURCES))\n"
    )


def build() -> Path:
    pyproject = '''[project]
name = "ai-quantum-standalone-colab"
version = "1.0.0"
description = "Self-contained AI-Quantum portfolio research pipeline for Google Colab"
requires-python = ">=3.11"
dependencies = [
  "numpy==2.2.6", "pandas==2.3.1", "pyarrow==24.0.0",
  "scipy==1.16.1", "scikit-learn==1.8.0", "xgboost==3.3.0",
  "matplotlib==3.10.5", "PyYAML==6.0.2", "Pillow==11.3.0"
]

[project.optional-dependencies]
dev = ["pytest==9.1.1"]

[tool.pytest.ini_options]
testpaths = ["tests"]
'''
    init = '''"""AI-Quantum standalone research package embedded in the notebook."""
__version__ = "1.0.0"
'''
    cells = [
        markdown('''# AI–Quantum Portfolio Optimization — Standalone Complete System

Notebook này chứa **toàn bộ mã nguồn hệ thống trong các cell**. Không `git clone`, không tải source từ GitHub/Drive và không phụ thuộc repository bên ngoài. Người dùng chỉ upload một CSV hoặc ZIP dữ liệu.

Pipeline nghiên cứu:

`Upload → schema/hash audit → point-in-time panel → feature engineering → XGBoost/EWMA → Adaptive Universe Reduction → cardinality QUBO → Exact/SA/Penalty-QAOA/XY-QAOA → constrained classical weights → walk-forward backtest → bootstrap/Holm tests → report`

EWMA là bộ ước lượng mean–covariance đa biến và tín hiệu đối chứng, không phải AI. XY-QAOA sử dụng Dicke state và ideal statevector simulator nội bộ; runtime simulator không chứng minh quantum speedup hay quantum advantage. Nếu gói dữ liệu không có đầy đủ disclosure binaries, PIT financial statements và lịch sử thành viên HOSE hoàn chỉnh, kết quả phải được diễn giải là **exploratory**.'''),
        markdown("## 1. Cấu hình tập trung"),
        code('''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile

EXECUTION_PROFILE = "FULL"  # Đổi thành "SMOKE" để kiểm tra 4 folds.
EXPECTED_CSV_SHA256 = "aea9644cfafc359ed04546deca62fea83509826864463669b219f370f1433eba"
STANDALONE_ROOT = Path("/content/ai_quantum_standalone")
WORKSPACE = STANDALONE_ROOT / "outputs" / "uploaded_research_data"
RESULTS_ROOT = Path("/content/ai_quantum_results")
VENV_DIR = Path("/content/ai_quantum_venv")
NOTEBOOK_CONFIG = {
    "random_seed": 42, "start_date": "2020-01-01", "end_date": "2025-12-31",
    "training_months": 24, "validation_months": 3, "testing_months": 1,
    "rebalance_frequency": "monthly", "maximum_universe_size": 300,
    "candidate_count": 8, "cardinality": 4, "minimum_history": 40,
    "maximum_weight": 0.40, "minimum_weight": 0.05, "risk_aversion": 1.25,
    "transaction_cost_bps": 10, "slippage_bps": 5, "turnover_penalty": 0.01,
    "qaoa_depth": 2, "shots": 1024, "optimizer_iterations": 45,
    "simulated_annealing_iterations": 800, "bootstrap_iterations": 500,
    "significance_level": 0.05, "execution_profile": EXECUTION_PROFILE,
}
assert EXECUTION_PROFILE in {"SMOKE", "FULL"}
print(json.dumps(NOTEBOOK_CONFIG, indent=2, ensure_ascii=False))'''),
        markdown("## 2. Tạo package tạm và nhúng toàn bộ source code"),
        code('''if STANDALONE_ROOT.exists():
    shutil.rmtree(STANDALONE_ROOT)
for directory in [
    STANDALONE_ROOT / "src", STANDALONE_ROOT / "scripts",
    STANDALONE_ROOT / "configs", STANDALONE_ROOT / "tests",
    STANDALONE_ROOT / "ai_quantum_system", RESULTS_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)
print("Standalone root:", STANDALONE_ROOT)'''),
        writefile("/content/ai_quantum_standalone/pyproject.toml", pyproject),
        writefile("/content/ai_quantum_standalone/src/__init__.py", init),
        writefile("/content/ai_quantum_standalone/src/cli.py", read("src/cli.py")),
        writefile("/content/ai_quantum_standalone/src/data_pipeline.py", read("src/data_pipeline.py")),
        writefile("/content/ai_quantum_standalone/src/research.py", read("src/research.py")),
        writefile("/content/ai_quantum_standalone/scripts/__init__.py", '"""Standalone helper scripts."""\n'),
        writefile(
            "/content/ai_quantum_standalone/scripts/import_colab_complete_csv.py",
            read("scripts/import_colab_complete_csv.py"),
        ),
        facade_cell(),
        writefile("/content/ai_quantum_standalone/configs/standalone_full.yaml", read("configs/data_17_8.yaml")),
        writefile("/content/ai_quantum_standalone/configs/standalone_smoke.yaml", smoke_config()),
        writefile("/content/ai_quantum_standalone/tests/test_standalone.py", read("colab/standalone_assets/test_standalone.py")),
        markdown("## 3. Cài thư viện trong môi trường cô lập tương thích Colab Python 3.12"),
        code('''if VENV_DIR.exists():
    shutil.rmtree(VENV_DIR)
# Colab may omit ensurepip/python3-venv. PyPA virtualenv supplies isolated seed wheels.
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "virtualenv==20.35.4"], check=True)
subprocess.run([sys.executable, "-m", "virtualenv", str(VENV_DIR)], check=True)
ENV_PYTHON = VENV_DIR / "bin" / "python"
PIPELINE_ENV = os.environ.copy()
PIPELINE_ENV["MPLBACKEND"] = "Agg"
subprocess.run([str(ENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
subprocess.run([str(ENV_PYTHON), "-m", "pip", "install", "-e", f"{STANDALONE_ROOT}[dev]"], check=True)
subprocess.run([str(ENV_PYTHON), "-m", "pip", "check"], check=True)
health = "import sys,numpy,pandas,pyarrow,scipy,sklearn,xgboost,matplotlib,yaml; print({'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'pandas':pandas.__version__,'pyarrow':pyarrow.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'matplotlib':matplotlib.__version__,'backend':matplotlib.get_backend()})"
subprocess.run([str(ENV_PYTHON), "-c", health], cwd=STANDALONE_ROOT, env=PIPELINE_ENV, check=True)
print("Isolated environment ready:", ENV_PYTHON)'''),
        markdown('''## 4. Upload dữ liệu

Chọn đúng một file `ai_quantum_complete_dataset.csv` hoặc `ai_quantum_complete_dataset.zip`. Nếu dùng một dataset mới hợp lệ, đặt `EXPECTED_CSV_SHA256 = ""` hoặc thay bằng hash đã kiểm toán của file đó.'''),
        code('''from google.colab import files
uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Hãy upload đúng một file CSV hoặc ZIP.")
UPLOAD_PATH = Path("/content") / next(iter(uploaded))
print("Uploaded:", UPLOAD_PATH, f"{UPLOAD_PATH.stat().st_size / 1e6:.2f} MB")'''),
        markdown("## 5. Giải nén an toàn, tính SHA-256 và kiểm tra hợp đồng file"),
        code('''def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

if UPLOAD_PATH.suffix.lower() == ".zip":
    extract_dir = Path("/content/standalone_uploaded_csv")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    with zipfile.ZipFile(UPLOAD_PATH) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"ZIP bị lỗi tại: {corrupt}")
        root = extract_dir.resolve()
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"ZIP chứa đường dẫn không an toàn: {member.filename}")
        archive.extractall(extract_dir)
    candidates = list(extract_dir.rglob("*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"ZIP phải chứa đúng một CSV; tìm thấy {len(candidates)}")
    CSV_PATH = candidates[0]
elif UPLOAD_PATH.suffix.lower() == ".csv":
    CSV_PATH = UPLOAD_PATH
else:
    raise ValueError("Chỉ chấp nhận CSV hoặc ZIP.")

CSV_SHA256 = sha256_stream(CSV_PATH)
if EXPECTED_CSV_SHA256 and CSV_SHA256.lower() != EXPECTED_CSV_SHA256.lower():
    raise ValueError(f"SHA-256 không khớp: {CSV_SHA256}")
print("CSV:", CSV_PATH)
print("SHA-256:", CSV_SHA256)'''),
        markdown("## 6. Import, schema validation, data-quality audit và leakage audit"),
        code('''import_command = [
    str(ENV_PYTHON), "scripts/import_colab_complete_csv.py", str(CSV_PATH),
    "--workspace", str(WORKSPACE),
]
subprocess.run(import_command, cwd=STANDALONE_ROOT, env=PIPELINE_ENV, check=True)
IMPORT_REPORT = json.loads((WORKSPACE / "outputs/reports/COLAB_CSV_IMPORT_REPORT.json").read_text(encoding="utf-8"))
assert IMPORT_REPORT["input_csv_sha256"] == CSV_SHA256
assert IMPORT_REPORT["quality_status"] == "pass"
assert IMPORT_REPORT["leakage_status"] in {"pass", "pass_with_limitations"}
assert IMPORT_REPORT["exploratory_run_permitted"]
print(json.dumps(IMPORT_REPORT, indent=2, ensure_ascii=False))'''),
        code('''import numpy as np
import pandas as pd
from IPython.display import Markdown, Image, display

normalized = WORKSPACE / "outputs/normalized"
prices = pd.read_parquet(normalized / "prices.parquet")
benchmark = pd.read_parquet(normalized / "benchmark.parquet")
master = pd.read_parquet(normalized / "security_master_full.parquet")
actions = pd.read_parquet(normalized / "corporate_actions.parquet")
quality = json.loads((WORKSPACE / "outputs/reports/data_quality.json").read_text(encoding="utf-8"))
leakage = json.loads((WORKSPACE / "outputs/reports/leakage_audit.json").read_text(encoding="utf-8"))
data_summary = pd.DataFrame({
    "Chỉ tiêu": ["SHA-256", "PRICE rows", "Runtime tickers", "Security master", "Start", "End", "Benchmark rows", "Corporate actions", "Data quality", "Leakage audit", "Research classification"],
    "Kết quả": [CSV_SHA256, len(prices), prices.ticker.nunique(), master.ticker.nunique(), str(prices.date.min().date()), str(prices.date.max().date()), len(benchmark), len(actions), quality["status"], leakage["status"], "EXPLORATORY"],
})
display(data_summary)
display(prices.head())
display(Markdown("**Phạm vi diễn giải:** runtime đủ cho nghiên cứu exploratory; không tự động nâng cấp thành confirmatory full-HOSE nếu thiếu PIT financial statements, disclosure binaries hoặc lịch sử membership đầy đủ."))'''),
        markdown("## 7. Chạy validation tests trước thực nghiệm"),
        code('''test_result = subprocess.run(
    [str(ENV_PYTHON), "-m", "pytest", "-q"], cwd=STANDALONE_ROOT,
    env=PIPELINE_ENV, text=True, capture_output=True,
)
print(test_result.stdout)
if test_result.stderr:
    print(test_result.stderr)
test_result.check_returncode()
print("Validation gate passed.")'''),
        markdown("## 8. Chạy pipeline SMOKE hoặc FULL trên chính dữ liệu upload"),
        code('''CONFIG_PATH = STANDALONE_ROOT / "configs" / (
    "standalone_smoke.yaml" if EXECUTION_PROFILE == "SMOKE" else "standalone_full.yaml"
)
experiments = WORKSPACE / "outputs/experiments"
experiments.mkdir(parents=True, exist_ok=True)
before = {path.resolve() for path in experiments.iterdir() if path.is_dir()}
runner = (
    "from pathlib import Path; from src.research import run_experiment; "
    f"print(run_experiment(Path(r'{WORKSPACE}'), Path(r'{CONFIG_PATH}')))"
)
subprocess.run([str(ENV_PYTHON), "-c", runner], cwd=STANDALONE_ROOT, env=PIPELINE_ENV, check=True)
after = {path.resolve() for path in experiments.iterdir() if path.is_dir()}
created = sorted(after - before, key=lambda path: path.stat().st_mtime)
if not created:
    raise RuntimeError("Pipeline không tạo experiment mới.")
ACTIVE = created[-1]
manifest = json.loads((ACTIVE / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("status") != "success":
    raise RuntimeError(f"Experiment không thành công: {manifest}")
expected_folds = 4 if EXECUTION_PROFILE == "SMOKE" else manifest["folds_requested"]
assert manifest["folds_completed"] == expected_folds
print("ACTIVE:", ACTIVE)
print(json.dumps({key: manifest.get(key) for key in ["status", "experiment_id", "folds_requested", "folds_completed", "actual_oos_start", "actual_oos_end", "data_class"]}, indent=2, ensure_ascii=False))'''),
        markdown("## 9. Chuẩn hóa artifact, tạo biểu đồ bổ sung và báo cáo tiếng Việt"),
        code('''import platform

aliases = {
    "data_quality.json": "data_quality_report.json",
    "fold_manifest.csv": "folds.csv",
    "feature_coverage_by_fold.csv": "features_summary.csv",
    "selected_universe.csv": "adaptive_universe.csv",
    "optimization_instances.json": "qubo_instances.json",
}
for source, target in aliases.items():
    source_path = ACTIVE / source
    if source_path.exists():
        shutil.copy2(source_path, ACTIVE / target)

environment = {
    "kernel_python": sys.version,
    "pipeline_python": str(ENV_PYTHON),
    "platform": platform.platform(),
    "execution_profile": EXECUTION_PROFILE,
    "mplbackend": PIPELINE_ENV["MPLBACKEND"],
}
(ACTIVE / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
(ACTIVE / "dataset_hash.json").write_text(json.dumps({"csv_sha256": CSV_SHA256, "csv_path": str(CSV_PATH)}, indent=2), encoding="utf-8")

rankings = pd.read_csv(ACTIVE / "rankings.csv")
comparisons = pd.read_csv(ACTIVE / "comparisons.csv")
metrics = pd.read_csv(ACTIVE / "strategy_metrics_summary.csv")
tests = pd.read_csv(ACTIVE / "statistical_tests.csv")
ablations = pd.read_csv(ACTIVE / "ablation_results.csv")
sensitivity = pd.read_csv(ACTIVE / "sensitivity_results.csv")
latest = pd.read_csv(ACTIVE / "latest_selected_portfolio.csv")
latest_summary = json.loads((ACTIVE / "latest_portfolio_summary.json").read_text(encoding="utf-8"))
constraints = pd.read_csv(ACTIVE / "constraint_diagnostics.csv")
weights = pd.read_csv(ACTIVE / "weights.csv")
exposure = pd.read_csv(ACTIVE / "exposure_by_fold.csv") if (ACTIVE / "exposure_by_fold.csv").exists() else pd.DataFrame()

figures = ACTIVE / "figures"
figures.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
full_weights = weights[weights.strategy.eq("full_pipeline_xy_qaoa")].copy()
if not full_weights.empty:
    pivot = full_weights.pivot_table(index="decision_time", columns="ticker", values="weight", aggfunc="sum", fill_value=0)
    top = pivot.mean().nlargest(min(10, len(pivot.columns))).index
    ax = pivot[top].plot.area(figsize=(12, 6), colormap="tab20")
    ax.set(title="Tỷ trọng các tài sản chính theo thời gian", xlabel="Ngày tái cân bằng", ylabel="Tỷ trọng")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.tight_layout(); plt.savefig(figures / "weights_over_time.png", dpi=160); plt.close()
if not exposure.empty and "cash_weight" in exposure:
    ax = exposure.plot(x="decision_time", y="cash_weight", figsize=(11, 4), legend=False)
    ax.set(title="Tỷ trọng tiền mặt theo fold", xlabel="Ngày tái cân bằng", ylabel="Cash weight")
    plt.tight_layout(); plt.savefig(figures / "cash_exposure.png", dpi=160); plt.close()
if "sector" in latest and latest["sector"].fillna("").str.len().gt(0).any():
    sector = latest.assign(sector=latest.sector.fillna("Unknown")).groupby("sector").target_weight.sum().sort_values()
    ax = sector.plot.barh(figsize=(9, 5)); ax.set(title="Phân bổ ngành của danh mục cuối", xlabel="Target weight", ylabel="Ngành")
    plt.tight_layout(); plt.savefig(figures / "sector_allocation.png", dpi=160); plt.close()

def row(test_name: str):
    found = tests.loc[tests.test.eq(test_name)]
    return found.iloc[0] if len(found) else None

h1 = row("xgboost_rank_ic_vs_ewma_rank_ic")
h2_return = row("adaptive_universe_forward_return_vs_fixed_topm")
h2_div = row("adaptive_universe_diversification_vs_fixed_topm")
h3 = row("xy_feasibility_vs_penalty_qaoa")
h4 = row("xy_optimality_gap_vs_penalty_qaoa")
h5 = tests.loc[tests.hypothesis.eq("H5")]
alpha = NOTEBOOK_CONFIG["significance_level"]

def significant(test_row) -> bool:
    return test_row is not None and pd.notna(test_row.p_value_holm) and float(test_row.p_value_holm) < alpha

hypothesis_text = []
if h1 is not None:
    h1_label = "được hỗ trợ" if significant(h1) and h1.mean_difference > 0 else "không được hỗ trợ"
    hypothesis_text.append(f"### H1 — {h1_label}\\nChênh lệch Rank IC XGBoost–EWMA là {h1.mean_difference:.4f}, CI 95% [{h1.ci_low:.4f}, {h1.ci_high:.4f}], p-Holm={h1.p_value_holm:.4f}. Kết luận chỉ dựa trên dự báo ngoài mẫu của run hiện tại.")
h2_count = int(significant(h2_return)) + int(significant(h2_div))
h2_label = "được hỗ trợ đầy đủ" if h2_count == 2 else "được hỗ trợ một phần" if h2_count == 1 else "không được hỗ trợ"
hypothesis_text.append(f"### H2 — {h2_label}\\nAUR được đánh giá đồng thời về forward return và đa dạng hóa; p-Holm tương ứng là {getattr(h2_return, 'p_value_holm', np.nan):.4f} và {getattr(h2_div, 'p_value_holm', np.nan):.4f}.")
if h3 is not None:
    h3_label = "được hỗ trợ trong ideal simulator" if significant(h3) and h3.mean_difference > 0 else "không được hỗ trợ"
    hypothesis_text.append(f"### H3 — {h3_label}\\nChênh lệch feasibility rate là {h3.mean_difference:.4f}, p-Holm={h3.p_value_holm:.4f}. Kết quả không được ngoại suy thành quantum advantage trên phần cứng thật.")
if h4 is not None:
    h4_label = "được hỗ trợ có điều kiện so với Penalty-QAOA" if significant(h4) and h4.mean_difference > 0 else "không được hỗ trợ"
    hypothesis_text.append(f"### H4 — {h4_label}\\nCải thiện optimality gap so với Penalty-QAOA là {h4.mean_difference:.4f}, p-Holm={h4.p_value_holm:.4f}; Exact vẫn là nghiệm tham chiếu và Simulated Annealing vẫn phải được đọc riêng.")
h5_sig = int((h5.p_value_holm < alpha).sum()) if len(h5) else 0
h5_label = "được hỗ trợ" if h5_sig and (h5.loc[h5.p_value_holm < alpha, "mean_difference"] > 0).all() else "không được hỗ trợ"
hypothesis_text.append(f"### H5 — {h5_label}\\nCó {h5_sig}/{len(h5)} so sánh hiệu quả tài chính đạt ý nghĩa sau hiệu chỉnh Holm. Lợi nhuận dương, nếu xuất hiện, không tự thân chứng minh alpha thống kê.")
hypothesis_text.append(f"### H6 — phân tích độ nhạy đã hoàn thành\\nCó {len(sensitivity)} quan sát sensitivity trong lưới cấu hình đã khai báo. Kết luận chỉ có giá trị trong lưới depth, shots, seed, cardinality, noise và chi phí được chạy.")
HYPOTHESIS_MARKDOWN = "\\n\\n".join(hypothesis_text)

pipeline_metrics = metrics.loc[metrics.strategy.eq("full_pipeline_xy_qaoa")]
pipeline_line = "Không có metric full pipeline."
if len(pipeline_metrics):
    value = pipeline_metrics.iloc[0]
    pipeline_line = f"Full pipeline đạt cumulative return {value.cumulative_return:.2%}, annualized return {value.annualized_return:.2%}, Sharpe {value.sharpe:.3f} và maximum drawdown {value.max_drawdown:.2%} sau chi phí trong cửa sổ ngoài mẫu."
report_vi = f"""# BÁO CÁO KẾT QUẢ HỆ THỐNG AI–QUANTUM STANDALONE

## 1. Tóm tắt hệ thống
Hệ thống kết hợp XGBoost, EWMA đa biến, Adaptive Universe Reduction, cardinality-constrained QUBO, Exact Solver, Simulated Annealing, Penalty-QAOA, feasible-subspace XY-QAOA và tối ưu tỷ trọng cổ điển trong thiết kế walk-forward point-in-time.

## 2. Dữ liệu và kiểm toán
Run sử dụng {len(prices):,} quan sát giá của {prices.ticker.nunique()} mã, giai đoạn {prices.date.min().date()}–{prices.date.max().date()}. Data quality là `{quality['status']}`, leakage audit là `{leakage['status']}` và phân loại nghiên cứu là exploratory.

## 3. Thiết kế thực nghiệm
Profile `{EXECUTION_PROFILE}` hoàn thành {manifest['folds_completed']}/{manifest['folds_requested']} folds. Mọi feature, scaler, mô hình và covariance được ước lượng trong training/validation window trước khi đánh giá test window.

## 4. Kết quả tín hiệu, AUR và solver
XGBoost được đánh giá bằng Rank IC ngoài mẫu; EWMA là đối chứng và bộ ước lượng covariance. AUR kết hợp tín hiệu, thanh khoản, rủi ro và tương quan để tạo tập 8 ứng viên. Solver comparison được lưu trong `comparisons.csv`; không sử dụng runtime simulator để tuyên bố quantum speedup.

## 5. Hiệu quả danh mục
{pipeline_line}

## 6. Kết luận giả thuyết
{HYPOTHESIS_MARKDOWN}

## 7. Rổ cuối và giới hạn
Rổ cuối tại {latest.decision_time.iloc[0]} gồm {', '.join(latest.ticker.astype(str))}. Đây là output nghiên cứu lịch sử, không phải khuyến nghị đầu tư hiện tại. Hạn chế chính gồm phân loại exploratory, simulator lý tưởng, dữ liệu ngành còn thiếu ở một số mã và chưa có bằng chứng quantum advantage.

## 8. Hướng nghiên cứu tiếp theo
Cần bổ sung PIT financial statements, lịch sử membership HOSE, kiểm chứng corporate actions đa nguồn và thử nghiệm noise/hardware trước khi đưa ra kết luận confirmatory.
"""
(ACTIVE / "research_report_vi.md").write_text(report_vi, encoding="utf-8")
print("Artifact normalization and Vietnamese report completed.")'''),
        markdown("## 10. Kết quả dữ liệu, mô hình, solver, danh mục và giả thuyết"),
        code('''display(Markdown(f"### Experiment `{manifest['experiment_id']}` — {manifest['folds_completed']}/{manifest['folds_requested']} folds"))
display(data_summary)
coverage_path = WORKSPACE / "outputs/reports/coverage_report.csv"
if coverage_path.exists():
    display(pd.read_csv(coverage_path).head(20))
fold_rank = rankings.groupby("fold")[["xgboost_rank_ic", "ewma_rank_ic"]].first()
display(pd.DataFrame({
    "Model": ["XGBoost", "EWMA"],
    "Mean Rank IC": [fold_rank.xgboost_rank_ic.mean(), fold_rank.ewma_rank_ic.mean()],
    "Median Rank IC": [fold_rank.xgboost_rank_ic.median(), fold_rank.ewma_rank_ic.median()],
    "IC hit rate": [(fold_rank.xgboost_rank_ic > 0).mean(), (fold_rank.ewma_rank_ic > 0).mean()],
}))
aur = pd.read_csv(ACTIVE / "aur_diagnostics.csv")
display(aur.head(20))
display(comparisons)
display(metrics.sort_values("sharpe", ascending=False))
display(ablations)
display(sensitivity.head(40))
display(tests)
display(latest)
display(constraints.tail(1).T.rename(columns={constraints.tail(1).index[0]: "Fold cuối"}))
display(Markdown("**Rổ cổ phiếu cuối là output lịch sử của backtest, không phải khuyến nghị đầu tư hiện tại.**"))'''),
        code('''for figure_name in [
    "equity_curve.png", "drawdown.png", "risk_return.png", "rank_ic_by_fold.png",
    "feasibility_rate.png", "optimality_gap.png", "turnover_and_cost.png",
    "sensitivity_analysis.png", "weights_over_time.png", "cash_exposure.png",
    "sector_allocation.png",
]:
    figure_path = ACTIVE / "figures" / figure_name
    if figure_path.exists():
        display(Markdown(f"### {figure_name.replace('_', ' ').replace('.png', '').title()}"))
        display(Image(filename=str(figure_path)))'''),
        code('''display(Markdown("# Kết luận kiểm định giả thuyết"))
display(Markdown(HYPOTHESIS_MARKDOWN))
display(Markdown((ACTIVE / "research_report_vi.md").read_text(encoding="utf-8")))'''),
        markdown("## 11. Audit artifact và tải kết quả"),
        code('''required_artifacts = [
    "config_freeze.json", "manifest.json", "environment.json", "dataset_hash.json",
    "data_quality_report.json", "leakage_audit.json", "folds.csv", "features_summary.csv",
    "rankings.csv", "adaptive_universe.csv", "qubo_instances.json", "solver_runs.csv",
    "comparisons.csv", "weights.csv", "trades.csv", "portfolio_returns.csv",
    "strategy_metrics_summary.csv", "statistical_tests.csv", "ablation_results.csv",
    "sensitivity_results.csv", "constraint_diagnostics.csv", "latest_selected_portfolio.csv",
    "latest_portfolio_summary.json", "research_report_vi.md",
]
missing = [name for name in required_artifacts if not (ACTIVE / name).exists()]
if missing:
    raise FileNotFoundError(f"Thiếu artifact bắt buộc: {missing}")
artifact_count = len([path for path in ACTIVE.rglob("*") if path.is_file()])
result_target = RESULTS_ROOT / manifest["experiment_id"]
if result_target.exists():
    shutil.rmtree(result_target)
shutil.copytree(ACTIVE, result_target)
archive = shutil.make_archive(str(RESULTS_ROOT / f"ai_quantum_{manifest['experiment_id']}"), "zip", root_dir=result_target)
print({"status": manifest["status"], "folds": manifest["folds_completed"], "artifacts": artifact_count, "zip": archive, "size_mb": round(Path(archive).stat().st_size / 1e6, 2)})
from google.colab import files
display(Markdown(f"Chạy `files.download(r'{archive}')` để tải toàn bộ kết quả."))'''),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "standalone": {
                "source_files_embedded": 5 + len(FACADES),
                "generated_by": Path(__file__).name,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(path)
    print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())
