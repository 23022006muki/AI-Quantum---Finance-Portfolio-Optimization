from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_data_29_8_bundle_contract() -> None:
    folder = ROOT / "data 29_8"
    csv_path = folder / "data_29_8.csv"
    zip_path = folder / "data_29_8.zip"
    manifest = json.loads((folder / "manifest_29_8.json").read_text(encoding="utf-8"))
    assert csv_path.exists() and zip_path.exists()
    assert sha256(csv_path) == manifest["csv_sha256"]
    assert sha256(zip_path) == manifest["zip_sha256"]
    assert manifest["price_end"] == "2026-08-28"
    assert manifest["provisional_forward_tickers"] >= 96
    assert manifest["duplicate_price_ticker_dates"] == 0
    assert manifest["missing_adjusted_close_price_rows"] == 0
    assert manifest["live_capital_authorized"] is False
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["data_29_8.csv"]


def test_colab_29_8_is_standalone_and_contains_full_system() -> None:
    notebook_path = ROOT / "colab" / "AUR_QAUR_Practical_Optimization_29_8_Full_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "git clone" not in text.lower()
    assert "github.com" not in text.lower()
    assert "run_constraint_strategy_search.py" in text
    assert "run_colab_29_8_complete.py" in text
    assert "confirmatory_hypothesis_tests.csv" in text
    assert "selected_practical_positive_return_evidence.csv" in text
    assert "holm_adjusted_pvalue" in text
    assert "confirmatory_xy_qaoa_holdout_audit.csv" in text
    assert "selected_practical_results_and_basket.png" in text
    assert "september_2026_shadow_and_executable_basket.csv" in text
    assert "RESEARCH_AUDIT_29_8_OK" in text
    write_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("%%writefile")
    ]
    assert len(write_cells) == 2
    for index, source in enumerate(write_cells):
        compile(source.split("\n", 1)[1], f"embedded-{index}", "exec")
