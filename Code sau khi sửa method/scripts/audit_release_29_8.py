from __future__ import annotations

"""Fail-fast release audit for the 29/8 data, notebook and result bundle."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = args.results.resolve()

    data_manifest = json.loads(
        (root / "data 29_8" / "manifest_29_8.json").read_text(encoding="utf-8")
    )
    assert sha256(root / "data 29_8" / "data_29_8.csv") == data_manifest["csv_sha256"]
    assert sha256(root / "data 29_8" / "data_29_8.zip") == data_manifest["zip_sha256"]

    required = {
        "run_manifest.json",
        "confirmatory_hypothesis_tests.csv",
        "confirmatory_seed_robustness.csv",
        "confirmatory_xy_qaoa_holdout_audit.csv",
        "practical_robust_ranking.csv",
        "selected_practical_period_results.csv",
        "selected_practical_positive_return_evidence.csv",
        "selected_practical_h4_by_period.csv",
        "selected_practical_seed_robustness.csv",
        "september_2026_shadow_and_executable_basket.csv",
        "FINAL_RESULTS_29_8_VI.md",
    }
    missing = sorted(name for name in required if not (results / name).exists())
    assert not missing, f"Missing artifacts: {missing}"

    run_manifest = json.loads(
        (results / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["dataset_sha256"] == data_manifest["csv_sha256"]
    assert run_manifest["live_capital_authorized"] is False
    assert run_manifest["quantum_advantage_claimed"] is False

    tests = pd.read_csv(results / "confirmatory_hypothesis_tests.csv")
    assert len(tests) == 5 and tests["hypothesis"].nunique() == 5
    period = pd.read_csv(results / "selected_practical_period_results.csv")
    assert len(period) == 6
    assert period["cumulative_return"].gt(0.0).all()
    assert period["maximum_drawdown"].ge(-0.20).all()
    positive = pd.read_csv(
        results / "selected_practical_positive_return_evidence.csv"
    )
    assert len(positive) == 6 and positive["positive_economically"].all()

    xy = pd.read_csv(results / "confirmatory_xy_qaoa_holdout_audit.csv")
    assert len(xy) == 30
    assert xy["feasibility_rate"].eq(1.0).all()
    basket = pd.read_csv(
        results / "september_2026_shadow_and_executable_basket.csv"
    )
    totals = basket.groupby("method")["shadow_weight"].sum()
    assert totals.between(0.999999, 1.000001).all()
    print("RELEASE_29_8_AUDIT_OK")


if __name__ == "__main__":
    main()
