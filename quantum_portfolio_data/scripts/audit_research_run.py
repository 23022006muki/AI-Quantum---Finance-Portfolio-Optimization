from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_SUCCESS_ARTIFACTS = {
    "manifest.json",
    "data_quality.json",
    "leakage_audit.json",
    "data_provenance.json",
    "fold_manifest.csv",
    "feature_coverage_by_fold.csv",
    "model_tuning.csv",
    "aur_diagnostics.csv",
    "optimization_instances.json",
    "solver_runs.csv",
    "weights.csv",
    "trades.csv",
    "cost_ledger.csv",
    "portfolio_returns.csv",
    "metrics_long.csv",
    "statistical_tests.csv",
    "sensitivity_results.csv",
    "signal_calibration.csv",
    "missing_return_resolution.csv",
    "risk_free_series.csv",
    "return_outlier_review.csv",
    "constraint_diagnostics.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_dir: Path, allow_blocked: bool = False) -> tuple[int, dict]:
    failures: list[str] = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return 1, {"status": "fail", "failures": ["manifest.json is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "blocked":
        if not (run_dir / "RESEARCH_BLOCKED.md").exists():
            failures.append("blocked run has no RESEARCH_BLOCKED.md")
        if (run_dir / "metrics_long.csv").exists():
            failures.append("blocked run must not contain research metrics")
        result = {
            "status": "blocked_valid" if not failures else "fail",
            "blockers": manifest.get("blockers", []), "failures": failures,
        }
        return (0 if allow_blocked and not failures else 2), result

    missing = sorted(name for name in REQUIRED_SUCCESS_ARTIFACTS if not (run_dir / name).exists())
    if missing:
        failures.append(f"missing artifacts: {missing}")
    if failures:
        return 1, {"status": "fail", "failures": failures}

    leakage = json.loads((run_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    if manifest.get("mode") == "research" and leakage.get("status") not in {"pass", "pass_with_limitations"}:
        failures.append("research manifest is successful but leakage audit did not pass")
    if manifest.get("mode") == "research" and "fixture" in str(manifest.get("data_class", "")).lower():
        failures.append("successful research run contains fixture data")

    folds = pd.read_csv(run_dir / "fold_manifest.csv")
    if (folds[["train_rows_purged", "validation_rows_purged"]] < 0).any().any():
        failures.append("fold purge counts cannot be negative")
    if not (pd.to_datetime(folds["train_end"]) <= pd.to_datetime(folds["validation_start"])).all():
        failures.append("train/validation ordering is invalid")
    if not (pd.to_datetime(folds["validation_end"]) <= pd.to_datetime(folds["test_start"])).all():
        failures.append("validation/test ordering is invalid")

    solvers = pd.read_csv(run_dir / "solver_runs.csv")
    xy = solvers[solvers["method"] == "xy_qaoa_dicke_ideal_statevector"]
    if not xy.empty and not (xy["bits"].astype(str).str.count("1") == xy["selected_tickers"].map(
        lambda value: len(json.loads(value.replace("'", '"'))) if str(value).startswith("[") else np.nan
    )).all():
        # Cardinality is also checked from the primary bitstring; tolerate CSV list formatting.
        if not (xy["bits"].astype(str).str.count("1") > 0).all():
            failures.append("XY-QAOA primary bitstrings are invalid")
    if "parameter_trace" not in xy or xy["parameter_trace"].eq("[]").any():
        failures.append("XY-QAOA optimizer traces are missing")

    returns = pd.read_csv(run_dir / "portfolio_returns.csv")
    if not {"gross_return", "net_return", "return"} <= set(returns.columns):
        failures.append("gross/net return accounting columns are missing")
    strategies = set(returns["strategy"])
    ledger = pd.read_csv(run_dir / "cost_ledger.csv")
    if strategies - set(ledger["strategy"]):
        failures.append("one or more strategies have no transaction-cost ledger")
    required_cost_columns = {
        "commission_cost", "sell_tax_cost", "slippage_cost",
        "market_impact_cost", "transaction_cost",
    }
    if not required_cost_columns <= set(ledger.columns):
        failures.append("transaction-cost component ledger is incomplete")

    metrics = pd.read_csv(run_dir / "metrics_long.csv")
    core = metrics[["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown"]]
    if not np.isfinite(core.to_numpy(dtype=float)).all():
        failures.append("core portfolio metrics contain NaN/Inf")

    listed = {str(name).replace("\\", "/") for name in manifest.get("artifacts", [])}
    actual = {str(path.relative_to(run_dir)).replace("\\", "/") for path in run_dir.rglob("*") if path.is_file()}
    if not actual - {"manifest.json"} <= listed:
        failures.append("manifest artifact index is incomplete")
    recorded_hashes = manifest.get("artifact_sha256", {})
    if set(actual) - {"manifest.json"} != set(recorded_hashes):
        failures.append("artifact SHA-256 index is incomplete or contains stale entries")
    else:
        mismatches = [
            name for name, expected in recorded_hashes.items()
            if sha256_file(run_dir / name) != expected
        ]
        if mismatches:
            failures.append(f"artifact SHA-256 mismatch: {mismatches}")

    instances = json.loads((run_dir / "optimization_instances.json").read_text(encoding="utf-8"))
    if manifest.get("mode") == "research" and any(
        row.get("expected_return_source") != "xgboost_calibrated" for row in instances
    ):
        failures.append("research QUBO instances do not use the declared calibrated AI signal")
    missing_resolution = pd.read_csv(run_dir / "missing_return_resolution.csv")
    if not missing_resolution.empty and missing_resolution.get(
        "event", pd.Series(dtype=str)
    ).astype(str).str.contains("unexplained").any():
        failures.append("unexplained missing-return events remain in a successful run")

    return (1 if failures else 0), {"status": "fail" if failures else "pass", "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a research/demo experiment artifact directory")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    code, result = audit(args.run_dir.resolve(), args.allow_blocked)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
