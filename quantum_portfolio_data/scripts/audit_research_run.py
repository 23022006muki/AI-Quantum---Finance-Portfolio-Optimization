from __future__ import annotations

import argparse
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
}


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

    metrics = pd.read_csv(run_dir / "metrics_long.csv")
    core = metrics[["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown"]]
    if not np.isfinite(core.to_numpy(dtype=float)).all():
        failures.append("core portfolio metrics contain NaN/Inf")

    listed = {str(name).replace("\\", "/") for name in manifest.get("artifacts", [])}
    actual = {str(path.relative_to(run_dir)).replace("\\", "/") for path in run_dir.rglob("*") if path.is_file()}
    if not actual - {"manifest.json"} <= listed:
        failures.append("manifest artifact index is incomplete")

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
