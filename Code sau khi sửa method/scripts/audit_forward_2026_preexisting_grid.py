from __future__ import annotations

"""Audit every strategy specification that existed before the 2026 forward test.

This is a diagnostic/post-selection analysis.  It must not be reported as a new
one-shot forward result because the 2026 outcomes are already observed.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_constraint_strategy_search import (
    build_features,
    financial_metrics,
    load_market_data,
    make_configs,
    make_folds,
    run_configuration,
)


CORE_COLUMNS = ["date", "ticker", "adjusted_close", "volume", "trading_value"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--forward-prices", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base, security, _ = load_market_data(args.base_dataset)
    del security
    forward = pd.read_parquet(args.forward_prices).copy()
    forward["date"] = pd.to_datetime(forward["date"], errors="coerce")
    for column in ("adjusted_close", "volume", "trading_value"):
        forward[column] = pd.to_numeric(forward[column], errors="coerce")
    prices = (
        pd.concat([base[CORE_COLUMNS], forward[CORE_COLUMNS]], ignore_index=True)
        .dropna(subset=CORE_COLUMNS)
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
    )
    features = build_features(prices)
    folds = make_folds(features["date"])
    forward_ids = [f["fold"] for f in folds if f["test_start"] >= pd.Timestamp("2026-01-01")]
    if not forward_ids:
        raise RuntimeError("No 2026 forward folds found")
    # Fold immediately before the forward window is only used to initialize the
    # previous portfolio for turnover and persistence calculations.
    anchor_id = min(forward_ids) - 1
    audit_folds = [f for f in folds if anchor_id <= f["fold"] <= max(forward_ids)]
    snapshots = pd.read_csv(args.snapshots, parse_dates=["decision_time"])
    snapshots = snapshots[snapshots["fold"].isin([f["fold"] for f in audit_folds])]
    returns_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()

    summary_rows: list[dict] = []
    fold_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    configs = make_configs()
    for number, config in enumerate(configs, start=1):
        returns, diagnostics, selections = run_configuration(
            config, snapshots, returns_panel, audit_folds,
        )
        forward_returns = returns[returns["fold"].isin(forward_ids)].copy()
        for method, group in forward_returns.groupby("method"):
            weights = selections[
                selections["fold"].isin(forward_ids)
                & selections["method"].eq(method)
            ]
            positive = weights[weights["weight"] > 0]
            summary_rows.append({
                **asdict(config),
                "method": method,
                **financial_metrics(group.sort_values("date")["return"]),
                "mean_max_weight": float(positive.groupby("fold")["weight"].max().mean()),
                "mean_effective_names": float(
                    positive.groupby("fold")["weight"].apply(lambda w: 1.0 / np.square(w).sum()).mean()
                ),
            })
        per_fold = (
            forward_returns.groupby(["config_id", "fold", "method"])["return"]
            .apply(lambda r: float((1.0 + r).prod() - 1.0))
            .rename("fold_return")
            .reset_index()
        )
        fold_rows.append(per_fold)
        diagnostic_rows.append(diagnostics[diagnostics["fold"].isin(forward_ids)])
        selection_rows.append(selections[selections["fold"].isin(forward_ids)])
        print(f"audited {number:02d}/{len(configs):02d}: {config.config_id}", flush=True)

    summary = pd.DataFrame(summary_rows)
    fold_table = pd.concat(fold_rows, ignore_index=True)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    selections = pd.concat(selection_rows, ignore_index=True)
    summary.to_csv(output / "preexisting_grid_forward_summary.csv", index=False)
    fold_table.to_csv(output / "preexisting_grid_fold_returns.csv", index=False)
    diagnostics.to_csv(output / "preexisting_grid_fold_diagnostics.csv", index=False)
    selections.to_csv(output / "preexisting_grid_selections.csv", index=False)

    # Signal audit: every score is known at the prior decision time; realised
    # returns are measured only over that fold's subsequent test month.
    signal_rows: list[dict] = []
    for fold in audit_folds:
        if fold["fold"] not in forward_ids:
            continue
        snap = snapshots[snapshots["fold"].eq(fold["fold"])].copy()
        realised = returns_panel.loc[
            (returns_panel.index >= fold["test_start"])
            & (returns_panel.index < fold["test_end"])
        ]
        realised = (1.0 + realised).prod(axis=0, skipna=True) - 1.0
        snap["realised_return"] = snap["ticker"].map(realised)
        snap["blend_signal"] = 0.70 * snap["xgb_signal"] + 0.30 * snap["momentum_signal"]
        usable = snap.dropna(subset=["realised_return"])
        row = {"fold": fold["fold"], "test_start": fold["test_start"], "test_end": fold["test_end"]}
        for signal in ("xgb_signal", "momentum_signal", "blend_signal"):
            result = stats.spearmanr(usable[signal], usable["realised_return"], nan_policy="omit")
            row[f"{signal}_rank_ic"] = float(result.statistic)
            row[f"{signal}_pvalue"] = float(result.pvalue)
        signal_rows.append(row)
    signal_audit = pd.DataFrame(signal_rows)
    signal_audit.to_csv(output / "forward_signal_rank_ic.csv", index=False)

    # Aggregate by the constraints that can directly limit a single-name shock.
    constraint_summary = (
        summary.groupby(["candidate_size", "portfolio_cardinality", "weight_upper", "weight_mode", "method"])
        .agg(
            specifications=("config_id", "nunique"),
            median_cumulative_return=("cumulative_return", "median"),
            best_cumulative_return=("cumulative_return", "max"),
            median_sharpe=("sharpe_zero_rf", "median"),
            median_drawdown=("maximum_drawdown", "median"),
        )
        .reset_index()
        .sort_values(["method", "median_sharpe"], ascending=[True, False])
    )
    constraint_summary.to_csv(output / "constraint_group_summary.csv", index=False)

    best = (
        summary.sort_values(["method", "sharpe_zero_rf"], ascending=[True, False])
        .groupby("method", as_index=False)
        .head(10)
    )
    best.to_csv(output / "top10_posthoc_by_method.csv", index=False)
    stable = (
        summary.pivot_table(index="config_id", columns="method", values=["cumulative_return", "sharpe_zero_rf", "maximum_drawdown"])
    )
    stable.columns = [f"{metric}_{method}" for metric, method in stable.columns]
    stable = stable.reset_index()
    stable["worst_method_sharpe"] = stable[["sharpe_zero_rf_AUR", "sharpe_zero_rf_QAUR"]].min(axis=1)
    stable["worst_method_return"] = stable[["cumulative_return_AUR", "cumulative_return_QAUR"]].min(axis=1)
    stable = stable.sort_values(["worst_method_sharpe", "worst_method_return"], ascending=False)
    stable.to_csv(output / "robust_across_methods_ranking.csv", index=False)

    manifest = {
        "status": "posthoc_diagnostic_only",
        "forward_fold_ids": forward_ids,
        "number_of_preexisting_configurations": len(configs),
        "selection_after_observing_2026_is_not_forward_evidence": True,
        "purpose": "diagnose constraint, allocation and signal failure",
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = f"""# 2026 pre-existing grid audit (post-hoc diagnostic)

This audit evaluates specifications that were already encoded before the 2026
result was inspected. Choosing a winner from this table is nevertheless a
post-selection decision and is **not** a new forward test.

## Most robust across AUR and QAUR

{stable.head(15).to_markdown(index=False)}

## Signal rank IC by forward fold

{signal_audit.to_markdown(index=False)}

## Top configurations by method

{best[["config_id", "family", "method", "candidate_size", "portfolio_cardinality", "weight_upper", "weight_mode", "cumulative_return", "sharpe_zero_rf", "maximum_drawdown", "mean_max_weight", "mean_effective_names"]].to_markdown(index=False)}

The next strategy specification must be frozen before collecting a new, unseen
paper-trading period. These tables can diagnose and design that specification,
but cannot establish live readiness or quantum advantage.
"""
    (output / "PREEXISTING_GRID_AUDIT.md").write_text(report, encoding="utf-8")
    print("\nMost robust configurations:\n", stable.head(10).to_string(index=False), flush=True)
    print("\nForward signal audit:\n", signal_audit.to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line diagnostics
        print(f"audit failed: {exc}", file=sys.stderr)
        raise
