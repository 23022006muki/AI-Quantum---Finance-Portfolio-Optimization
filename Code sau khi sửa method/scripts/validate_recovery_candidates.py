from __future__ import annotations

"""Robustness audit for recovery candidates designed after observing 2026.

All outputs are research diagnostics.  They define what may be frozen for the
next unseen paper-trading window; they do not convert 2026 into forward proof.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from run_constraint_strategy_search import (
    StrategyConfig,
    build_features,
    financial_metrics,
    load_market_data,
    make_configs,
    make_folds,
    run_configuration,
)


CORE_COLUMNS = ["date", "ticker", "adjusted_close", "volume", "trading_value"]
BASE_SEED = 20260829


def sample_name(fold: int) -> str:
    if fold <= 28:
        return "development_2022_2024"
    if fold <= 43:
        return "historical_holdout_2024_2025"
    if fold == 44:
        return "bridge_december_2025"
    return "observed_forward_2026"


def summarize(returns: pd.DataFrame, label: str, extra: dict | None = None) -> list[dict]:
    rows: list[dict] = []
    tagged = returns.copy()
    tagged["sample"] = tagged["fold"].map(sample_name)
    for (sample, method), group in tagged.groupby(["sample", "method"]):
        row = {"run": label, "sample": sample, "method": method}
        if extra:
            row.update(extra)
        row.update(financial_metrics(group.sort_values("date")["return"]))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--forward-prices", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    base, _, _ = load_market_data(args.base_dataset)
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
    return_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    folds = make_folds(features["date"])
    snapshots = pd.read_csv(args.snapshots, parse_dates=["decision_time"])

    existing = {c.config_id: c for c in make_configs()}
    base_candidates = [existing[name] for name in ("C1_IV_X", "C2_IV_X", "C3_IV_X")]
    candidates: list[StrategyConfig] = []
    for config in base_candidates:
        candidates.append(config)
        candidates.append(replace(config, config_id=config.config_id + "_QAWS", qa_warm_start=True))

    candidate_summary_rows: list[dict] = []
    candidate_return_parts: list[pd.DataFrame] = []
    candidate_selection_parts: list[pd.DataFrame] = []
    for config in candidates:
        returns, diagnostics, selections = run_configuration(config, snapshots, return_panel, folds)
        candidate_summary_rows.extend(summarize(returns, config.config_id, {"cost_bps": config.transaction_cost_bps}))
        candidate_return_parts.append(returns)
        candidate_selection_parts.append(selections)
        print(f"candidate complete: {config.config_id}", flush=True)

    candidate_summary = pd.DataFrame(candidate_summary_rows)
    candidate_returns = pd.concat(candidate_return_parts, ignore_index=True)
    candidate_selections = pd.concat(candidate_selection_parts, ignore_index=True)
    candidate_summary.to_csv(output / "candidate_period_summary.csv", index=False)
    candidate_returns.to_csv(output / "candidate_returns.csv", index=False)
    candidate_selections.to_csv(output / "candidate_selections.csv", index=False)

    # Choose the research candidate with the highest worst-method Sharpe in the
    # observed 2026 diagnostic period. This is explicitly post-hoc.
    observed = candidate_summary[candidate_summary["sample"].eq("observed_forward_2026")]
    robust = (
        observed.groupby("run")
        .agg(
            worst_method_sharpe=("sharpe_zero_rf", "min"),
            worst_method_return=("cumulative_return", "min"),
            worst_method_drawdown=("maximum_drawdown", "min"),
        )
        .sort_values(["worst_method_sharpe", "worst_method_return"], ascending=False)
        .reset_index()
    )
    research_candidate_id = str(robust.iloc[0]["run"])
    research_candidate = next(c for c in candidates if c.config_id == research_candidate_id)
    robust.to_csv(output / "candidate_robust_ranking.csv", index=False)

    cost_rows: list[dict] = []
    cost_returns: list[pd.DataFrame] = []
    for cost_bps in (0.0, 25.0, 50.0, 75.0, 100.0):
        config = replace(
            research_candidate,
            config_id=f"{research_candidate_id}_COST{int(cost_bps)}",
            transaction_cost_bps=cost_bps,
        )
        returns, _, _ = run_configuration(config, snapshots, return_panel, folds)
        cost_rows.extend(summarize(returns, config.config_id, {"cost_bps": cost_bps}))
        cost_returns.append(returns)
        print(f"cost stress complete: {cost_bps:.0f} bps", flush=True)
    cost_table = pd.DataFrame(cost_rows)
    cost_table.to_csv(output / "candidate_cost_stress.csv", index=False)
    pd.concat(cost_returns, ignore_index=True).to_csv(output / "candidate_cost_returns.csv", index=False)

    seed_rows: list[dict] = []
    for seed in (7, 29, 101, 1009, BASE_SEED):
        returns, _, _ = run_configuration(research_candidate, snapshots, return_panel, folds, qa_seed=seed)
        forward_returns = returns[
            returns["fold"].map(sample_name).eq("observed_forward_2026")
            & returns["method"].eq("QAUR")
        ]
        seed_rows.append({"seed": seed, **financial_metrics(forward_returns.sort_values("date")["return"])})
        print(f"seed stress complete: {seed}", flush=True)
    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(output / "candidate_qaur_seed_stress.csv", index=False)

    # Rolling six-fold metrics expose regime dependence hidden by aggregate rows.
    rolling_rows: list[dict] = []
    candidate_data = candidate_returns[candidate_returns["config_id"].eq(research_candidate_id)]
    fold_ids = sorted(candidate_data["fold"].unique())
    for method in ("AUR", "QAUR"):
        method_data = candidate_data[candidate_data["method"].eq(method)]
        for end_index in range(5, len(fold_ids)):
            window = fold_ids[end_index - 5:end_index + 1]
            group = method_data[method_data["fold"].isin(window)].sort_values("date")
            rolling_rows.append({
                "method": method,
                "start_fold": window[0],
                "end_fold": window[-1],
                "end_date": group["date"].max(),
                **financial_metrics(group["return"]),
            })
    rolling = pd.DataFrame(rolling_rows)
    rolling.to_csv(output / "candidate_rolling_six_fold_metrics.csv", index=False)

    latest_fold = max(fold_ids)
    latest = candidate_selections[
        candidate_selections["config_id"].eq(research_candidate_id)
        & candidate_selections["fold"].eq(latest_fold)
    ].copy()
    latest.to_csv(output / "research_candidate_latest_portfolio.csv", index=False)

    stability = (
        candidate_summary[candidate_summary["run"].eq(research_candidate_id)]
        .pivot(index="sample", columns="method", values=["cumulative_return", "sharpe_zero_rf", "maximum_drawdown"])
    )
    stability.columns = [f"{metric}_{method}" for metric, method in stability.columns]
    stability = stability.reset_index()
    stability.to_csv(output / "research_candidate_time_stability.csv", index=False)

    manifest = {
        "status": "posthoc_recovery_candidate_validation",
        "research_candidate": asdict(research_candidate),
        "selection_uses_observed_2026": True,
        "live_capital_authorized": False,
        "quantum_advantage_claimed": False,
        "next_step": "freeze candidate and collect a new untouched paper-trading window",
    }
    (output / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = f"""# Recovery candidate robustness audit

The candidate below was selected after inspecting 2026 and therefore defines a
future protocol; it is not evidence from an untouched test.

## Candidate ranking on observed 2026

{robust.to_markdown(index=False)}

## Selected candidate across time regimes

{stability.to_markdown(index=False)}

## Transaction-cost stress

{cost_table[cost_table["sample"].eq("observed_forward_2026")].to_markdown(index=False)}

## QAUR seed stress on observed 2026

{seed_table.to_markdown(index=False)}

## Latest component portfolio

{latest.to_markdown(index=False)}

Because selection used observed 2026, the only valid next confirmation is a new
untouched paper-trading period with frozen code, data policy and parameters.
"""
    (output / "RECOVERY_CANDIDATE_REPORT.md").write_text(report, encoding="utf-8")
    print("\nCandidate ranking:\n", robust.to_string(index=False), flush=True)
    print("\nSelected candidate time stability:\n", stability.to_string(index=False), flush=True)
    print("\nSeed stress:\n", seed_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
