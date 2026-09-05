from __future__ import annotations

"""Sensitivity analysis for a common trailing-market regime gate."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from run_constraint_strategy_search import build_features, financial_metrics, load_market_data, make_folds


CORE_COLUMNS = ["date", "ticker", "adjusted_close", "volume", "trading_value"]


def sample_of(fold: int) -> str:
    if fold <= 28:
        return "development"
    if fold <= 43:
        return "historical_holdout"
    if fold == 44:
        return "bridge"
    return "observed_2026"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--forward-prices", type=Path, required=True)
    parser.add_argument("--candidate-returns", type=Path, required=True)
    parser.add_argument("--august-returns", type=Path, required=True)
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
    market = features.pivot(index="date", columns="ticker", values="return_1d").mean(axis=1).sort_index()
    folds = make_folds(features["date"])
    previous = folds[-1]
    folds.append({
        key: (value + pd.DateOffset(months=1) if key != "fold" else int(value) + 1)
        for key, value in previous.items()
    })
    fold_map = {f["fold"]: f for f in folds}

    candidate = pd.read_csv(args.candidate_returns, parse_dates=["date"])
    candidate = candidate[candidate["config_id"].eq("C1_IV_X")]
    august = pd.read_csv(args.august_returns, parse_dates=["date"])
    candidate = (
        pd.concat([candidate, august], ignore_index=True)
        .drop_duplicates(["fold", "date", "method"], keep="last")
        .sort_values(["method", "fold", "date"])
    )

    lookbacks = (5, 10, 15, 20, 30, 40, 60, 90, 120, 160, 200, 252)
    thresholds = (-0.02, 0.0, 0.02)
    switching_costs = (0.0, 25.0, 50.0, 100.0)
    metric_rows: list[dict] = []
    exposure_rows: list[dict] = []
    return_store: dict[str, pd.DataFrame] = {}
    for lookback, threshold, cost_bps in itertools.product(lookbacks, thresholds, switching_costs):
        policy_id = f"MG_L{lookback}_T{int(round(threshold*100)):d}_C{int(cost_bps)}"
        parts: list[pd.DataFrame] = []
        for method in ("AUR", "QAUR"):
            previous_exposure = 0.0
            method_data = candidate[candidate["method"].eq(method)]
            for fold_id in sorted(method_data["fold"].unique()):
                current = method_data[method_data["fold"].eq(fold_id)].copy()
                test_start = fold_map[fold_id]["test_start"]
                history = market.loc[market.index < test_start].tail(lookback)
                growth = float((1.0 + history.fillna(0.0)).prod() - 1.0)
                exposure = float(growth > threshold)
                current["return"] *= exposure
                if len(current):
                    current.loc[current.index[0], "return"] -= abs(exposure - previous_exposure) * cost_bps / 10000.0
                current["policy_id"] = policy_id
                current["exposure"] = exposure
                parts.append(current)
                exposure_rows.append({
                    "policy_id": policy_id,
                    "lookback": lookback,
                    "threshold": threshold,
                    "switching_cost_bps": cost_bps,
                    "fold": fold_id,
                    "method": method,
                    "market_growth": growth,
                    "exposure": exposure,
                })
                previous_exposure = exposure
        transformed = pd.concat(parts, ignore_index=True)
        transformed["sample"] = transformed["fold"].map(sample_of)
        for (sample, method), group in transformed.groupby(["sample", "method"]):
            fold_exposure = group.groupby("fold")["exposure"].first()
            metric_rows.append({
                "policy_id": policy_id,
                "lookback": lookback,
                "threshold": threshold,
                "switching_cost_bps": cost_bps,
                "sample": sample,
                "method": method,
                "mean_exposure": float(fold_exposure.mean()),
                **financial_metrics(group.sort_values("date")["return"]),
            })
        return_store[policy_id] = transformed

    metrics = pd.DataFrame(metric_rows)
    exposures = pd.DataFrame(exposure_rows)
    metrics.to_csv(output / "market_gate_sensitivity_metrics.csv", index=False)
    exposures.to_csv(output / "market_gate_sensitivity_exposures.csv", index=False)

    substantive = metrics[metrics["sample"].isin(["development", "historical_holdout", "observed_2026"])]
    ranking = (
        substantive.groupby(["policy_id", "lookback", "threshold", "switching_cost_bps"])
        .agg(
            worst_return=("cumulative_return", "min"),
            worst_sharpe=("sharpe_zero_rf", "min"),
            worst_drawdown=("maximum_drawdown", "min"),
            minimum_exposure=("mean_exposure", "min"),
            median_return=("cumulative_return", "median"),
        )
        .reset_index()
    )
    ranking["stable"] = (
        (ranking["worst_return"] > 0)
        & (ranking["worst_drawdown"] >= -0.20)
        & (ranking["minimum_exposure"] >= 0.40)
    )
    ranking = ranking.sort_values(["stable", "worst_sharpe", "worst_return"], ascending=False)
    ranking.to_csv(output / "market_gate_robust_ranking.csv", index=False)

    # Use the strongest stable policy, but also expose the zero-threshold,
    # 25-bps neighbourhood so a narrow optimum cannot be hidden.
    winner_id = str(ranking.iloc[0]["policy_id"])
    winner_metrics = metrics[metrics["policy_id"].eq(winner_id)]
    winner_metrics.to_csv(output / "selected_market_gate_metrics.csv", index=False)
    return_store[winner_id].to_csv(output / "selected_market_gate_returns.csv", index=False)
    neighbourhood = metrics[
        metrics["threshold"].eq(0.0)
        & metrics["switching_cost_bps"].eq(25.0)
        & metrics["sample"].isin(["development", "historical_holdout", "observed_2026"])
    ]
    neighbourhood.to_csv(output / "zero_threshold_lookback_neighbourhood.csv", index=False)

    latest_growth = {
        str(lookback): float((1.0 + market.tail(lookback).fillna(0.0)).prod() - 1.0)
        for lookback in lookbacks
    }
    (output / "latest_market_growth.json").write_text(
        pd.Series(latest_growth).to_json(indent=2), encoding="utf-8"
    )
    report = f"""# Market-gate sensitivity audit

## Strongest stable policies

{ranking.head(20).to_markdown(index=False)}

## Selected sensitivity winner

{winner_metrics.to_markdown(index=False)}

## Zero-threshold lookback neighbourhood

{neighbourhood.to_markdown(index=False)}

This is a post-hoc sensitivity study. It is used to choose a simple protocol for
the next unseen paper window, not to create a fresh historical claim.
"""
    (output / "MARKET_GATE_SENSITIVITY.md").write_text(report, encoding="utf-8")
    print("\nRobust leaders:\n", ranking.head(15).to_string(index=False), flush=True)
    print("\nSelected metrics:\n", winner_metrics.to_string(index=False), flush=True)
    print("\nLatest growth:\n", pd.Series(latest_growth).to_string(), flush=True)


if __name__ == "__main__":
    main()
