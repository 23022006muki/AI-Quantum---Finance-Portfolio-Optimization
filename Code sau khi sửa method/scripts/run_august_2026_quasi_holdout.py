from __future__ import annotations

"""Evaluate the frozen recovery candidate on the previously unscored August fold.

The data already existed locally, so this is a quasi-holdout rather than a
prospective paper-trading record.  Candidate selection only used complete folds
through 2026-07-31 and does not use the results produced by this script.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from run_constraint_strategy_search import (
    build_features,
    build_fold_cache,
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
    parser.add_argument("--historical-snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    base, security, _ = load_market_data(args.base_dataset)
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
    complete_folds = make_folds(features["date"])
    previous = complete_folds[-1]
    partial_fold = {
        key: (value + pd.DateOffset(months=1) if key != "fold" else int(value) + 1)
        for key, value in previous.items()
    }
    if partial_fold["test_start"] != pd.Timestamp("2026-08-02"):
        raise RuntimeError(f"Unexpected partial-fold start: {partial_fold['test_start']}")
    new_snapshot, diagnostics = build_fold_cache(features, security, [partial_fold], output)
    historical = pd.read_csv(args.historical_snapshots, parse_dates=["decision_time"])
    snapshots = (
        pd.concat([historical, new_snapshot], ignore_index=True)
        .drop_duplicates(["fold", "ticker"], keep="last")
    )
    all_folds = complete_folds + [partial_fold]
    return_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    config = next(config for config in make_configs() if config.config_id == "C1_IV_X")
    returns, fold_diagnostics, selections = run_configuration(config, snapshots, return_panel, all_folds)
    partial_returns = returns[returns["fold"].eq(partial_fold["fold"])].copy()
    partial_selections = selections[selections["fold"].eq(partial_fold["fold"])].copy()
    partial_returns.to_csv(output / "august_quasi_holdout_returns.csv", index=False)
    partial_selections.to_csv(output / "august_quasi_holdout_portfolio.csv", index=False)
    fold_diagnostics[fold_diagnostics["fold"].eq(partial_fold["fold"])].to_csv(
        output / "august_quasi_holdout_diagnostics.csv", index=False
    )
    diagnostics.to_csv(output / "august_forecast_diagnostics.csv", index=False)

    summary = pd.DataFrame([
        {"method": method, **financial_metrics(group.sort_values("date")["return"])}
        for method, group in partial_returns.groupby("method")
    ])
    full_ew = return_panel.loc[
        (return_panel.index >= partial_fold["test_start"])
        & (return_panel.index <= return_panel.index.max())
    ].mean(axis=1)
    baseline = pd.DataFrame([{"method": "FULL_UNIVERSE_EW", **financial_metrics(full_ew)}])
    summary.to_csv(output / "august_quasi_holdout_summary.csv", index=False)
    baseline.to_csv(output / "august_quasi_holdout_baseline.csv", index=False)
    full_ew.rename("return").reset_index().to_csv(output / "august_full_ew_returns.csv", index=False)

    manifest = {
        "status": "quasi_holdout_not_prospective",
        "candidate_frozen_before_scoring": asdict(config),
        "candidate_selection_cutoff": "2026-07-31",
        "partial_fold": {key: str(value) for key, value in partial_fold.items()},
        "observed_test_end": str(partial_returns["date"].max().date()),
        "sessions": int(partial_returns["date"].nunique()),
        "parameters_retuned_on_august": False,
        "live_capital_authorized": False,
    }
    (output / "quasi_holdout_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = f"""# August 2026 quasi-holdout

The candidate was selected using complete folds only through 2026-07-31. August
returns were not used to select or tune it. Because the raw August data already
existed before this run, this remains a quasi-holdout rather than prospective
paper trading.

## Candidate

{summary.to_markdown(index=False)}

## Full-universe equal-weight baseline

{baseline.to_markdown(index=False)}

## Portfolio decided before the August test interval

{partial_selections.to_markdown(index=False)}
"""
    (output / "AUGUST_QUASI_HOLDOUT_REPORT.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(baseline.to_string(index=False), flush=True)
    print(partial_selections.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
