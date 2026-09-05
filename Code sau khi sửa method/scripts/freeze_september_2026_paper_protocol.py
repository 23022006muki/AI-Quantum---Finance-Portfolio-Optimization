from __future__ import annotations

"""Freeze the first genuinely prospective paper-trading portfolio and protocol."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from run_constraint_strategy_search import (
    build_features,
    build_fold_cache,
    load_market_data,
    make_configs,
    make_folds,
    run_configuration,
)


CORE_COLUMNS = ["date", "ticker", "adjusted_close", "volume", "trading_value"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--forward-prices", type=Path, required=True)
    parser.add_argument("--historical-snapshots", type=Path, required=True)
    parser.add_argument("--strategy-script", type=Path, required=True)
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
    partial_august = {
        key: (value + pd.DateOffset(months=1) if key != "fold" else int(value) + 1)
        for key, value in previous.items()
    }
    prospective_september = {
        key: (value + pd.DateOffset(months=1) if key != "fold" else int(value) + 1)
        for key, value in partial_august.items()
    }
    if prospective_september["test_start"] != pd.Timestamp("2026-09-02"):
        raise RuntimeError(f"Unexpected prospective start: {prospective_september['test_start']}")
    new_snapshots, forecast_diagnostics = build_fold_cache(
        features, security, [partial_august, prospective_september], output
    )
    historical = pd.read_csv(args.historical_snapshots, parse_dates=["decision_time"])
    snapshots = (
        pd.concat([historical, new_snapshots], ignore_index=True)
        .drop_duplicates(["fold", "ticker"], keep="last")
    )
    return_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    config = next(config for config in make_configs() if config.config_id == "C1_IV_X")
    all_folds = complete_folds + [partial_august, prospective_september]
    _, diagnostics, selections = run_configuration(config, snapshots, return_panel, all_folds)
    target = selections[selections["fold"].eq(prospective_september["fold"])].copy()
    target = target[target["selected_downstream"]].sort_values(["method", "weight"], ascending=[True, False])
    target.to_csv(output / "september_2026_target_portfolios.csv", index=False)
    forecast_diagnostics.to_csv(output / "paper_forecast_diagnostics.csv", index=False)
    diagnostics[diagnostics["fold"].eq(prospective_september["fold"])].to_csv(
        output / "paper_reduction_diagnostics.csv", index=False
    )

    a_set = set(target[target["method"].eq("AUR")]["ticker"])
    q_set = set(target[target["method"].eq("QAUR")]["ticker"])
    jaccard = len(a_set & q_set) / len(a_set | q_set) if a_set | q_set else 1.0
    lock = {
        "status": "FROZEN_PROSPECTIVE_PAPER_ONLY",
        "frozen_at_local_date": "2026-08-29",
        "first_test_interval_start": str(prospective_september["test_start"].date()),
        "first_test_interval_end_exclusive": str(prospective_september["test_end"].date()),
        "decision_time": str(new_snapshots[new_snapshots["fold"].eq(prospective_september["fold"])]["decision_time"].iloc[0].date()),
        "candidate": asdict(config),
        "candidate_selected_after_observing_jan_jul_2026": True,
        "august_quasi_holdout_not_counted_as_prospective": True,
        "no_parameter_retuning_during_paper_window": True,
        "source_hashes": {
            "base_dataset_sha256": sha256(args.base_dataset.resolve()),
            "forward_prices_sha256": sha256(args.forward_prices.resolve()),
            "strategy_script_sha256": sha256(args.strategy_script.resolve()),
        },
        "aur_qaur_target_jaccard": jaccard,
        "transaction_cost_bps": config.transaction_cost_bps,
        "live_capital_authorized": False,
        "quantum_advantage_claimed": False,
        "operational_rules": [
            "rebalance monthly using only information available before the test interval",
            "record intended and executed prices, commissions, taxes, slippage and rejected orders",
            "quarantine a ticker when price adjustment or corporate action is unresolved",
            "do not trade if cross-source price discrepancy exceeds the documented tolerance",
            "report both AUR and QAUR branches with the same downstream implementation",
        ],
        "promotion_gates": {
            "minimum_prospective_months_for_small_pilot": 12,
            "minimum_prospective_months_for_full_assessment": 24,
            "positive_net_return_after_observed_costs": True,
            "maximum_drawdown_no_worse_than": -0.20,
            "deflated_sharpe_probability_at_least": 0.95,
            "significant_excess_pvalue_below": 0.05,
            "cross_source_and_corporate_action_certification": True,
            "zero_unexplained_data_or_order_audit_breaks": True,
        },
        "quantum_advantage_gates": [
            "run identical QUBO instances on a named physical QPU",
            "compare against exact, multi-start local search and simulated annealing",
            "include queue, embedding, sampling and post-processing time",
            "report objective gap, time-to-solution, feasibility and success probability",
            "demonstrate statistically reproducible advantage as instance size scales",
        ],
    }
    (output / "FROZEN_PROTOCOL.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ledger = pd.DataFrame(columns=[
        "date", "method", "ticker", "target_weight", "executed_weight", "open_price",
        "close_price", "gross_return", "commission", "tax", "slippage", "net_return",
        "data_source_primary", "data_source_secondary", "audit_status", "notes",
    ])
    ledger.to_csv(output / "paper_trading_ledger.csv", index=False)
    report = f"""# Frozen prospective paper-trading protocol

This lock starts with the first available session on or after
**{prospective_september['test_start'].date()}**. Parameters, source policy and
evaluation gates cannot be changed during the paper window.

## September target portfolios

{target.to_markdown(index=False)}

- AUR/QAUR target Jaccard: {jaccard:.4f}
- Declared transaction cost: {config.transaction_cost_bps:.0f} bps per turnover
- Maximum single-name weight: {config.weight_upper:.0%}
- Status: paper only; no real-capital or quantum-advantage authorization

The JSON lock contains file hashes, operational rules and promotion gates. Any
code or data-policy change creates a new protocol version and restarts the
prospective evidence clock for the changed strategy.
"""
    (output / "PAPER_PROTOCOL_README.md").write_text(report, encoding="utf-8")
    print(target.to_string(index=False), flush=True)
    print(json.dumps({"jaccard": jaccard, "status": lock["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
