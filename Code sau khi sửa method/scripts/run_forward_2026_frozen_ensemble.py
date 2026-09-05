from __future__ import annotations

"""One-shot 2026 out-of-time evaluation of the frozen prequential ensemble."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
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
from run_live_readiness_quantum_benchmark import (
    annual_sharpe,
    deflated_sharpe_probability,
    moving_block_bootstrap_difference,
    prequential_configuration_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--forward-prices", type=Path, required=True)
    parser.add_argument("--historical-snapshots", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--warm-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    base_prices, security, _ = load_market_data(args.base_dataset)
    forward = pd.read_parquet(args.forward_prices).copy()
    forward["date"] = pd.to_datetime(forward["date"], errors="coerce")
    for column in ("adjusted_close", "volume", "trading_value"):
        forward[column] = pd.to_numeric(forward[column], errors="coerce")
    required = ["date", "ticker", "adjusted_close", "volume", "trading_value"]
    forward = forward.dropna(subset=required)
    # The historical research file also contains metadata columns that are not
    # present in the provisional CafeF panel. Feature engineering only needs
    # the five fields below, so align explicitly instead of silently fabricating
    # metadata or requiring identical source schemas.
    base_core = base_prices[required].copy()
    forward_core = forward[required].copy()
    combined_prices = (
        pd.concat([base_core, forward_core], ignore_index=True)
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    features = build_features(combined_prices)
    folds = make_folds(features["date"])
    genuine_forward_folds = [fold for fold in folds if fold["test_start"] >= pd.Timestamp("2026-01-01")]
    if not genuine_forward_folds:
        raise RuntimeError("No complete 2026 forward fold is available.")

    historical_snapshots = pd.read_csv(args.historical_snapshots, parse_dates=["decision_time"])
    historical_fold_ids = set(historical_snapshots["fold"].astype(int))
    missing_folds = [fold for fold in folds if fold["fold"] not in historical_fold_ids]
    new_snapshots, new_diagnostics = build_fold_cache(features, security, missing_folds, output)
    snapshots = pd.concat([historical_snapshots, new_snapshots], ignore_index=True)
    snapshots = snapshots[snapshots["fold"].isin([fold["fold"] for fold in folds])]
    snapshots.to_csv(output / "all_forecast_snapshots.csv", index=False)
    new_diagnostics.to_csv(output / "forward_forecast_diagnostics.csv", index=False)
    return_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()

    warm_configs = [config for config in make_configs() if config.config_id.startswith("W_")]
    all_returns: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []
    for config in warm_configs:
        returns, diagnostics, selections = run_configuration(
            config, snapshots, return_panel, folds,
        )
        all_returns.append(returns)
        all_diagnostics.append(diagnostics)
        all_selections.append(selections)
        print(f"forward rerun completed: {config.config_id}", flush=True)
    returns_table = pd.concat(all_returns, ignore_index=True)
    diagnostics_table = pd.concat(all_diagnostics, ignore_index=True)
    selections_table = pd.concat(all_selections, ignore_index=True)
    returns_table.to_csv(output / "all_configuration_returns.csv", index=False)
    diagnostics_table.to_csv(output / "all_fold_diagnostics.csv", index=False)
    selections_table.to_csv(output / "all_configuration_selections.csv", index=False)

    lookbacks = (6, 9, 12, 18, 24)
    ensemble_parts: list[pd.DataFrame] = []
    choice_parts: list[pd.DataFrame] = []
    for lookback in lookbacks:
        selected, choices = prequential_configuration_selection(returns_table, folds, lookback)
        selected["lookback_folds"] = lookback
        choices["lookback_folds"] = lookback
        ensemble_parts.append(selected)
        choice_parts.append(choices)
    choices_table = pd.concat(choice_parts, ignore_index=True)
    choices_table.to_csv(output / "prequential_choices.csv", index=False)
    ensemble = (
        pd.concat(ensemble_parts, ignore_index=True)
        .groupby(["fold", "date", "method"], as_index=False)["return"].mean()
    )
    ensemble.to_csv(output / "frozen_ensemble_returns.csv", index=False)
    forward_fold_ids = [fold["fold"] for fold in genuine_forward_folds]
    forward_ensemble = ensemble[ensemble["fold"].isin(forward_fold_ids)].copy()
    forward_ensemble.to_csv(output / "forward_2026_returns.csv", index=False)

    summary = pd.DataFrame([
        {"method": method, **financial_metrics(group.sort_values("date")["return"])}
        for method, group in forward_ensemble.groupby("method")
    ])
    summary.to_csv(output / "forward_2026_summary.csv", index=False)
    full_ew = return_panel.loc[
        (return_panel.index >= genuine_forward_folds[0]["test_start"])
        & (return_panel.index < genuine_forward_folds[-1]["test_end"])
    ].mean(axis=1)
    full_summary = pd.DataFrame([{"method": "FULL_UNIVERSE_EW", **financial_metrics(full_ew)}])
    full_summary.to_csv(output / "forward_2026_baseline.csv", index=False)

    # Conservative multiple-testing reference from every previously completed config.
    trial_frames = []
    for directory in (args.phase1_dir, args.warm_dir, args.overlay_dir):
        frame = pd.read_csv(Path(directory) / "configuration_results.csv")
        trial_frames.append(frame[frame["sample"].eq("development")][["config_id", "method", "sharpe_zero_rf"]])
    trials = pd.concat(trial_frames, ignore_index=True).drop_duplicates(["config_id", "method"])
    dsr_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    for method, group in forward_ensemble.groupby("method"):
        strategy = group.set_index("date")["return"].sort_index()
        dsr_rows.append({
            "method": method,
            **deflated_sharpe_probability(strategy, trials["sharpe_zero_rf"], int(trials["config_id"].nunique()) + 1),
        })
        bootstrap_rows.append({
            "method": method,
            "comparator": "FULL_UNIVERSE_EW",
            **moving_block_bootstrap_difference(strategy, full_ew, block_length=10, seed=20260829),
        })
    dsr = pd.DataFrame(dsr_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    dsr.to_csv(output / "forward_2026_deflated_sharpe.csv", index=False)
    bootstrap.to_csv(output / "forward_2026_bootstrap.csv", index=False)

    last_fold = max(forward_fold_ids)
    final_choices = choices_table[choices_table["fold"].eq(last_fold)]
    chosen_configs = final_choices["selected_config"].value_counts().rename_axis("config_id").reset_index(name="votes")
    chosen_configs.to_csv(output / "latest_configuration_votes.csv", index=False)
    latest_selections = selections_table[
        selections_table["fold"].eq(last_fold)
        & selections_table["config_id"].isin(chosen_configs["config_id"])
    ].copy()
    latest_selections.to_csv(output / "latest_component_portfolios.csv", index=False)

    gates = pd.DataFrame([
        {"gate": "forward_return_positive_both", "passed": bool((summary["cumulative_return"] > 0).all())},
        {"gate": "forward_sharpe_above_1_both", "passed": bool((summary["sharpe_zero_rf"] > 1).all())},
        {"gate": "forward_drawdown_below_20pct", "passed": bool((summary["maximum_drawdown"] >= -0.20).all())},
        {"gate": "forward_DSR_probability_above_95pct", "passed": bool((dsr["deflated_sharpe_probability"] >= 0.95).all())},
        {"gate": "significant_forward_excess_vs_full_EW", "passed": bool((bootstrap["pvalue_one_sided_positive"] < 0.05).all())},
        {"gate": "minimum_24_month_forward_record", "passed": False},
        {"gate": "cross_source_and_corporate_action_certified", "passed": False},
    ])
    gates.to_csv(output / "forward_2026_gates.csv", index=False)

    manifest = {
        "status": "one_shot_frozen_forward_evaluation",
        "forward_fold_ids": forward_fold_ids,
        "forward_test_start": str(genuine_forward_folds[0]["test_start"].date()),
        "forward_test_end": str(genuine_forward_folds[-1]["test_end"].date()),
        "forward_sessions": int(forward_ensemble["date"].nunique()),
        "configurations": [asdict(config) for config in warm_configs],
        "lookbacks": list(lookbacks),
        "parameters_retuned_on_2026": False,
        "gates_passed": int(gates["passed"].sum()),
        "gates_total": len(gates),
        "limitations": [
            "CafeF forward price panel is provisional and not cross-source certified",
            "2026 period becomes observed after this one-shot evaluation and must not be retuned",
            "VNAllshare TRI forward benchmark is not yet available in the local audited dataset",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# Frozen 2026 forward evaluation

{summary.to_markdown(index=False)}

## Full-universe baseline

{full_summary.to_markdown(index=False)}

## Deflated Sharpe

{dsr.to_markdown(index=False)}

## Bootstrap versus Full-Universe EW

{bootstrap.to_markdown(index=False)}

## Gates

{gates.to_markdown(index=False)}

No parameter was retuned on the 2026 period. The source panel remains provisional and cannot authorize live capital by itself.
"""
    (output / "FORWARD_2026_REPORT.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(full_summary.to_string(index=False), flush=True)
    print(gates.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
