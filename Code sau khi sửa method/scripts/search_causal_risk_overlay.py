from __future__ import annotations

"""Search a causal, common risk overlay for the frozen C1_IV_X branches.

The overlay is applied identically after AUR/QAUR portfolio allocation. Every
fold uses only information strictly prior to its test start. Hyperparameters are
ranked on the original development folds; later periods are reported separately.
"""

import argparse
import itertools
import json
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
    parser.add_argument("--historical-diagnostics", type=Path, required=True)
    parser.add_argument("--forward-diagnostics", type=Path, required=True)
    parser.add_argument("--august-diagnostics", type=Path, required=True)
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
    august_fold = {
        key: (value + pd.DateOffset(months=1) if key != "fold" else int(value) + 1)
        for key, value in previous.items()
    }
    folds = folds + [august_fold]
    fold_map = {fold["fold"]: fold for fold in folds}

    candidate = pd.read_csv(args.candidate_returns, parse_dates=["date"])
    candidate = candidate[candidate["config_id"].eq("C1_IV_X")].copy()
    august_path = args.output.parent / "august_2026_quasi_holdout_20260829" / "august_quasi_holdout_returns.csv"
    if august_path.exists():
        august_returns = pd.read_csv(august_path, parse_dates=["date"])
        candidate = pd.concat([candidate, august_returns], ignore_index=True)
    candidate = candidate.drop_duplicates(["fold", "date", "method"], keep="last")

    diagnostics = pd.concat([
        pd.read_csv(args.historical_diagnostics),
        pd.read_csv(args.forward_diagnostics),
        pd.read_csv(args.august_diagnostics),
    ], ignore_index=True).drop_duplicates("fold", keep="last")
    validation_ic = diagnostics.set_index("fold")["validation_rank_ic"].to_dict()

    specifications: list[dict] = []
    spec_id = 0
    for market_lookback, strategy_lookback, vol_target, vol_lookback, ic_threshold in itertools.product(
        (0, 20, 60, 120, 200),
        (0, 3, 6),
        (0.0, 0.10, 0.12, 0.15),
        (60, 120),
        (-1.0, 0.0, 0.05, 0.10),
    ):
        # Volatility lookback has no effect without a target; retain one copy.
        if vol_target == 0 and vol_lookback != 60:
            continue
        spec_id += 1
        specifications.append({
            "spec_id": f"OV{spec_id:04d}",
            "market_lookback": market_lookback,
            "strategy_lookback_folds": strategy_lookback,
            "volatility_target": vol_target,
            "volatility_lookback_days": vol_lookback,
            "minimum_validation_ic": ic_threshold,
            "overlay_turnover_cost_bps": 25.0,
        })

    metric_rows: list[dict] = []
    exposure_rows: list[dict] = []
    selected_return_parts: dict[str, pd.DataFrame] = {}
    for specification in specifications:
        output_parts: list[pd.DataFrame] = []
        for method in ("AUR", "QAUR"):
            method_data = candidate[candidate["method"].eq(method)].sort_values(["fold", "date"])
            previous_exposure = 0.0
            for fold_id in sorted(method_data["fold"].unique()):
                current = method_data[method_data["fold"].eq(fold_id)].copy()
                if current.empty or fold_id not in fold_map:
                    continue
                test_start = fold_map[fold_id]["test_start"]
                risk_on = True
                market_growth = np.nan
                if specification["market_lookback"] > 0:
                    history = market.loc[market.index < test_start].tail(specification["market_lookback"])
                    market_growth = float((1.0 + history.fillna(0.0)).prod() - 1.0)
                    risk_on = risk_on and market_growth > 0.0
                strategy_growth = np.nan
                if specification["strategy_lookback_folds"] > 0:
                    prior_ids = [i for i in sorted(method_data["fold"].unique()) if i < fold_id]
                    prior_ids = prior_ids[-specification["strategy_lookback_folds"]:]
                    history = method_data[method_data["fold"].isin(prior_ids)]["return"]
                    if prior_ids:
                        strategy_growth = float((1.0 + history).prod() - 1.0)
                        risk_on = risk_on and strategy_growth > 0.0
                ic = float(validation_ic.get(fold_id, np.nan))
                if specification["minimum_validation_ic"] >= 0:
                    risk_on = risk_on and np.isfinite(ic) and ic >= specification["minimum_validation_ic"]

                exposure = 1.0 if risk_on else 0.0
                estimated_vol = np.nan
                if risk_on and specification["volatility_target"] > 0:
                    history = method_data[
                        (method_data["date"] < test_start)
                    ].sort_values("date").tail(specification["volatility_lookback_days"])["return"]
                    if len(history) >= 20:
                        estimated_vol = float(history.std(ddof=1) * np.sqrt(252))
                        exposure = min(1.0, specification["volatility_target"] / max(estimated_vol, 1e-9))
                current["return"] = current["return"] * exposure
                if len(current):
                    current.loc[current.index[0], "return"] -= (
                        abs(exposure - previous_exposure)
                        * specification["overlay_turnover_cost_bps"] / 10000.0
                    )
                current["spec_id"] = specification["spec_id"]
                current["exposure"] = exposure
                output_parts.append(current)
                exposure_rows.append({
                    **specification,
                    "fold": fold_id,
                    "method": method,
                    "exposure": exposure,
                    "market_growth": market_growth,
                    "strategy_growth": strategy_growth,
                    "validation_ic": ic,
                    "estimated_volatility": estimated_vol,
                })
                previous_exposure = exposure
        transformed = pd.concat(output_parts, ignore_index=True)
        transformed["sample"] = transformed["fold"].map(sample_of)
        for (sample, method), group in transformed.groupby(["sample", "method"]):
            exposure = group.groupby("fold")["exposure"].first()
            metric_rows.append({
                **specification,
                "sample": sample,
                "method": method,
                "mean_exposure": float(exposure.mean()),
                "active_folds": int((exposure > 0).sum()),
                "total_folds": int(len(exposure)),
                **financial_metrics(group.sort_values("date")["return"]),
            })
        selected_return_parts[specification["spec_id"]] = transformed

    metrics = pd.DataFrame(metric_rows)
    exposures = pd.DataFrame(exposure_rows)
    metrics.to_csv(output / "causal_overlay_grid_metrics.csv", index=False)
    exposures.to_csv(output / "causal_overlay_grid_exposures.csv", index=False)

    development = metrics[metrics["sample"].eq("development")]
    ranking = (
        development.groupby("spec_id")
        .agg(
            worst_method_sharpe=("sharpe_zero_rf", "min"),
            worst_method_return=("cumulative_return", "min"),
            worst_method_drawdown=("maximum_drawdown", "min"),
            minimum_exposure=("mean_exposure", "min"),
        )
        .reset_index()
    )
    definitions = pd.DataFrame(specifications)
    ranking = ranking.merge(definitions, on="spec_id", how="left")
    ranking["eligible"] = (
        (ranking["minimum_exposure"] >= 0.40)
        & (ranking["worst_method_drawdown"] >= -0.25)
        & (ranking["worst_method_return"] > 0.0)
    )
    ranking = ranking.sort_values(
        ["eligible", "worst_method_sharpe", "worst_method_return"], ascending=False
    )
    ranking.to_csv(output / "development_policy_ranking.csv", index=False)
    winner_id = str(ranking.iloc[0]["spec_id"])
    winner_definition = definitions[definitions["spec_id"].eq(winner_id)].iloc[0].to_dict()
    winner_metrics = metrics[metrics["spec_id"].eq(winner_id)].copy()
    winner_metrics.to_csv(output / "selected_policy_all_periods.csv", index=False)
    selected_return_parts[winner_id].to_csv(output / "selected_policy_returns.csv", index=False)
    exposures[exposures["spec_id"].eq(winner_id)].to_csv(output / "selected_policy_exposures.csv", index=False)

    # Post-hoc stability diagnostic: policies positive for both methods in every
    # substantive sample. It is not used as the formal development selection.
    substantive = metrics[metrics["sample"].isin(["development", "historical_holdout", "observed_2026"])]
    stability = (
        substantive.groupby("spec_id")
        .agg(
            worst_segment_return=("cumulative_return", "min"),
            worst_segment_sharpe=("sharpe_zero_rf", "min"),
            worst_segment_drawdown=("maximum_drawdown", "min"),
            minimum_segment_exposure=("mean_exposure", "min"),
        )
        .reset_index()
        .merge(definitions, on="spec_id", how="left")
        .sort_values(["worst_segment_sharpe", "worst_segment_return"], ascending=False)
    )
    stability.to_csv(output / "posthoc_cross_regime_stability.csv", index=False)

    manifest = {
        "status": "causal_overlay_research_search",
        "base_candidate": "C1_IV_X",
        "specifications_tested": len(specifications),
        "formal_selection_sample": "development folds 0-28",
        "selected_policy": winner_definition,
        "selection_after_2026_is_research_only": True,
        "common_overlay_for_aur_and_qaur": True,
        "live_capital_authorized": False,
        "quantum_advantage_claimed": False,
    }
    (output / "overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = f"""# Causal common risk-overlay search

The same overlay is applied after both AUR and QAUR portfolio allocations. All
fold decisions use only prior information. Hyperparameters are ranked on the
original development period; later periods are kept as separate diagnostics.

## Development-selected policy

{pd.DataFrame([winner_definition]).to_markdown(index=False)}

## Selected policy across periods

{winner_metrics.to_markdown(index=False)}

## Strongest post-hoc cross-regime policies

{stability.head(15).to_markdown(index=False)}

The cross-regime table is diagnostic because 2026 had already been observed.
Only a newly frozen future paper window can confirm the selected method.
"""
    (output / "CAUSAL_OVERLAY_REPORT.md").write_text(report, encoding="utf-8")
    print(f"tested {len(specifications)} policies", flush=True)
    print("\nDevelopment-selected policy:\n", pd.DataFrame([winner_definition]).to_string(index=False), flush=True)
    print("\nSelected policy metrics:\n", winner_metrics.to_string(index=False), flush=True)
    print("\nCross-regime leaders:\n", stability.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
