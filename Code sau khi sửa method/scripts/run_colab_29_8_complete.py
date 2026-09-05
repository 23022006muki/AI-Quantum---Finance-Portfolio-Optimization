from __future__ import annotations

"""Complete 29/8 experiment orchestrator used locally and in Google Colab.

The module deliberately reports two evidence layers:

1. confirmatory historical evidence: configuration selection on folds 0--28 and
   testing on untouched folds 29--43;
2. practical method-design evidence: broader constraint/allocation/gate search
   using all information observed by 29 August 2026.  This layer may select a
   paper-trading protocol, but it is never relabelled as prospective evidence.

All AUR/QAUR comparisons use the same downstream portfolio QUBO, exact
fixed-cardinality screening reference, weight allocator, transaction costs and
market gate.  QAUR remains a classical surrogate for a quantum-ready QUBO.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_constraint_strategy_search import (
    SEED,
    StrategyConfig,
    build_features,
    build_fold_cache,
    exact_cardinality_qubo,
    ewma_covariance,
    financial_metrics,
    load_market_data,
    make_configs,
    make_folds,
    paired_tests,
    run_configuration,
    xy_qaoa_statevector_audit,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_sample(fold: int) -> str:
    if fold <= 28:
        return "development_2022_2024"
    if fold <= 43:
        return "historical_holdout_2024_2025"
    if fold == 44:
        return "bridge_december_2025"
    return "observed_2026"


def add_august_and_prospective_folds(complete: list[dict]) -> list[dict]:
    previous = complete[-1]
    august = {
        key: (value + pd.DateOffset(months=1) if key != "fold" else int(value) + 1)
        for key, value in previous.items()
    }
    september = {
        key: (value + pd.DateOffset(months=1) if key != "fold" else int(value) + 1)
        for key, value in august.items()
    }
    if august["test_start"] != pd.Timestamp("2026-08-02"):
        raise RuntimeError(f"Unexpected August fold: {august}")
    if september["test_start"] != pd.Timestamp("2026-09-02"):
        raise RuntimeError(f"Unexpected September fold: {september}")
    return complete + [august, september]


def apply_market_gate(
    returns: pd.DataFrame,
    folds: list[dict],
    market: pd.Series,
    lookback: int,
    switching_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one common, causal gate to both reducer branches."""
    fold_map = {int(fold["fold"]): fold for fold in folds}
    output: list[pd.DataFrame] = []
    exposure_rows: list[dict] = []
    previous_exposure = 0.0
    for fold_id in sorted(returns["fold"].unique()):
        if int(fold_id) not in fold_map:
            continue
        fold = fold_map[int(fold_id)]
        decision_time = pd.Timestamp(fold["test_start"]) - pd.Timedelta(days=1)
        if lookback <= 0:
            growth, exposure = np.nan, 1.0
        else:
            trailing = market.loc[market.index <= decision_time].tail(lookback).fillna(0.0)
            growth = float((1.0 + trailing).prod() - 1.0)
            exposure = float(growth > 0.0)
        chunk = returns[returns["fold"].eq(fold_id)].copy()
        chunk["return"] *= exposure
        if len(chunk) and exposure != previous_exposure:
            first_date = chunk["date"].min()
            first_rows = chunk["date"].eq(first_date)
            chunk.loc[first_rows, "return"] -= switching_cost_bps / 10000.0
        chunk["market_gate_exposure"] = exposure
        chunk["market_gate_growth"] = growth
        chunk["market_gate_lookback"] = lookback
        output.append(chunk)
        exposure_rows.append({
            "fold": int(fold_id),
            "decision_time": decision_time,
            "market_growth": growth,
            "exposure": exposure,
            "lookback": lookback,
        })
        previous_exposure = exposure
    return pd.concat(output, ignore_index=True), pd.DataFrame(exposure_rows)


def summarize_periods(returns: pd.DataFrame, config_id: str, gate: int) -> list[dict]:
    tagged = returns.copy()
    tagged["sample"] = tagged["fold"].astype(int).map(observed_sample)
    rows: list[dict] = []
    for (sample, method), group in tagged.groupby(["sample", "method"]):
        if sample == "bridge_december_2025":
            continue
        rows.append({
            "config_id": config_id,
            "market_gate_lookback": gate,
            "sample": sample,
            "method": method,
            **financial_metrics(group.sort_values("date")["return"]),
        })
    return rows


def robust_candidate_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby(["config_id", "market_gate_lookback"])
    ranking = grouped.agg(
        worst_return=("cumulative_return", "min"),
        worst_sharpe=("sharpe_zero_rf", "min"),
        mean_sharpe=("sharpe_zero_rf", "mean"),
        worst_drawdown=("maximum_drawdown", "min"),
        positive_cells=("cumulative_return", lambda x: int((x > 0).sum())),
        total_cells=("cumulative_return", "size"),
    ).reset_index()
    ranking["passes_positive_all"] = ranking["positive_cells"].eq(ranking["total_cells"])
    ranking["passes_drawdown_20pct"] = ranking["worst_drawdown"].ge(-0.20)
    ranking["practical_gate_passed"] = (
        ranking["passes_positive_all"] & ranking["passes_drawdown_20pct"]
    )
    ranking = ranking.sort_values(
        ["practical_gate_passed", "worst_sharpe", "worst_return", "mean_sharpe"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ranking


def block_bootstrap_mean_difference(
    difference: pd.Series,
    seed: int = 20260829,
    block_length: int = 20,
    repetitions: int = 5000,
) -> dict:
    values = pd.Series(difference).dropna().to_numpy(float)
    if len(values) < 2:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "pvalue": np.nan}
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block_length + 1))
    boot = np.empty(repetitions)
    for b in range(repetitions):
        pieces: list[np.ndarray] = []
        while sum(len(piece) for piece in pieces) < len(values):
            start = int(rng.choice(starts))
            pieces.append(values[start:start + block_length])
        boot[b] = np.concatenate(pieces)[:len(values)].mean()
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "pvalue": float((np.sum(boot <= 0.0) + 1) / (repetitions + 1)),
    }


def practical_h4_by_period(returns: pd.DataFrame) -> pd.DataFrame:
    tagged = returns.copy()
    tagged["sample"] = tagged["fold"].astype(int).map(observed_sample)
    rows: list[dict] = []
    for sample, sample_returns in tagged.groupby("sample"):
        if sample == "bridge_december_2025":
            continue
        wide = sample_returns.pivot(index="date", columns="method", values="return").dropna()
        difference = wide["QAUR"] - wide["AUR"]
        test = stats.ttest_1samp(difference, 0.0, alternative="greater")
        bootstrap = block_bootstrap_mean_difference(difference)
        rows.append({
            "sample": sample,
            "observations": len(difference),
            "mean_daily_difference": float(difference.mean()),
            "paired_t_statistic": float(test.statistic),
            "paired_t_pvalue_one_sided": float(test.pvalue),
            "block_bootstrap_ci_low": bootstrap["ci_low"],
            "block_bootstrap_ci_high": bootstrap["ci_high"],
            "block_bootstrap_pvalue_one_sided": bootstrap["pvalue"],
            "supported_5pct": bool(test.pvalue < 0.05 and bootstrap["pvalue"] < 0.05),
            "evidence_label": "posthoc_method_design_not_confirmatory",
        })
    return pd.DataFrame(rows)


def holm_adjust(pvalues: pd.Series) -> np.ndarray:
    """Holm family-wise-error adjustment without an extra dependency."""
    values = pd.Series(pvalues, dtype=float).to_numpy()
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def practical_positive_return_evidence(returns: pd.DataFrame) -> pd.DataFrame:
    """Separate positive realised P&L from statistical evidence of mean > 0."""
    tagged = returns.copy()
    tagged["sample"] = tagged["fold"].astype(int).map(observed_sample)
    rows: list[dict] = []
    for (sample, method), group in tagged.groupby(["sample", "method"]):
        if sample == "bridge_december_2025":
            continue
        daily = group.sort_values("date")["return"].dropna()
        test = stats.ttest_1samp(daily, 0.0, alternative="greater")
        bootstrap = block_bootstrap_mean_difference(daily)
        rows.append({
            "sample": sample,
            "method": method,
            "observations": len(daily),
            "mean_daily_return": float(daily.mean()),
            "cumulative_return": float((1.0 + daily).prod() - 1.0),
            "one_sample_t_pvalue": float(test.pvalue),
            "block_bootstrap_ci_low": bootstrap["ci_low"],
            "block_bootstrap_ci_high": bootstrap["ci_high"],
            "block_bootstrap_pvalue": bootstrap["pvalue"],
        })
    table = pd.DataFrame(rows)
    table["combined_conservative_pvalue"] = table[
        ["one_sample_t_pvalue", "block_bootstrap_pvalue"]
    ].max(axis=1)
    table["holm_adjusted_pvalue"] = holm_adjust(
        table["combined_conservative_pvalue"]
    )
    table["positive_economically"] = table["cumulative_return"].gt(0.0)
    table["positive_mean_supported_holm_5pct"] = table[
        "holm_adjusted_pvalue"
    ].lt(0.05)
    table["evidence_label"] = "posthoc_method_design_not_confirmatory"
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-cache", type=Path)
    parser.add_argument("--max-practical-configs", type=int, default=0)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    forecast_dir = output / "forecast_cache"
    forecast_dir.mkdir(exist_ok=True)

    prices, security, benchmark = load_market_data(dataset)
    features = build_features(prices)
    complete_folds = make_folds(features["date"])
    all_folds = add_august_and_prospective_folds(complete_folds)
    observed_folds = all_folds[:-1]
    if args.snapshot_cache:
        cache = args.snapshot_cache.resolve()
        snapshots = pd.read_csv(cache / "forecast_snapshots.csv", parse_dates=["decision_time"])
        forecast_diagnostics = pd.read_csv(cache / "forecast_diagnostics.csv", parse_dates=["decision_time"])
    else:
        snapshots, forecast_diagnostics = build_fold_cache(
            features, security, all_folds, forecast_dir
        )
    snapshots = snapshots.drop(columns=["validation_rank_ic"], errors="ignore").merge(
        forecast_diagnostics[["fold", "validation_rank_ic"]], on="fold", how="left"
    )
    return_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    market = return_panel.mean(axis=1).sort_index()

    # Confirmatory layer: exactly the 2020--2025 experiment and original split.
    historical_folds = [fold for fold in complete_folds if int(fold["fold"]) <= 43]
    confirmatory_configs = make_configs()
    confirmatory_returns: dict[str, pd.DataFrame] = {}
    confirmatory_diagnostics: list[pd.DataFrame] = []
    confirmatory_selections: dict[str, pd.DataFrame] = {}
    confirmatory_summary: list[dict] = []
    for index, config in enumerate(confirmatory_configs, 1):
        returns, diagnostics, selections = run_configuration(
            config, snapshots, return_panel, historical_folds
        )
        confirmatory_returns[config.config_id] = returns
        confirmatory_selections[config.config_id] = selections
        confirmatory_diagnostics.append(diagnostics)
        for sample, keep in (
            ("development", returns["fold"].le(28)),
            ("holdout", returns["fold"].between(29, 43)),
        ):
            for method, group in returns[keep].groupby("method"):
                confirmatory_summary.append({
                    "config_id": config.config_id,
                    "sample": sample,
                    "method": method,
                    **financial_metrics(group.sort_values("date")["return"]),
                })
        print(f"confirmatory {index:02d}/{len(confirmatory_configs)} {config.config_id}", flush=True)
    confirmatory_summary_table = pd.DataFrame(confirmatory_summary)
    development = confirmatory_summary_table[
        confirmatory_summary_table["sample"].eq("development")
    ].pivot(index="config_id", columns="method", values="sharpe_zero_rf")
    development["selection_score"] = development[["AUR", "QAUR"]].mean(axis=1)
    confirmatory_best_id = str(development["selection_score"].idxmax())
    confirmatory_fold_table = pd.concat(confirmatory_diagnostics, ignore_index=True)
    confirmatory_best_folds = confirmatory_fold_table[
        confirmatory_fold_table["config_id"].eq(confirmatory_best_id)
    ]
    confirmatory_tests = paired_tests(
        confirmatory_returns[confirmatory_best_id], confirmatory_best_folds, 28
    )
    confirmatory_best_config = next(
        config for config in confirmatory_configs if config.config_id == confirmatory_best_id
    )
    confirmatory_best_returns = confirmatory_returns[confirmatory_best_id].copy()
    confirmatory_best_selections = confirmatory_selections[confirmatory_best_id].copy()

    # Classical baselines are evaluated on exactly the same untouched holdout
    # dates as the locked AUR/QAUR configuration.  These rows are kept separate
    # from H4 because the confirmatory paired test remains QAUR minus AUR.
    holdout_start = historical_folds[29]["test_start"]
    holdout_end = historical_folds[-1]["test_end"]
    full_universe_ew = return_panel.loc[
        (return_panel.index >= holdout_start) & (return_panel.index < holdout_end)
    ].mean(axis=1)
    benchmark_holdout = benchmark.set_index("date")["return"].loc[
        lambda values: (values.index >= holdout_start) & (values.index < holdout_end)
    ].dropna()
    confirmatory_holdout_baselines = pd.DataFrame([
        {"method": "FULL_UNIVERSE_EW", **financial_metrics(full_universe_ew)},
        {"method": "VNALLSHARE_TRI", **financial_metrics(benchmark_holdout)},
    ])
    confirmatory_seed_rows: list[dict] = []
    for qa_seed in (7, 42, 99):
        seed_returns, seed_diagnostics, _ = run_configuration(
            confirmatory_best_config, snapshots, return_panel, historical_folds,
            qa_seed=qa_seed,
        )
        holdout = seed_returns[seed_returns["fold"].between(29, 43)]
        seed_perf = {
            method: financial_metrics(group.sort_values("date")["return"])
            for method, group in holdout.groupby("method")
        }
        objective = seed_diagnostics.pivot(
            index="fold", columns="method", values="reduction_objective"
        ).dropna()
        confirmatory_seed_rows.append({
            "qa_seed": qa_seed,
            "aur_sharpe": seed_perf["AUR"]["sharpe_zero_rf"],
            "qaur_sharpe": seed_perf["QAUR"]["sharpe_zero_rf"],
            "qaur_minus_aur_sharpe": (
                seed_perf["QAUR"]["sharpe_zero_rf"]
                - seed_perf["AUR"]["sharpe_zero_rf"]
            ),
            "mean_qaur_objective_advantage": float(
                (objective["QAUR"] - objective["AUR"]).mean()
            ),
        })
    confirmatory_seed_robustness = pd.DataFrame(confirmatory_seed_rows)
    h5_supported = bool(
        (confirmatory_seed_robustness["qaur_minus_aur_sharpe"] > 0).all()
    )
    confirmatory_tests = pd.concat([
        confirmatory_tests,
        pd.DataFrame([{
            "hypothesis": "H5_QAUR_financial_direction_robust_across_seeds",
            "estimate": float(
                (confirmatory_seed_robustness["qaur_minus_aur_sharpe"] > 0).mean()
            ),
            "statistic": np.nan,
            "pvalue_one_sided": np.nan,
            "supported_5pct": h5_supported,
        }]),
    ], ignore_index=True)
    confirmatory_tests["holm_adjusted_pvalue"] = np.nan
    inferential = confirmatory_tests["pvalue_one_sided"].notna()
    confirmatory_tests.loc[inferential, "holm_adjusted_pvalue"] = holm_adjust(
        confirmatory_tests.loc[inferential, "pvalue_one_sided"]
    )
    confirmatory_tests["supported_holm_5pct"] = (
        confirmatory_tests["holm_adjusted_pvalue"].lt(0.05)
    )
    confirmatory_tests["evidence_label"] = "confirmatory_untouched_historical_holdout"

    # Shared downstream XY-QAOA audit.  The same circuit depth, optimisation
    # budget and number of shots are used for the AUR and QAUR candidate sets.
    # The exact fixed-cardinality solution remains the reference used to
    # calculate success probability and optimality gap.
    confirmatory_xy_rows: list[dict] = []
    for fold in historical_folds:
        if int(fold["fold"]) <= 28:
            continue
        fold_id = int(fold["fold"])
        snapshot = snapshots[snapshots["fold"].eq(fold_id)].copy()
        decision_time = pd.Timestamp(snapshot["decision_time"].iloc[0])
        for method in ("AUR", "QAUR"):
            candidates = confirmatory_selections[confirmatory_best_id].loc[
                lambda frame: frame["fold"].eq(fold_id)
                & frame["method"].eq(method),
                "ticker",
            ].tolist()
            candidate_snapshot = snapshot.set_index("ticker").reindex(candidates)
            mu = (
                confirmatory_best_config.signal_blend
                * candidate_snapshot["xgb_signal"]
                + (1.0 - confirmatory_best_config.signal_blend)
                * candidate_snapshot["momentum_signal"]
            ).to_numpy(float)
            cov = ewma_covariance(
                return_panel,
                candidates,
                decision_time,
                confirmatory_best_config.covariance_span,
                confirmatory_best_config.covariance_shrinkage,
            )
            _, q_matrix = exact_cardinality_qubo(
                mu,
                cov,
                confirmatory_best_config.portfolio_cardinality,
                confirmatory_best_config.risk_aversion_qubo,
            )
            audit = xy_qaoa_statevector_audit(
                q_matrix,
                confirmatory_best_config.portfolio_cardinality,
                SEED + fold_id,
            )
            confirmatory_xy_rows.append({
                "fold": fold_id,
                "method": method,
                "candidate_size": len(candidates),
                "portfolio_cardinality": confirmatory_best_config.portfolio_cardinality,
                "depth": 2,
                "budget": 30,
                "shots": 1024,
                **audit,
            })
    confirmatory_xy_audit = pd.DataFrame(confirmatory_xy_rows)

    # Practical method-design layer: constraint/allocation families plus a
    # common market gate.  This uses observed 2026 and is explicitly post-hoc.
    practical_configs = [
        config for config in make_configs()
        if config.family == "constraint_and_allocation"
    ]
    if args.max_practical_configs > 0:
        practical_configs = practical_configs[:args.max_practical_configs]
    practical_return_map: dict[tuple[str, int], pd.DataFrame] = {}
    practical_selection_map: dict[str, pd.DataFrame] = {}
    practical_diagnostic_map: dict[str, pd.DataFrame] = {}
    practical_summary_rows: list[dict] = []
    exposure_parts: list[pd.DataFrame] = []
    for index, config in enumerate(practical_configs, 1):
        base_returns, diagnostics, selections = run_configuration(
            config, snapshots, return_panel, observed_folds
        )
        practical_selection_map[config.config_id] = selections
        practical_diagnostic_map[config.config_id] = diagnostics
        for gate in (0, 20, 30, 40):
            gated, exposures = apply_market_gate(
                base_returns, observed_folds, market, gate,
                switching_cost_bps=25.0 if gate > 0 else 0.0,
            )
            practical_return_map[(config.config_id, gate)] = gated
            practical_summary_rows.extend(summarize_periods(gated, config.config_id, gate))
            exposures["config_id"] = config.config_id
            exposure_parts.append(exposures)
        print(f"practical {index:02d}/{len(practical_configs)} {config.config_id}", flush=True)

    practical_summary = pd.DataFrame(practical_summary_rows)
    practical_ranking = robust_candidate_ranking(practical_summary)
    if not practical_ranking["practical_gate_passed"].any():
        raise RuntimeError("No practical configuration passed positive-return and drawdown gates.")
    selected_row = practical_ranking[practical_ranking["practical_gate_passed"]].iloc[0]
    practical_best_id = str(selected_row["config_id"])
    practical_best_gate = int(selected_row["market_gate_lookback"])
    practical_best_config = next(c for c in practical_configs if c.config_id == practical_best_id)
    practical_best_returns = practical_return_map[(practical_best_id, practical_best_gate)]
    exploratory_h4 = practical_h4_by_period(practical_best_returns)
    positive_return_evidence = practical_positive_return_evidence(
        practical_best_returns
    )

    # Seed robustness for the selected practical base configuration.
    seed_rows: list[dict] = []
    for qa_seed in (7, 29, 101, 1009, 20260829):
        base_returns, diagnostics, _ = run_configuration(
            practical_best_config, snapshots, return_panel, observed_folds, qa_seed=qa_seed
        )
        gated, _ = apply_market_gate(
            base_returns, observed_folds, market, practical_best_gate,
            25.0 if practical_best_gate > 0 else 0.0,
        )
        observed = gated[gated["fold"].ge(45)]
        perf = {
            method: financial_metrics(group.sort_values("date")["return"])
            for method, group in observed.groupby("method")
        }
        pivot = diagnostics.pivot(index="fold", columns="method", values="reduction_objective").dropna()
        seed_rows.append({
            "qa_seed": qa_seed,
            "aur_sharpe": perf["AUR"]["sharpe_zero_rf"],
            "qaur_sharpe": perf["QAUR"]["sharpe_zero_rf"],
            "qaur_minus_aur_sharpe": perf["QAUR"]["sharpe_zero_rf"] - perf["AUR"]["sharpe_zero_rf"],
            "mean_qaur_objective_advantage": float((pivot["QAUR"] - pivot["AUR"]).mean()),
        })
    seed_robustness = pd.DataFrame(seed_rows)

    # Generate the September shadow basket using all past states, then apply the
    # selected common gate.  The prospective fold has no future returns yet.
    _, prospective_diagnostics, prospective_selections = run_configuration(
        practical_best_config, snapshots, return_panel, all_folds
    )
    prospective_fold = int(all_folds[-1]["fold"])
    shadow = prospective_selections[
        prospective_selections["fold"].eq(prospective_fold)
        & prospective_selections["selected_downstream"]
    ].copy()
    decision_time = pd.Timestamp(snapshots[snapshots["fold"].eq(prospective_fold)]["decision_time"].iloc[0])
    if practical_best_gate > 0:
        trailing = market.loc[market.index <= decision_time].tail(practical_best_gate).fillna(0.0)
        current_growth = float((1 + trailing).prod() - 1)
        current_exposure = float(current_growth > 0)
    else:
        current_growth, current_exposure = np.nan, 1.0
    shadow["shadow_weight"] = shadow["weight"]
    shadow["executable_weight"] = shadow["shadow_weight"] * current_exposure
    shadow["cash_weight"] = 1.0 - current_exposure
    shadow["market_growth"] = current_growth
    shadow["market_gate_exposure"] = current_exposure

    # Preserve the complete prospective Top-K sets, not only the four selected
    # assets.  The explanatory Colab cells merge these rows with the live
    # forecast/risk snapshot so that the final basket is auditable from source
    # signals rather than entered manually.
    prospective_candidates = prospective_selections[
        prospective_selections["fold"].eq(prospective_fold)
    ].copy()
    prospective_snapshot = snapshots[snapshots["fold"].eq(prospective_fold)][[
        "ticker", "decision_time", "xgb_signal", "momentum_signal",
        "liquidity_20d", "volatility_20d",
    ]].copy()
    prospective_candidates = prospective_candidates.merge(
        prospective_snapshot, on="ticker", how="left", validate="many_to_one"
    )
    prospective_candidates["shadow_weight"] = prospective_candidates["weight"]
    prospective_candidates["executable_weight"] = (
        prospective_candidates["shadow_weight"] * current_exposure
    )
    prospective_candidates["cash_weight"] = 1.0 - current_exposure
    prospective_candidates["market_growth"] = current_growth
    prospective_candidates["market_gate_exposure"] = current_exposure

    # Export every table required for paper reporting and reproducibility.
    confirmatory_summary_table.to_csv(output / "confirmatory_configuration_results.csv", index=False)
    confirmatory_tests.to_csv(output / "confirmatory_hypothesis_tests.csv", index=False)
    confirmatory_seed_robustness.to_csv(
        output / "confirmatory_seed_robustness.csv", index=False
    )
    confirmatory_xy_audit.to_csv(
        output / "confirmatory_xy_qaoa_holdout_audit.csv", index=False
    )
    confirmatory_best_returns.to_csv(
        output / "confirmatory_best_returns.csv", index=False
    )
    confirmatory_best_folds.to_csv(
        output / "confirmatory_best_fold_diagnostics.csv", index=False
    )
    confirmatory_best_selections.to_csv(
        output / "confirmatory_best_selections.csv", index=False
    )
    confirmatory_holdout_baselines.to_csv(
        output / "confirmatory_holdout_baselines.csv", index=False
    )
    practical_summary.to_csv(output / "practical_configuration_period_results.csv", index=False)
    practical_ranking.to_csv(output / "practical_robust_ranking.csv", index=False)
    practical_best_returns.to_csv(output / "selected_practical_returns.csv", index=False)
    exploratory_h4.to_csv(output / "selected_practical_h4_by_period.csv", index=False)
    positive_return_evidence.to_csv(
        output / "selected_practical_positive_return_evidence.csv", index=False
    )
    seed_robustness.to_csv(output / "selected_practical_seed_robustness.csv", index=False)
    practical_diagnostic_map[practical_best_id].to_csv(
        output / "selected_practical_fold_diagnostics.csv", index=False
    )
    practical_selection_map[practical_best_id].to_csv(
        output / "selected_practical_selections.csv", index=False
    )
    pd.concat(exposure_parts, ignore_index=True).to_csv(output / "market_gate_exposures.csv", index=False)
    shadow.to_csv(output / "september_2026_shadow_and_executable_basket.csv", index=False)
    prospective_candidates.to_csv(
        output / "september_2026_candidate_audit.csv", index=False
    )
    forecast_diagnostics.to_csv(output / "forecast_diagnostics.csv", index=False)
    snapshots.to_csv(output / "forecast_snapshots.csv", index=False)

    selected_periods = practical_summary[
        practical_summary["config_id"].eq(practical_best_id)
        & practical_summary["market_gate_lookback"].eq(practical_best_gate)
    ].copy()
    selected_periods.to_csv(output / "selected_practical_period_results.csv", index=False)
    pd.DataFrame(all_folds).to_csv(output / "walk_forward_fold_manifest.csv", index=False)
    pd.DataFrame([asdict(config) for config in confirmatory_configs]).to_csv(
        output / "confirmatory_configuration_definitions.csv", index=False
    )
    pd.DataFrame([asdict(config) for config in practical_configs]).to_csv(
        output / "practical_configuration_definitions.csv", index=False
    )
    manifest = {
        "dataset": dataset.name,
        "dataset_sha256": sha256_file(dataset),
        "observed_price_end": str(features["date"].max().date()),
        "confirmatory_fold_split": {"development": "0-28", "holdout": "29-43"},
        "confirmatory_configurations_screened": len(confirmatory_configs),
        "confirmatory_best_config": confirmatory_best_id,
        "practical_evidence_label": "posthoc_method_design_using_data_observed_by_2026_08_29",
        "practical_configurations_screened": len(practical_configs),
        "market_gate_lookbacks_screened": [0, 20, 30, 40],
        "practical_best_config": asdict(practical_best_config),
        "practical_best_market_gate_lookback": practical_best_gate,
        "practical_selection_rule": "positive return in all 3 periods x 2 reducers; MDD >= -20%; maximize worst Sharpe",
        "prospective_protocol_start": "2026-09-02",
        "current_market_growth": current_growth,
        "current_exposure": current_exposure,
        "live_capital_authorized": False,
        "quantum_advantage_claimed": False,
        "qa_backend": "classical cardinality-preserving surrogate for quantum-ready QUBO",
        "shared_downstream_screening": "exact fixed-cardinality QUBO reference",
        "xy_qaoa_holdout_audit_instances": int(len(confirmatory_xy_audit)),
        "xy_qaoa_mean_feasibility_rate": float(
            confirmatory_xy_audit["feasibility_rate"].mean()
        ),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# Kết quả tối ưu hóa thực tiễn — bản dữ liệu 29/8

## Lớp xác nhận lịch sử

- Cấu hình chọn chỉ từ development: `{confirmatory_best_id}`.
- Holdout: folds 29--43.

{confirmatory_tests.to_markdown(index=False)}

XY-QAOA holdout audit ({len(confirmatory_xy_audit)} instances): mean feasibility
rate = {confirmatory_xy_audit["feasibility_rate"].mean():.4f}, mean optimality gap
= {confirmatory_xy_audit["optimality_gap"].mean():.6f}.

## Lớp thiết kế phương pháp thực tiễn

- Cấu hình: `{practical_best_id}`.
- Common market gate: {practical_best_gate} phiên.
- Nhãn bằng chứng: **post-hoc method design**, chưa phải prospective proof.

{selected_periods.to_markdown(index=False)}

## H4 theo từng giai đoạn

{exploratory_h4.to_markdown(index=False)}

## Lợi nhuận dương: hiệu quả kinh tế và ý nghĩa thống kê

{positive_return_evidence.to_markdown(index=False)}

## Rổ tháng 9/2026

{shadow[["method", "ticker", "shadow_weight", "executable_weight", "cash_weight", "market_growth"]].to_markdown(index=False)}

## Kết luận hợp lệ

Phương pháp được phép chuyển sang paper trading không vốn từ 02/09/2026 nếu giữ
nguyên tham số. Không có kết quả nào trong lớp practical được diễn giải thành
quantum advantage hoặc cho phép triển khai vốn thật.
"""
    (output / "FINAL_RESULTS_29_8_VI.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print(selected_periods.to_string(index=False), flush=True)
    print(shadow[["method", "ticker", "shadow_weight", "executable_weight"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
