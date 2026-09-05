from __future__ import annotations

"""Robustness gates for live capital and solver benchmarks for quantum claims.

This script never turns a simulator result into a quantum-advantage claim.  It
creates auditable pass/fail gates and records which evidence is still missing.
"""

import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from run_constraint_strategy_search import (
    StrategyConfig,
    absolute_correlation,
    build_features,
    common_scores,
    financial_metrics,
    load_market_data,
    make_folds,
    reduction_objective,
    xy_qaoa_statevector_audit,
)


SEED = 20260829


def annual_sharpe(r: pd.Series) -> float:
    x = pd.Series(r).dropna()
    std = float(x.std(ddof=1))
    return float(x.mean() / std * np.sqrt(252)) if std > 0 else np.nan


def moving_block_bootstrap_difference(
    strategy: pd.Series,
    comparator: pd.Series,
    block_length: int = 20,
    replications: int = 5000,
    seed: int = SEED,
) -> dict:
    paired = pd.concat([strategy.rename("strategy"), comparator.rename("comparator")], axis=1).dropna()
    difference = (paired["strategy"] - paired["comparator"]).to_numpy(float)
    n = len(difference)
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, n - block_length + 1))
    boot_means = np.empty(replications)
    blocks_needed = int(np.ceil(n / block_length))
    for replication in range(replications):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([difference[start:start + block_length] for start in sampled_starts])[:n]
        boot_means[replication] = sample.mean()
    estimate = float(difference.mean())
    return {
        "observations": n,
        "mean_daily_difference": estimate,
        "annualized_arithmetic_difference": estimate * 252,
        "ci_2_5": float(np.quantile(boot_means, 0.025) * 252),
        "ci_97_5": float(np.quantile(boot_means, 0.975) * 252),
        "pvalue_one_sided_positive": float(np.mean(boot_means <= 0.0)),
    }


def deflated_sharpe_probability(
    returns: pd.Series,
    annual_sharpes_from_trials: pd.Series,
    number_of_trials: int,
) -> dict:
    x = pd.Series(returns).dropna()
    observed_daily = float(x.mean() / x.std(ddof=1))
    trial_daily = pd.Series(annual_sharpes_from_trials).dropna() / np.sqrt(252)
    mean_trial = float(trial_daily.mean())
    std_trial = float(trial_daily.std(ddof=1))
    euler_gamma = 0.5772156649015329
    expected_max = mean_trial + std_trial * (
        (1 - euler_gamma) * stats.norm.ppf(1 - 1 / number_of_trials)
        + euler_gamma * stats.norm.ppf(1 - 1 / (number_of_trials * math.e))
    )
    skew = float(stats.skew(x, bias=False))
    kurtosis = float(stats.kurtosis(x, fisher=False, bias=False))
    denominator = math.sqrt(
        max(1e-12, (1 - skew * observed_daily + (kurtosis - 1) * observed_daily**2 / 4) / (len(x) - 1))
    )
    probability = float(stats.norm.cdf((observed_daily - expected_max) / denominator))
    return {
        "observed_annual_sharpe_arithmetic": observed_daily * np.sqrt(252),
        "expected_max_annual_sharpe_under_trials": expected_max * np.sqrt(252),
        "deflated_sharpe_probability": probability,
        "number_of_trials": number_of_trials,
        "skewness": skew,
        "pearson_kurtosis": kurtosis,
    }


def cost_stress_returns(
    returns: pd.DataFrame,
    fold_diagnostics: pd.DataFrame,
    current_cost_bps: float,
    target_cost_bps: float,
) -> pd.DataFrame:
    stressed = returns.copy()
    extra_cost = (target_cost_bps - current_cost_bps) / 10000.0
    if abs(extra_cost) < 1e-15:
        return stressed
    turnover = fold_diagnostics.set_index(["fold", "method"])["portfolio_turnover"]
    first_rows = stressed.sort_values("date").groupby(["fold", "method"], sort=False).head(1).index
    for index in first_rows:
        key = (int(stressed.loc[index, "fold"]), stressed.loc[index, "method"])
        stressed.loc[index, "return"] -= extra_cost * float(turnover.loc[key])
    return stressed


def prequential_configuration_selection(
    all_returns: pd.DataFrame,
    folds: list[dict],
    lookback_folds: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_returns: list[pd.DataFrame] = []
    choices: list[dict] = []
    for current_fold in range(lookback_folds, len(folds)):
        start_fold = current_fold - lookback_folds
        past = all_returns[(all_returns["fold"] >= start_fold) & (all_returns["fold"] < current_fold)]
        scores = []
        for config_id, config_group in past.groupby("config_id"):
            method_sharpes = [annual_sharpe(group["return"]) for _, group in config_group.groupby("method")]
            scores.append((float(np.nanmean(method_sharpes)), config_id))
        score, chosen_config = max(scores, key=lambda item: (item[0], item[1]))
        current = all_returns[
            (all_returns["fold"].eq(current_fold)) & (all_returns["config_id"].eq(chosen_config))
        ].copy()
        current["selection_score"] = score
        selected_returns.append(current)
        choices.append({"fold": current_fold, "selected_config": chosen_config, "trailing_score": score})
    return pd.concat(selected_returns, ignore_index=True), pd.DataFrame(choices)


def objective_matrix(unary: np.ndarray, corr: np.ndarray, penalty: float) -> np.ndarray:
    return -np.diag(unary) + 0.5 * penalty * corr


def greedy_solution(unary: np.ndarray, corr: np.ndarray, k: int, penalty: float) -> np.ndarray:
    selected: list[int] = []
    remaining = set(range(len(unary)))
    while len(selected) < k:
        best = max(
            remaining,
            key=lambda i: (unary[i] - penalty * corr[i, selected].sum(), -i),
        )
        selected.append(best)
        remaining.remove(best)
    bits = np.zeros(len(unary), dtype=np.int8)
    bits[selected] = 1
    return bits


def best_improving_swap(
    start_bits: np.ndarray,
    unary: np.ndarray,
    corr: np.ndarray,
    penalty: float,
) -> np.ndarray:
    bits = start_bits.copy()
    for _ in range(100):
        selected = np.flatnonzero(bits)
        outside = np.flatnonzero(1 - bits)
        move, best_delta = None, 0.0
        for i in selected:
            retained = selected[selected != i]
            old_pair = corr[i, retained].sum()
            for j in outside:
                delta = unary[j] - unary[i] - penalty * (corr[j, retained].sum() - old_pair)
                if delta > best_delta + 1e-12:
                    best_delta, move = float(delta), (i, j)
        if move is None:
            break
        bits[move[0]] = 0
        bits[move[1]] = 1
    return bits


def simulated_annealing_cardinality(
    unary: np.ndarray,
    corr: np.ndarray,
    k: int,
    penalty: float,
    seed: int,
    steps: int = 12000,
    restarts: int = 4,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    greedy = greedy_solution(unary, corr, k, penalty)
    best_bits = greedy.copy()
    best_value = reduction_objective(best_bits, unary, corr, penalty)
    for restart in range(restarts):
        bits = greedy.copy() if restart == 0 else np.r_[np.ones(k, dtype=np.int8), np.zeros(len(unary) - k, dtype=np.int8)][rng.permutation(len(unary))]
        value = reduction_objective(bits, unary, corr, penalty)
        for step in range(steps):
            inside = np.flatnonzero(bits)
            outside = np.flatnonzero(1 - bits)
            i = int(rng.choice(inside))
            j = int(rng.choice(outside))
            retained = inside[inside != i]
            delta = float(unary[j] - unary[i] - penalty * (corr[j, retained].sum() - corr[i, retained].sum()))
            temperature = 0.15 * (1e-3 / 0.15) ** (step / max(steps - 1, 1))
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-12)):
                bits[i] = 0
                bits[j] = 1
                value += delta
                if value > best_value:
                    best_bits, best_value = bits.copy(), value
    return best_improving_swap(best_bits, unary, corr, penalty)


def multistart_swap_search(
    unary: np.ndarray,
    corr: np.ndarray,
    k: int,
    penalty: float,
    seed: int,
    restarts: int = 6,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = [greedy_solution(unary, corr, k, penalty)]
    for _ in range(max(0, restarts - 1)):
        bits = np.zeros(len(unary), dtype=np.int8)
        bits[rng.choice(len(unary), size=k, replace=False)] = 1
        starts.append(bits)
    solutions = [best_improving_swap(bits, unary, corr, penalty) for bits in starts]
    return max(solutions, key=lambda bits: reduction_objective(bits, unary, corr, penalty))


def exact_solution(unary: np.ndarray, corr: np.ndarray, k: int, penalty: float) -> tuple[np.ndarray, float]:
    best_bits, best_value = None, -np.inf
    for combo in combinations(range(len(unary)), k):
        bits = np.zeros(len(unary), dtype=np.int8)
        bits[list(combo)] = 1
        value = reduction_objective(bits, unary, corr, penalty)
        if value > best_value:
            best_bits, best_value = bits, value
    return best_bits, float(best_value)


def solver_benchmark(
    snapshots: pd.DataFrame,
    return_panel: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows: list[dict] = []
    small_rows: list[dict] = []
    representative_folds = [29, 32, 35, 38, 41, 43]
    for fold in representative_folds:
        snapshot = snapshots[snapshots["fold"].eq(fold)].copy()
        x = common_scores(snapshot, set(), config)
        tickers = x["ticker"].tolist()
        unary = x["unary_score"].to_numpy(float)
        corr = absolute_correlation(return_panel, tickers, pd.Timestamp(snapshot["decision_time"].iloc[0]))
        solver_values: dict[str, float] = {}
        for solver in ("greedy_AUR", "warm_swap_QAUR", "multistart_swap_QAUR", "simulated_annealing"):
            started = time.perf_counter()
            if solver == "greedy_AUR":
                bits = greedy_solution(unary, corr, config.candidate_size, config.correlation_penalty)
            elif solver == "warm_swap_QAUR":
                bits = best_improving_swap(
                    greedy_solution(unary, corr, config.candidate_size, config.correlation_penalty),
                    unary, corr, config.correlation_penalty,
                )
            elif solver == "multistart_swap_QAUR":
                bits = multistart_swap_search(
                    unary, corr, config.candidate_size, config.correlation_penalty, SEED + fold,
                )
            else:
                bits = simulated_annealing_cardinality(
                    unary, corr, config.candidate_size, config.correlation_penalty, SEED + fold,
                )
            runtime = time.perf_counter() - started
            value = reduction_objective(bits, unary, corr, config.correlation_penalty)
            solver_values[solver] = value
            full_rows.append({
                "fold": fold,
                "universe_size": len(unary),
                "cardinality": config.candidate_size,
                "solver": solver,
                "objective": value,
                "runtime_seconds": runtime,
            })

        order = np.argsort(unary)[::-1]
        for n in (8, 10, 12):
            indices = order[:n]
            small_unary = unary[indices]
            small_corr = corr[np.ix_(indices, indices)]
            k = min(4, n // 2)
            exact_started = time.perf_counter()
            _, exact_value = exact_solution(small_unary, small_corr, k, config.correlation_penalty)
            exact_runtime = time.perf_counter() - exact_started
            small_rows.append({
                "fold": fold, "n": n, "k": k, "solver": "exact_enumeration",
                "objective": exact_value, "objective_gap": 0.0,
                "runtime_seconds": exact_runtime, "feasibility_rate": 1.0,
                "success_probability": 1.0, "executed_on_qpu": False,
            })
            for solver in ("greedy_AUR", "warm_swap_QAUR", "multistart_swap_QAUR", "simulated_annealing"):
                started = time.perf_counter()
                if solver == "greedy_AUR":
                    bits = greedy_solution(small_unary, small_corr, k, config.correlation_penalty)
                elif solver == "warm_swap_QAUR":
                    bits = best_improving_swap(
                        greedy_solution(small_unary, small_corr, k, config.correlation_penalty),
                        small_unary, small_corr, config.correlation_penalty,
                    )
                elif solver == "multistart_swap_QAUR":
                    bits = multistart_swap_search(
                        small_unary, small_corr, k, config.correlation_penalty, SEED + fold + n,
                    )
                else:
                    bits = simulated_annealing_cardinality(
                        small_unary, small_corr, k, config.correlation_penalty, SEED + fold + n,
                        steps=3000, restarts=3,
                    )
                runtime = time.perf_counter() - started
                value = reduction_objective(bits, small_unary, small_corr, config.correlation_penalty)
                small_rows.append({
                    "fold": fold, "n": n, "k": k, "solver": solver,
                    "objective": value,
                    "objective_gap": float((exact_value - value) / max(abs(exact_value), 1e-12)),
                    "runtime_seconds": runtime, "feasibility_rate": 1.0,
                    "success_probability": np.nan, "executed_on_qpu": False,
                })

            q = objective_matrix(small_unary, small_corr, config.correlation_penalty)
            started = time.perf_counter()
            qaoa = xy_qaoa_statevector_audit(q, k, SEED + fold + n, depth=2, budget=30, shots=1024)
            runtime = time.perf_counter() - started
            small_rows.append({
                "fold": fold, "n": n, "k": k, "solver": "XY_QAOA_statevector",
                "objective": np.nan, "objective_gap": qaoa["optimality_gap"],
                "runtime_seconds": runtime,
                "feasibility_rate": qaoa["feasibility_rate"],
                "success_probability": qaoa["success_probability"],
                "executed_on_qpu": False,
            })
    return pd.DataFrame(full_rows), pd.DataFrame(small_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--warm-dir", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    warm = args.warm_dir.resolve()
    best_returns = pd.read_csv(warm / "best_configuration_returns.csv", parse_dates=["date"])
    best_folds = pd.read_csv(warm / "best_configuration_fold_diagnostics.csv")
    all_warm_returns = pd.read_csv(warm / "all_configuration_returns.csv", parse_dates=["date"])
    snapshots = pd.read_csv(warm / "forecast_snapshots.csv", parse_dates=["decision_time"])
    manifest = json.loads((warm / "run_manifest.json").read_text(encoding="utf-8"))
    config = StrategyConfig(**manifest["best_config"])

    prices, _, benchmark = load_market_data(args.dataset)
    features = build_features(prices)
    folds = make_folds(features["date"])
    return_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    holdout_first = int(manifest["holdout_first_fold"])
    holdout_start = folds[holdout_first]["test_start"]
    holdout_end = folds[-1]["test_end"]
    full_ew = return_panel.loc[(return_panel.index >= holdout_start) & (return_panel.index < holdout_end)].mean(axis=1)
    benchmark_returns = benchmark.set_index("date")["return"].loc[lambda x: (x.index >= holdout_start) & (x.index < holdout_end)].dropna()

    # Cost stress and yearly stability.
    cost_rows: list[dict] = []
    stressed_returns: dict[int, pd.DataFrame] = {}
    for cost in (0, 25, 50, 75, 100):
        stressed = cost_stress_returns(best_returns, best_folds, 25, cost)
        stressed_returns[cost] = stressed
        for sample, mask in (("holdout", stressed["fold"] >= holdout_first), ("all", stressed["fold"] >= 0)):
            for method, group in stressed[mask].groupby("method"):
                cost_rows.append({"cost_bps": cost, "sample": sample, "method": method, **financial_metrics(group.sort_values("date")["return"])})
    cost_stress = pd.DataFrame(cost_rows)
    cost_stress.to_csv(output / "transaction_cost_stress.csv", index=False)

    year_rows: list[dict] = []
    for (year, method), group in best_returns.assign(year=best_returns["date"].dt.year).groupby(["year", "method"]):
        year_rows.append({"year": year, "method": method, **financial_metrics(group.sort_values("date")["return"])})
    yearly = pd.DataFrame(year_rows)
    yearly.to_csv(output / "yearly_performance.csv", index=False)

    # Prequential configuration selection uses only trailing folds.
    prequential, choices = prequential_configuration_selection(all_warm_returns, folds, lookback_folds=12)
    prequential.to_csv(output / "prequential_returns.csv", index=False)
    choices.to_csv(output / "prequential_config_choices.csv", index=False)
    prequential_summary = pd.DataFrame([
        {"method": method, **financial_metrics(group.sort_values("date")["return"])}
        for method, group in prequential.groupby("method")
    ])
    prequential_summary.to_csv(output / "prequential_summary.csv", index=False)
    prequential_sensitivity_rows: list[dict] = []
    ensemble_parts: list[pd.DataFrame] = []
    for lookback in (6, 9, 12, 18, 24):
        sensitivity_returns, sensitivity_choices = prequential_configuration_selection(
            all_warm_returns, folds, lookback_folds=lookback,
        )
        sensitivity_returns = sensitivity_returns.copy()
        sensitivity_returns["lookback_folds"] = lookback
        ensemble_parts.append(sensitivity_returns)
        for sample, mask in (
            ("available_path", sensitivity_returns["fold"] >= lookback),
            ("common_holdout", sensitivity_returns["fold"] >= holdout_first),
        ):
            for method, group in sensitivity_returns[mask].groupby("method"):
                prequential_sensitivity_rows.append({
                    "lookback_folds": lookback,
                    "sample": sample,
                    "method": method,
                    "configuration_changes": int((sensitivity_choices["selected_config"] != sensitivity_choices["selected_config"].shift()).sum()),
                    **financial_metrics(group.sort_values("date")["return"]),
                })
    prequential_sensitivity = pd.DataFrame(prequential_sensitivity_rows)
    prequential_sensitivity.to_csv(output / "prequential_sensitivity.csv", index=False)
    ensemble_returns = (
        pd.concat(ensemble_parts, ignore_index=True)
        .groupby(["fold", "date", "method"], as_index=False)["return"].mean()
    )
    ensemble_returns.to_csv(output / "prequential_multilookback_ensemble_returns.csv", index=False)
    ensemble_summary_rows: list[dict] = []
    for sample, mask in (
        ("common_available_path", ensemble_returns["fold"] >= 24),
        ("common_holdout", ensemble_returns["fold"] >= holdout_first),
    ):
        for method, group in ensemble_returns[mask].groupby("method"):
            ensemble_summary_rows.append({"sample": sample, "method": method, **financial_metrics(group.sort_values("date")["return"])})
    ensemble_summary = pd.DataFrame(ensemble_summary_rows)
    ensemble_summary.to_csv(output / "prequential_multilookback_ensemble_summary.csv", index=False)

    # Multiple-testing inputs from every completed experiment family.
    trial_frames = []
    for directory in (args.phase1_dir, args.warm_dir, args.overlay_dir):
        frame = pd.read_csv(Path(directory) / "configuration_results.csv")
        trial_frames.append(frame[frame["sample"].eq("development")][["config_id", "method", "sharpe_zero_rf"]])
    trials = pd.concat(trial_frames, ignore_index=True).drop_duplicates(["config_id", "method"])
    number_of_trials = int(trials["config_id"].nunique())

    holdout = best_returns[best_returns["fold"] >= holdout_first]
    robustness_rows: list[dict] = []
    dsr_rows: list[dict] = []
    for method, group in holdout.groupby("method"):
        strategy = group.set_index("date")["return"].sort_index()
        for comparator_name, comparator in (("FULL_UNIVERSE_EW", full_ew), ("VNALLSHARE_TRI", benchmark_returns)):
            robustness_rows.append({
                "method": method,
                "comparator": comparator_name,
                **moving_block_bootstrap_difference(strategy, comparator),
            })
        dsr_rows.append({"method": method, **deflated_sharpe_probability(strategy, trials["sharpe_zero_rf"], number_of_trials)})
    bootstrap = pd.DataFrame(robustness_rows)
    bootstrap.to_csv(output / "block_bootstrap_excess_return.csv", index=False)
    dsr = pd.DataFrame(dsr_rows)
    dsr.to_csv(output / "deflated_sharpe.csv", index=False)
    ensemble_robustness_rows: list[dict] = []
    ensemble_dsr_rows: list[dict] = []
    ensemble_holdout = ensemble_returns[ensemble_returns["fold"] >= holdout_first]
    for method, group in ensemble_holdout.groupby("method"):
        strategy = group.set_index("date")["return"].sort_index()
        for comparator_name, comparator in (("FULL_UNIVERSE_EW", full_ew), ("VNALLSHARE_TRI", benchmark_returns)):
            ensemble_robustness_rows.append({
                "method": method, "comparator": comparator_name,
                **moving_block_bootstrap_difference(strategy, comparator, seed=SEED + 1),
            })
        ensemble_dsr_rows.append({"method": method, **deflated_sharpe_probability(strategy, trials["sharpe_zero_rf"], number_of_trials + 1)})
    ensemble_robustness = pd.DataFrame(ensemble_robustness_rows)
    ensemble_robustness.to_csv(output / "ensemble_block_bootstrap_excess_return.csv", index=False)
    ensemble_dsr = pd.DataFrame(ensemble_dsr_rows)
    ensemble_dsr.to_csv(output / "ensemble_deflated_sharpe.csv", index=False)

    # Solver benchmark and quantum-claim gates.
    full_solver, small_solver = solver_benchmark(snapshots, return_panel, config)
    full_solver.to_csv(output / "full_universe_solver_benchmark.csv", index=False)
    small_solver.to_csv(output / "small_instance_quantum_benchmark.csv", index=False)

    holdout_metrics = {
        method: financial_metrics(group.sort_values("date")["return"])
        for method, group in holdout.groupby("method")
    }
    cost75 = cost_stress[(cost_stress["cost_bps"].eq(75)) & (cost_stress["sample"].eq("holdout"))]
    excess_full_significant = bool((bootstrap[bootstrap["comparator"].eq("FULL_UNIVERSE_EW")]["pvalue_one_sided_positive"] < 0.05).all())
    excess_benchmark_significant = bool((bootstrap[bootstrap["comparator"].eq("VNALLSHARE_TRI")]["pvalue_one_sided_positive"] < 0.05).all())
    gates = pd.DataFrame([
        {"gate": "positive_holdout_return_both_methods", "passed": all(v["cumulative_return"] > 0 for v in holdout_metrics.values()), "evidence": str({m: v["cumulative_return"] for m, v in holdout_metrics.items()})},
        {"gate": "holdout_sharpe_at_least_1_both_methods", "passed": all(v["sharpe_zero_rf"] >= 1 for v in holdout_metrics.values()), "evidence": str({m: v["sharpe_zero_rf"] for m, v in holdout_metrics.items()})},
        {"gate": "holdout_max_drawdown_no_worse_than_minus_20pct", "passed": all(v["maximum_drawdown"] >= -0.20 for v in holdout_metrics.values()), "evidence": str({m: v["maximum_drawdown"] for m, v in holdout_metrics.items()})},
        {"gate": "positive_sharpe_at_75bps", "passed": bool((cost75["sharpe_zero_rf"] > 0).all()), "evidence": cost75[["method", "sharpe_zero_rf"]].to_dict("records").__str__()},
        {"gate": "significant_excess_return_vs_full_EW", "passed": excess_full_significant, "evidence": bootstrap[bootstrap["comparator"].eq("FULL_UNIVERSE_EW")].to_dict("records").__str__()},
        {"gate": "significant_excess_return_vs_VNAllshare", "passed": excess_benchmark_significant, "evidence": bootstrap[bootstrap["comparator"].eq("VNALLSHARE_TRI")].to_dict("records").__str__()},
        {"gate": "deflated_sharpe_probability_at_least_95pct", "passed": bool((dsr["deflated_sharpe_probability"] >= 0.95).all()), "evidence": dsr[["method", "deflated_sharpe_probability"]].to_dict("records").__str__()},
        {"gate": "at_least_24_months_untouched_forward_or_paper_track_record", "passed": False, "evidence": "Current temporal holdout is about 15 months and is now observed."},
        {"gate": "market_data_current_with_operational_live_pipeline", "passed": False, "evidence": "Dataset ends 2025-12-31; no audited 2026 paper-trading feed/order pipeline."},
    ])
    gates.to_csv(output / "live_capital_readiness_gates.csv", index=False)

    qaoa_rows = small_solver[small_solver["solver"].eq("XY_QAOA_statevector")]
    classical_rows = small_solver[small_solver["solver"].isin(["exact_enumeration", "simulated_annealing", "warm_swap_QAUR", "multistart_swap_QAUR"])]
    quantum_gates = pd.DataFrame([
        {"gate": "executed_on_physical_QPU", "passed": False, "evidence": "All XY-QAOA results are ideal statevector simulations."},
        {"gate": "matched_best_classical_baselines", "passed": False, "evidence": "Exact enumeration and simulated annealing included; commercial/state-of-the-art solvers and tuned HPC baselines still missing."},
        {"gate": "better_solution_quality_than_best_classical", "passed": False, "evidence": f"Exact classical gap is zero by definition; mean statevector observed gap={qaoa_rows['objective_gap'].mean():.6g}."},
        {"gate": "lower_end_to_end_wall_clock_than_best_classical", "passed": False, "evidence": f"Mean XY-QAOA simulator runtime={qaoa_rows['runtime_seconds'].mean():.6f}s; mean exact/heuristic runtime={classical_rows['runtime_seconds'].mean():.6f}s."},
        {"gate": "scaling_crossover_demonstrated", "passed": False, "evidence": "Statevector tested only through n=12; no QPU scaling crossover."},
        {"gate": "hardware_noise_and_repeated_run_statistics", "passed": False, "evidence": "No hardware calibration, queue, shot-noise or error-mitigation study."},
        {"gate": "independent_reproducibility", "passed": False, "evidence": "No independent QPU reproduction."},
    ])
    quantum_gates.to_csv(output / "quantum_advantage_gates.csv", index=False)

    # Figures.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, group in cost_stress[cost_stress["sample"].eq("holdout")].groupby("method"):
        axes[0].plot(group["cost_bps"], group["sharpe_zero_rf"], marker="o", label=method)
    axes[0].set_title("Holdout Sharpe under transaction-cost stress")
    axes[0].set_xlabel("Cost (bps)")
    axes[0].legend()
    runtime_summary = small_solver.groupby(["n", "solver"], as_index=False)["runtime_seconds"].mean()
    for solver, group in runtime_summary.groupby("solver"):
        axes[1].plot(group["n"], group["runtime_seconds"], marker="o", label=solver)
    axes[1].set_yscale("log")
    axes[1].set_title("Small-instance solver runtime")
    axes[1].set_xlabel("Variables n")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "readiness_and_solver_benchmark.png", dpi=180)
    plt.close()

    live_passes = int(gates["passed"].sum())
    quantum_passes = int(quantum_gates["passed"].sum())
    report = f"""# Live-capital readiness and quantum-advantage audit

## Outcome

- Live-capital gates passed: **{live_passes}/{len(gates)}**.
- Quantum-advantage gates passed: **{quantum_passes}/{len(quantum_gates)}**.
- Config audited: `{config.config_id}`.
- Dataset ends at 2025-12-31; this is not a current live recommendation.

## Live-capital readiness gates

{gates.to_markdown(index=False)}

## Transaction-cost stress

{cost_stress[cost_stress['sample'].eq('holdout')].to_markdown(index=False)}

## Deflated Sharpe after {number_of_trials} tested configurations

{dsr.to_markdown(index=False)}

## Paired moving-block bootstrap versus baselines

{bootstrap.to_markdown(index=False)}

## Prequential configuration selection

{prequential_summary.to_markdown(index=False)}

## Prequential lookback sensitivity

{prequential_sensitivity.to_markdown(index=False)}

## Multi-lookback prequential ensemble

{ensemble_summary.to_markdown(index=False)}

{ensemble_robustness.to_markdown(index=False)}

{ensemble_dsr.to_markdown(index=False)}

## Quantum-advantage gates

{quantum_gates.to_markdown(index=False)}

## Recommendation

The current evidence supports continued research and a shadow/paper portfolio, not unrestricted live capital. A staged pilot can only be considered after a fresh forward period, current data and operational controls. The present simulator cannot support a quantum-advantage statement. A valid claim requires physical-QPU runs, matched wall-clock accounting and statistically superior quality/time scaling against the strongest tuned classical solvers.
"""
    (output / "READINESS_AND_QUANTUM_AUDIT.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "config_id": config.config_id,
        "live_capital_gates_passed": live_passes,
        "live_capital_gates_total": len(gates),
        "quantum_advantage_gates_passed": quantum_passes,
        "quantum_advantage_gates_total": len(quantum_gates),
        "number_of_configuration_trials_for_dsr": number_of_trials,
        "runtime_seconds": time.perf_counter() - started,
        "quantum_advantage_claimed": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
