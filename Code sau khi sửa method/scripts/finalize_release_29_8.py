from __future__ import annotations

"""Add H5 seed robustness and the shared XY-QAOA audit to an existing run.

This idempotent helper is mainly useful when the expensive configuration grid
was launched before the final audit code was added.  It recomputes the selected
confirmatory configuration only; it never changes the selected configuration or
the practical period returns.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_constraint_strategy_search import (
    SEED,
    build_features,
    exact_cardinality_qubo,
    ewma_covariance,
    financial_metrics,
    load_market_data,
    make_configs,
    make_folds,
    run_configuration,
    xy_qaoa_statevector_audit,
)
from run_colab_29_8_complete import holm_adjust, practical_positive_return_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best_id = str(manifest["confirmatory_best_config"])
    config = next(item for item in make_configs() if item.config_id == best_id)

    price, _, _ = load_market_data(args.dataset.resolve())
    features = build_features(price)
    return_panel = features.pivot(
        index="date", columns="ticker", values="return_1d"
    ).sort_index()
    folds = [
        fold for fold in make_folds(features["date"])
        if int(fold["fold"]) <= 43
    ]
    snapshots = pd.read_csv(
        output / "forecast_snapshots.csv", parse_dates=["decision_time"]
    )

    seed_rows: list[dict] = []
    base_selections: pd.DataFrame | None = None
    for qa_seed in (7, 42, 99):
        returns, diagnostics, selections = run_configuration(
            config, snapshots, return_panel, folds, qa_seed=qa_seed
        )
        if qa_seed == SEED:
            base_selections = selections
        holdout = returns[returns["fold"].between(29, 43)]
        performance = {
            method: financial_metrics(group.sort_values("date")["return"])
            for method, group in holdout.groupby("method")
        }
        objective = diagnostics.pivot(
            index="fold", columns="method", values="reduction_objective"
        ).dropna()
        seed_rows.append({
            "qa_seed": qa_seed,
            "aur_sharpe": performance["AUR"]["sharpe_zero_rf"],
            "qaur_sharpe": performance["QAUR"]["sharpe_zero_rf"],
            "qaur_minus_aur_sharpe": (
                performance["QAUR"]["sharpe_zero_rf"]
                - performance["AUR"]["sharpe_zero_rf"]
            ),
            "mean_qaur_objective_advantage": float(
                (objective["QAUR"] - objective["AUR"]).mean()
            ),
        })
    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(output / "confirmatory_seed_robustness.csv", index=False)

    tests_path = output / "confirmatory_hypothesis_tests.csv"
    tests = pd.read_csv(tests_path)
    h5_name = "H5_QAUR_financial_direction_robust_across_seeds"
    tests = tests[tests["hypothesis"].ne(h5_name)]
    h5_supported = bool((seed_table["qaur_minus_aur_sharpe"] > 0).all())
    tests = pd.concat([
        tests,
        pd.DataFrame([{
            "hypothesis": h5_name,
            "estimate": float(
                (seed_table["qaur_minus_aur_sharpe"] > 0).mean()
            ),
            "statistic": np.nan,
            "pvalue_one_sided": np.nan,
            "supported_5pct": h5_supported,
            "evidence_label": "confirmatory_untouched_historical_holdout",
        }]),
    ], ignore_index=True)
    tests["holm_adjusted_pvalue"] = np.nan
    inferential = tests["pvalue_one_sided"].notna()
    tests.loc[inferential, "holm_adjusted_pvalue"] = holm_adjust(
        tests.loc[inferential, "pvalue_one_sided"]
    )
    tests["supported_holm_5pct"] = tests["holm_adjusted_pvalue"].lt(0.05)
    tests.to_csv(tests_path, index=False)

    selected_returns = pd.read_csv(
        output / "selected_practical_returns.csv", parse_dates=["date"]
    )
    positive_evidence = practical_positive_return_evidence(selected_returns)
    positive_evidence.to_csv(
        output / "selected_practical_positive_return_evidence.csv", index=False
    )

    if base_selections is None:
        _, _, base_selections = run_configuration(
            config, snapshots, return_panel, folds, qa_seed=SEED
        )
    xy_rows: list[dict] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        if fold_id <= 28:
            continue
        snapshot = snapshots[snapshots["fold"].eq(fold_id)].copy()
        decision_time = pd.Timestamp(snapshot["decision_time"].iloc[0])
        for method in ("AUR", "QAUR"):
            candidates = base_selections.loc[
                base_selections["fold"].eq(fold_id)
                & base_selections["method"].eq(method),
                "ticker",
            ].tolist()
            candidate_snapshot = snapshot.set_index("ticker").reindex(candidates)
            mu = (
                config.signal_blend * candidate_snapshot["xgb_signal"]
                + (1.0 - config.signal_blend)
                * candidate_snapshot["momentum_signal"]
            ).to_numpy(float)
            cov = ewma_covariance(
                return_panel,
                candidates,
                decision_time,
                config.covariance_span,
                config.covariance_shrinkage,
            )
            _, q_matrix = exact_cardinality_qubo(
                mu, cov, config.portfolio_cardinality, config.risk_aversion_qubo
            )
            audit = xy_qaoa_statevector_audit(
                q_matrix, config.portfolio_cardinality, SEED + fold_id
            )
            xy_rows.append({
                "fold": fold_id,
                "method": method,
                "candidate_size": len(candidates),
                "portfolio_cardinality": config.portfolio_cardinality,
                "depth": 2,
                "budget": 30,
                "shots": 1024,
                **audit,
            })
    xy_audit = pd.DataFrame(xy_rows)
    xy_audit.to_csv(
        output / "confirmatory_xy_qaoa_holdout_audit.csv", index=False
    )

    manifest["xy_qaoa_holdout_audit_instances"] = int(len(xy_audit))
    manifest["xy_qaoa_mean_feasibility_rate"] = float(
        xy_audit["feasibility_rate"].mean()
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_path = output / "FINAL_RESULTS_29_8_VI.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## Audit bổ sung H5 và XY-QAOA\n"
    report = report.split(marker, 1)[0]
    report += (
        marker
        + "\n"
        + tests.tail(1).to_markdown(index=False)
        + "\n\n"
        + f"XY-QAOA được audit trên {len(xy_audit)} instances holdout; "
        + f"feasibility rate trung bình = {xy_audit['feasibility_rate'].mean():.4f}, "
        + f"optimality gap trung bình = {xy_audit['optimality_gap'].mean():.6f}.\n"
        + "\nLợi nhuận cộng dồn dương được tách khỏi kiểm định mean daily return > 0; "
        + "xem `selected_practical_positive_return_evidence.csv`.\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(tests.tail(1).to_string(index=False))
    print(xy_audit.groupby("method")[["feasibility_rate", "optimality_gap", "success_probability"]].mean())


if __name__ == "__main__":
    main()
