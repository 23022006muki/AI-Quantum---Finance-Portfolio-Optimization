from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

from .reduction import adaptive_reduce, quantum_assisted_reduce
from .optimization import portfolio_qubo, xy_qaoa_select, optimize_weights


def _metrics(r: pd.Series) -> dict:
    r = r.dropna()
    wealth = (1 + r).cumprod()
    ann = float(wealth.iloc[-1] ** (252 / len(r)) - 1) if len(r) else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
    dd = wealth / wealth.cummax() - 1
    return {"observations": len(r), "cumulative_return": float(wealth.iloc[-1] - 1), "annualized_return": ann, "annualized_volatility": vol, "sharpe_zero_rf": ann / vol if vol > 0 else np.nan, "maximum_drawdown": float(dd.min())}


def run(root: Path, config_path: Path, output_dir: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    old = root.parent / "quantum_portfolio_data"
    experiment = old / "outputs" / "experiments" / "20260813T164535-21c9b569ce"
    forecasts = pd.read_csv(experiment / "rankings.csv", parse_dates=["decision_time"])
    folds = pd.read_csv(experiment / "fold_manifest.csv", parse_dates=["test_start", "test_end"])
    features = pd.read_parquet(old / "outputs" / "curated" / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    returns = features.pivot(index="date", columns="ticker", values="_ret1").sort_index()
    snapshot_cols = ["date", "ticker", "liquidity_20d", "volatility_20d"]
    feature_slice = features[snapshot_cols].sort_values("date")
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows, return_rows, solver_rows = [], [], []
    previous_universe = {"AUR": set(), "QAUR": set()}
    previous_weights = {"AUR": {}, "QAUR": {}}
    previous_full_weights = {}
    for fold in folds.itertuples(index=False):
        rank = forecasts[forecasts["fold"] == fold.fold].copy()
        decision = rank["decision_time"].max()
        snap_date = feature_slice.loc[feature_slice["date"] <= decision, "date"].max()
        snap = feature_slice[feature_slice["date"] == snap_date].merge(rank[["ticker", "signal", "xgboost_expected_return"]], on="ticker", how="inner")
        history = returns.loc[(returns.index < decision) & (returns.index >= decision - pd.Timedelta(days=180))]
        reducers = {
            "AUR": adaptive_reduce(snap, history, previous_universe["AUR"], cfg["candidate_size"], cfg["adaptive"]),
            "QAUR": quantum_assisted_reduce(snap, history, previous_universe["QAUR"], cfg["candidate_size"], cfg["quantum_assisted"], cfg["seed"] + fold.fold, cfg["qa_restarts"], cfg["qa_swap_steps"]),
        }
        for method, reduced in reducers.items():
            candidates = list(reduced.tickers)
            mu = rank.set_index("ticker").reindex(candidates)["xgboost_expected_return"].fillna(0).to_numpy(float)
            cov = history.reindex(columns=candidates).ewm(span=cfg["covariance_span"]).cov().groupby(level=1).tail(1).droplevel(0).reindex(index=candidates, columns=candidates).fillna(0).to_numpy(float)
            if not np.isfinite(cov).all() or np.allclose(cov, 0): cov = np.eye(len(candidates)) * 1e-4
            q = portfolio_qubo(mu, cov, cfg["risk_aversion"])
            solved = xy_qaoa_select(q, cfg["portfolio_cardinality"], cfg["seed"] + fold.fold, cfg["qaoa_depth"], cfg["qaoa_parameter_trials"], cfg["qaoa_shots"])
            chosen_idx = np.flatnonzero(solved["bits"])
            chosen = [candidates[i] for i in chosen_idx]
            chosen_mu, chosen_cov = mu[chosen_idx], cov[np.ix_(chosen_idx, chosen_idx)]
            weights = optimize_weights(chosen_mu, chosen_cov, cfg["weight_lower"], cfg["weight_upper"], cfg["weight_risk_aversion"])
            target = dict(zip(chosen, weights))
            turnover = 0.5 * sum(abs(target.get(t, 0) - previous_weights[method].get(t, 0)) for t in set(target) | set(previous_weights[method]))
            test = returns.loc[(returns.index >= fold.test_start) & (returns.index < fold.test_end), chosen].fillna(0)
            daily = test.to_numpy() @ weights
            if len(daily): daily[0] -= turnover * cfg["transaction_cost_bps"] / 10000
            for date, value in zip(test.index, daily): return_rows.append({"fold": fold.fold, "date": date, "method": method, "return": value})
            for t in candidates: selection_rows.append({"fold": fold.fold, "decision_time": decision, "method": method, "ticker": t, "selected_downstream": t in chosen, "reduction_objective": reduced.objective})
            solver_rows.append({"fold": fold.fold, "method": method, "backend": solved["backend"], "energy": solved["energy"], "feasibility_rate": solved["feasibility_rate"], "turnover": turnover, "reduction_backend": reduced.diagnostics["backend"]})
            previous_universe[method], previous_weights[method] = set(candidates), target
        # Feasibility baseline: the full universe cannot enter the same 8-qubit
        # statevector solver. Equal weight is therefore reported separately and
        # never interpreted as a controlled quantum comparison.
        full_names = snap["ticker"].tolist()
        full_target = {t: 1.0 / len(full_names) for t in full_names}
        full_turnover = 0.5 * sum(abs(full_target.get(t, 0) - previous_full_weights.get(t, 0)) for t in set(full_target) | set(previous_full_weights))
        full_test = returns.loc[(returns.index >= fold.test_start) & (returns.index < fold.test_end), full_names].fillna(0)
        full_daily = full_test.mean(axis=1).to_numpy()
        if len(full_daily): full_daily[0] -= full_turnover * cfg["transaction_cost_bps"] / 10000
        for date, value in zip(full_test.index, full_daily): return_rows.append({"fold": fold.fold, "date": date, "method": "FULL_UNIVERSE_EW", "return": value})
        previous_full_weights = full_target
    selections = pd.DataFrame(selection_rows); portfolio_returns = pd.DataFrame(return_rows); solvers = pd.DataFrame(solver_rows)
    selections.to_csv(output_dir / "universe_and_asset_selections.csv", index=False)
    portfolio_returns.to_csv(output_dir / "portfolio_returns.csv", index=False)
    solvers.to_csv(output_dir / "solver_diagnostics.csv", index=False)
    summary = pd.DataFrame([{"method": m, **_metrics(g.set_index("date")["return"])} for m, g in portfolio_returns.groupby("method")])
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    overlap = []
    for f, g in selections.groupby("fold"):
        a, q = set(g[(g.method == "AUR")].ticker), set(g[(g.method == "QAUR")].ticker)
        overlap.append({"fold": f, "intersection": len(a & q), "jaccard": len(a & q) / len(a | q)})
    pd.DataFrame(overlap).to_csv(output_dir / "reduction_comparison.csv", index=False)
    wide = portfolio_returns.pivot(index="date", columns="method", values="return").dropna(subset=["AUR", "QAUR"])
    from scipy import stats
    paired = stats.ttest_rel(wide["QAUR"], wide["AUR"])
    nonzero = wide["QAUR"] - wide["AUR"]
    wilcoxon_p = float(stats.wilcoxon(nonzero).pvalue) if (nonzero != 0).any() else 1.0
    pd.DataFrame([{"comparison": "QAUR_minus_AUR", "n": len(wide), "mean_daily_difference": float(nonzero.mean()), "paired_t_pvalue": float(paired.pvalue), "wilcoxon_pvalue": wilcoxon_p}]).to_csv(output_dir / "statistical_tests.csv", index=False)
    (output_dir / "run_manifest.json").write_text(json.dumps({"framework": "AUR_vs_QAUR_shared_XY_QAOA", "folds": int(folds.shape[0]), "forecast_source": str(experiment), "qa_backend_disclosure": "classical surrogate for quantum-ready QUBO"}, indent=2), encoding="utf-8")
    return output_dir
