from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .data_pipeline import Paths, build_universe, leakage_audit, sha256_file, validate_data


FEATURES = [
    "return_5d", "return_20d", "return_60d", "return_120d", "sma_ratio_20",
    "ema_ratio_20", "rsi_14", "macd", "atr_14", "volatility_20d",
    "downside_volatility_20d", "drawdown_60d", "liquidity_20d", "beta_60d",
    "roe_pit", "revenue_growth_yoy_pit", "policy_rate_pit",
]


class ResearchRunBlocked(RuntimeError):
    """Raised after an auditable blocked-run artifact has been written."""

    def __init__(self, message: str, output_dir: Path):
        super().__init__(message)
        self.output_dir = output_dir


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = -d.clip(upper=0).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def build_features(prices: pd.DataFrame, target_horizon_days: int = 20) -> pd.DataFrame:
    base = prices.sort_values(["ticker", "date"]).copy()
    base["_ret1"] = base.groupby("ticker")["adjusted_close"].pct_change()
    market_return = base.groupby("date")["_ret1"].mean()
    frames = []
    for ticker, g in base.groupby("ticker"):
        x = g.copy().sort_values("date")
        p = x["adjusted_close"].astype(float)
        r = p.pct_change()
        for n in (5, 20, 60, 120):
            x[f"return_{n}d"] = p.pct_change(n)
        x["sma_ratio_20"] = p / p.rolling(20).mean() - 1
        ema12, ema26 = p.ewm(span=12, adjust=False).mean(), p.ewm(span=26, adjust=False).mean()
        x["ema_ratio_20"] = p / p.ewm(span=20, adjust=False).mean() - 1
        x["rsi_14"] = _rsi(p)
        x["macd"] = ema12 - ema26
        tr = pd.concat([
            x["high"] - x["low"], (x["high"] - p.shift()).abs(), (x["low"] - p.shift()).abs()
        ], axis=1).max(axis=1)
        x["atr_14"] = tr.rolling(14).mean() / p
        x["volatility_20d"] = r.rolling(20).std()
        x["downside_volatility_20d"] = r.where(r < 0, 0).rolling(20).std()
        x["drawdown_60d"] = p / p.rolling(60).max() - 1
        x["liquidity_20d"] = np.log1p(x["trading_value"].rolling(20).mean())
        market = x["date"].map(market_return)
        x["beta_60d"] = r.rolling(60).cov(market) / market.rolling(60).var()
        x["target_return_20d"] = p.shift(-target_horizon_days) / p - 1
        x["label_end_time"] = pd.to_datetime(x["date"]).shift(-target_horizon_days)
        x["target_horizon_days"] = int(target_horizon_days)
        x["target_rank"] = np.nan
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    out["target_rank"] = out.groupby("date")["target_return_20d"].rank(pct=True)
    out["feature_available_at"] = pd.to_datetime(
        out.get("available_at", out["date"]), errors="coerce"
    )
    out["roe_pit"] = np.nan
    out["revenue_growth_yoy_pit"] = np.nan
    out["policy_rate_pit"] = np.nan
    return out


def attach_point_in_time_features(features: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    out_frames = []
    financial_path = paths.normalized / "financial_statements.parquet"
    if financial_path.exists():
        financial = pd.read_parquet(financial_path)
        financial["available_at"] = pd.to_datetime(financial["available_at"])
        financial = financial.sort_values(["ticker", "available_at"])
        financial["roe_pit"] = financial["net_income"] / financial["equity"].replace(0, np.nan)
        financial["revenue_growth_yoy_pit"] = financial.groupby("ticker")["revenue"].pct_change(4)
        for ticker, group in features.groupby("ticker", sort=False):
            right = financial[financial.ticker == ticker][
                ["available_at", "roe_pit", "revenue_growth_yoy_pit"]
            ].rename(columns={"available_at": "financial_available_at"}).sort_values("financial_available_at")
            left = group.drop(columns=["roe_pit", "revenue_growth_yoy_pit"]).sort_values(
                "feature_available_at"
            )
            if right.empty:
                left["roe_pit"] = np.nan
                left["revenue_growth_yoy_pit"] = np.nan
            else:
                left = pd.merge_asof(
                    left, right, left_on="feature_available_at", right_on="financial_available_at",
                    direction="backward", allow_exact_matches=True,
                ).drop(columns=["financial_available_at"])
            out_frames.append(left)
        features = pd.concat(out_frames, ignore_index=True)
    macro_path = paths.normalized / "macro.parquet"
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
        macro["available_at"] = pd.to_datetime(macro["available_at"])
        policy = macro[macro["series_id"].astype(str).str.contains("POLICY_RATE", case=False)].copy()
        if not policy.empty:
            policy = policy.sort_values("available_at")[["available_at", "value"]].rename(
                columns={"available_at": "macro_available_at", "value": "policy_rate_new"}
            )
            features = pd.merge_asof(
                features.sort_values("feature_available_at"), policy,
                left_on="feature_available_at", right_on="macro_available_at", direction="backward",
            ).drop(columns=["macro_available_at"])
            features["policy_rate_pit"] = features["policy_rate_new"]
            features = features.drop(columns=["policy_rate_new"])
    return features


def make_folds(dates: pd.Series, train_months: int, validation_months: int,
               test_months: int, max_folds: int | None,
               embargo_days: int = 0) -> list[dict]:
    unique = pd.Series(pd.to_datetime(dates).sort_values().unique())
    first = unique.min() + pd.DateOffset(months=train_months + validation_months)
    last = unique.max() - pd.DateOffset(months=test_months)
    anchors = pd.date_range(first, last, freq="ME")
    if max_folds:
        anchors = anchors[-max_folds:]
    folds = []
    for i, test_start in enumerate(anchors):
        train_start = test_start - pd.DateOffset(months=train_months + validation_months)
        validation_start = test_start - pd.DateOffset(months=validation_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        folds.append({
            "fold": i, "train_start": train_start, "train_end": validation_start,
            "validation_start": validation_start,
            "validation_end": test_start, "test_start": test_start, "test_end": test_end,
            "embargo_days": int(embargo_days),
        })
    return folds


def purged_fold_frames(features: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create chronological train/validation/test frames with label purging and embargo."""
    embargo = pd.Timedelta(days=int(fold.get("embargo_days", 0)))
    train_raw = features[
        (features.date >= fold["train_start"]) & (features.date < fold["train_end"])
    ].copy()
    val_raw = features[
        (features.date >= fold["validation_start"]) & (features.date < fold["validation_end"])
    ].copy()
    test = features[
        (features.date > fold["test_start"]) & (features.date <= fold["test_end"])
    ].copy()
    train_cutoff = pd.Timestamp(fold["validation_start"]) - embargo
    validation_cutoff = pd.Timestamp(fold["test_start"]) - embargo
    train = train_raw[
        train_raw["label_end_time"].notna() & (train_raw["label_end_time"] < train_cutoff)
        & (train_raw["feature_available_at"] < train_cutoff)
    ].copy()
    validation = val_raw[
        val_raw["label_end_time"].notna() & (val_raw["label_end_time"] < validation_cutoff)
        & (val_raw["feature_available_at"] < validation_cutoff)
    ].copy()
    audit = {
        **fold,
        "train_rows_raw": len(train_raw), "train_rows_after_purge": len(train),
        "train_rows_purged": len(train_raw) - len(train),
        "validation_rows_raw": len(val_raw), "validation_rows_after_purge": len(validation),
        "validation_rows_purged": len(val_raw) - len(validation),
        "test_rows": len(test), "train_label_cutoff": train_cutoff,
        "validation_label_cutoff": validation_cutoff,
    }
    return train, validation, test, audit


def fit_ranker(train: pd.DataFrame, validation: pd.DataFrame, cfg: dict):
    usable = train.dropna(subset=["target_rank"])
    if usable.empty:
        raise ValueError("No purged training labels are available for this fold.")
    coverage = usable[FEATURES].notna().mean()
    threshold = float(cfg.get("min_feature_coverage", 0.05))
    active_features = coverage[coverage >= threshold].index.tolist()
    if not active_features:
        raise ValueError("All features are below the configured fold coverage threshold.")
    imputer = SimpleImputer(strategy="median").fit(usable[active_features])
    scaler = StandardScaler().fit(imputer.transform(usable[active_features]))
    x_train = scaler.transform(imputer.transform(usable[active_features]))
    y_train = usable["target_rank"].to_numpy()
    validation_usable = validation.dropna(subset=["target_rank"])
    tuning_rows = []
    base_estimators = int(cfg["n_estimators"])
    base_depth = int(cfg["max_depth"])
    candidates = [
        (max(20, base_estimators // 2), max(2, base_depth - 1)),
        (base_estimators, base_depth),
        (base_estimators, max(2, base_depth + 1)),
    ]
    best = None
    for n_estimators, max_depth in dict.fromkeys(candidates):
        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=cfg["learning_rate"], objective="reg:squarederror",
            random_state=int(cfg.get("seed", 42)), n_jobs=1,
        )
        model.fit(x_train, y_train)
        if validation_usable.empty:
            validation_ic = np.nan
        else:
            x_val = scaler.transform(imputer.transform(validation_usable[active_features]))
            validation_ic = float(spearmanr(
                model.predict(x_val), validation_usable["target_rank"]
            ).statistic)
        score = validation_ic if np.isfinite(validation_ic) else -np.inf
        tuning_rows.append({
            "n_estimators": n_estimators, "max_depth": max_depth,
            "validation_rank_ic": validation_ic,
        })
        if best is None or score > best[0]:
            best = (score, model, n_estimators, max_depth)
    return {
        "imputer": imputer, "scaler": scaler, "model": best[1],
        "active_features": active_features,
        "feature_coverage": coverage.to_dict(), "tuning": tuning_rows,
        "selected_params": {"n_estimators": best[2], "max_depth": best[3]},
    }


def predict(model_bundle, df: pd.DataFrame) -> np.ndarray:
    return model_bundle["model"].predict(model_bundle["scaler"].transform(
        model_bundle["imputer"].transform(df[model_bundle["active_features"]])
    ))


def adaptive_reduce(snapshot: pd.DataFrame, history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    snap = snapshot.copy()
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    snap["signal_z"] = z(snap["signal"])
    snap["liquidity_z"] = z(snap["liquidity_20d"])
    snap["risk_z"] = z(snap["volatility_20d"])
    snap["base_score"] = (
        cfg["signal_weight"] * snap["signal_z"]
        + cfg["liquidity_weight"] * snap["liquidity_z"]
        - cfg["risk_weight"] * snap["risk_z"]
    )
    m_max = min(int(cfg.get("max_candidate_size", cfg["candidate_size"])),
                int(cfg["qubit_budget"]), len(snap))
    m_min = min(m_max, max(int(cfg.get("min_candidate_size", cfg.get("cardinality", 1))),
                           int(cfg.get("cardinality", 1))))
    selected: list[str] = []
    returns = history.pivot(index="date", columns="ticker", values="ret1").tail(120)
    corr = returns.corr().fillna(0)
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    average_abs_correlation = float(upper.abs().mean()) if len(upper) else 0.0
    signal_dispersion = float(snap["signal"].std(ddof=0))
    dispersion_reference = float(snap["signal"].abs().median()) + 1e-12
    relative_dispersion = signal_dispersion / dispersion_reference
    m = m_max
    reasons = []
    if relative_dispersion < float(cfg.get("low_signal_dispersion_ratio", 0.10)):
        m = max(m_min, m - 1)
        reasons.append("low_signal_dispersion")
    if average_abs_correlation > float(cfg.get("high_correlation_threshold", 0.65)):
        m = max(m_min, m - 1)
        reasons.append("high_cross_sectional_correlation")
    if not reasons:
        reasons.append("full_budget_supported")
    for _ in range(m):
        remain = snap[~snap["ticker"].isin(selected)].copy()
        if selected:
            remain["corr_penalty"] = [
                float(corr.loc[t, selected].abs().mean()) if t in corr.index else 0.0
                for t in remain["ticker"]
            ]
        else:
            remain["corr_penalty"] = 0.0
        remain["adaptive_score"] = remain["base_score"] - cfg["correlation_penalty"] * remain["corr_penalty"]
        selected.append(remain.sort_values(["adaptive_score", "ticker"], ascending=[False, True]).iloc[0]["ticker"])
    snap["selected_candidate"] = snap["ticker"].isin(selected)
    snap["decision_reason"] = np.where(snap["selected_candidate"],
        "selected_by_signal_liquidity_risk_and_correlation", "outside_qubit_budget")
    snap["eligible_count"] = len(snap)
    snap["selected_m"] = m
    snap["signal_dispersion"] = signal_dispersion
    snap["relative_signal_dispersion"] = relative_dispersion
    snap["average_abs_correlation"] = average_abs_correlation
    snap["candidate_size_reason"] = "|".join(reasons)
    return snap.sort_values(["selected_candidate", "base_score"], ascending=[False, False])


def qubo_instance(mu: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    return risk_aversion * cov - (1 - risk_aversion) * np.diag(mu)


def qubo_to_ising(q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Map symmetric ``x.T @ Q @ x`` to ``offset + h.z + sum J_ij z_i z_j``.

    The binary/spin convention is x=(1-z)/2 with z in {-1,+1}.
    """
    q = (np.asarray(q, dtype=float) + np.asarray(q, dtype=float).T) / 2
    n = len(q)
    offset = 0.0
    h = np.zeros(n)
    j = np.zeros((n, n))
    for i in range(n):
        offset += q[i, i] / 2
        h[i] -= q[i, i] / 2
        for k in range(i + 1, n):
            offset += q[i, k] / 2
            h[i] -= q[i, k] / 2
            h[k] -= q[i, k] / 2
            j[i, k] = j[k, i] = q[i, k] / 2
    return float(offset), h, j


def ising_energy(spins: np.ndarray, offset: float, h: np.ndarray, j: np.ndarray) -> float:
    pair = sum(j[i, k] * spins[i] * spins[k]
               for i in range(len(spins)) for k in range(i + 1, len(spins)))
    return float(offset + h @ spins + pair)


def ewma_mean_cov(returns: pd.DataFrame, span: int = 60,
                  horizon: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a multivariate EWMA mean vector and covariance matrix.

    Rows are observations ordered from oldest to newest and columns are assets.
    The newest observations receive the largest weights. The estimates are scaled
    to the requested holding horizon.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("EWMA requires at least two observations and one asset.")
    if not np.isfinite(values).all():
        raise ValueError("EWMA input must contain only finite returns.")
    if span <= 1 or horizon <= 0:
        raise ValueError("EWMA span must exceed one and horizon must be positive.")
    alpha = 2.0 / (span + 1.0)
    ages = np.arange(values.shape[0] - 1, -1, -1)
    weights = alpha * np.power(1.0 - alpha, ages)
    weights /= weights.sum()
    mean_daily = weights @ values
    centered = values - mean_daily
    covariance_daily = (centered * weights[:, None]).T @ centered
    # Correct the finite-sample bias of normalized reliability weights.
    denominator = 1.0 - float(weights @ weights)
    if denominator > 1e-12:
        covariance_daily /= denominator
    covariance_daily = (covariance_daily + covariance_daily.T) / 2
    covariance_daily += np.eye(values.shape[1]) * 1e-10
    return mean_daily * horizon, covariance_daily * horizon


def aligned_previous_weights(tickers: list[str], previous: dict[str, float]) -> np.ndarray:
    """Return pre-trade weights aligned to the currently selected tickers."""
    return np.asarray([previous.get(ticker, 0.0) for ticker in tickers], dtype=float)


def portfolio_turnover(previous: dict[str, float],
                       target: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Compute one-way turnover over the union of old and new holdings."""
    names = sorted(set(previous) | set(target))
    trades = {name: target.get(name, 0.0) - previous.get(name, 0.0) for name in names}
    return float(sum(abs(value) for value in trades.values())), trades


def drift_weights(target: dict[str, float], test_returns: pd.DataFrame) -> dict[str, float]:
    """Carry target weights to the next rebalance after realized asset returns."""
    if not target:
        return {}
    columns = list(target)
    aligned = test_returns.reindex(columns=columns).fillna(0.0)
    growth = (1.0 + aligned).prod(axis=0).to_numpy()
    values = np.asarray([target[name] for name in columns]) * growth
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        return target.copy()
    return {name: float(value / total) for name, value in zip(columns, values) if value > 1e-12}


def simulate_buy_and_hold(
    target: dict[str, float], test_returns: pd.DataFrame, transaction_cost: float = 0.0,
) -> dict:
    """Simulate units drifting between rebalances, with an exact initial cost debit."""
    if not target or test_returns.empty:
        empty = pd.Series(dtype=float)
        return {"gross_returns": empty, "net_returns": empty, "ending_weights": target.copy(),
                "gross_wealth": empty, "net_wealth": empty}
    tickers = list(target)
    returns = test_returns.reindex(columns=tickers).fillna(0.0).astype(float)
    weights = np.asarray([target[ticker] for ticker in tickers], dtype=float)
    weights /= weights.sum()
    gross_values = weights.copy()
    net_values = weights * max(0.0, 1.0 - float(transaction_cost))
    gross_wealth, net_wealth = [], []
    for row in returns.to_numpy():
        gross_values *= 1.0 + row
        net_values *= 1.0 + row
        gross_wealth.append(float(gross_values.sum()))
        net_wealth.append(float(net_values.sum()))
    gross_wealth = pd.Series(gross_wealth, index=returns.index, name="gross_wealth")
    net_wealth = pd.Series(net_wealth, index=returns.index, name="net_wealth")
    gross_returns = gross_wealth.pct_change()
    net_returns = net_wealth.pct_change()
    gross_returns.iloc[0] = gross_wealth.iloc[0] - 1.0
    net_returns.iloc[0] = net_wealth.iloc[0] - 1.0
    total = float(net_values.sum())
    ending = ({ticker: float(value / total) for ticker, value in zip(tickers, net_values)}
              if total > 0 else target.copy())
    return {
        "gross_returns": gross_returns, "net_returns": net_returns,
        "ending_weights": ending, "gross_wealth": gross_wealth, "net_wealth": net_wealth,
    }


def record_rebalanced_strategy(
    name: str, fold: dict, target: dict[str, float], test_returns: pd.DataFrame,
    previous_weights: dict[str, dict[str, float]], cost_rate: float,
    weight_rows: list[dict], trade_rows: list[dict], return_rows: list[dict],
) -> dict:
    """Apply one common accounting policy to every benchmark and proposed strategy."""
    previous = previous_weights.setdefault(name, {})
    turnover, changes = portfolio_turnover(previous, target)
    simulation = simulate_buy_and_hold(target, test_returns, cost_rate * turnover)
    previous_weights[name] = simulation["ending_weights"]
    decision_time = fold["test_start"]
    trade_time = test_returns.index.min() if not test_returns.empty else pd.NaT
    for ticker in sorted(set(previous) | set(target)):
        weight_rows.append({
            "fold": fold["fold"], "decision_time": decision_time, "strategy": name,
            "ticker": ticker, "weight": target.get(ticker, 0.0),
            "pre_trade_weight": previous.get(ticker, 0.0),
        })
        trade_rows.append({
            "fold": fold["fold"], "trade_time": trade_time, "strategy": name,
            "ticker": ticker, "pre_trade_weight": previous.get(ticker, 0.0),
            "target_weight": target.get(ticker, 0.0), "trade_weight": changes[ticker],
            "turnover": abs(changes[ticker]),
            "transaction_cost": abs(changes[ticker]) * cost_rate,
        })
    for date in simulation["net_returns"].index:
        return_rows.append({
            "fold": fold["fold"], "date": date, "strategy": name,
            "gross_return": simulation["gross_returns"].loc[date],
            "net_return": simulation["net_returns"].loc[date],
            "return": simulation["net_returns"].loc[date],
        })
    return {"turnover": turnover, "transaction_cost": cost_rate * turnover,
            **simulation}


def energy(bits: np.ndarray, q: np.ndarray) -> float:
    return float(bits @ q @ bits)


def feasible_states(n: int, k: int) -> np.ndarray:
    states = []
    for combo in itertools.combinations(range(n), k):
        b = np.zeros(n, dtype=int)
        b[list(combo)] = 1
        states.append(b)
    return np.asarray(states)


def exact_solver(q: np.ndarray, k: int) -> dict:
    states = feasible_states(len(q), k)
    energies = np.array([energy(s, q) for s in states])
    idx = int(np.argmin(energies))
    return {"method": "exact", "bits": states[idx], "energy": float(energies[idx]),
            "feasibility_rate": 1.0, "runtime_seconds": 0.0}


def simulated_annealing(q: np.ndarray, k: int, seed: int, steps: int = 800) -> dict:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(q)
    bits = np.zeros(n, dtype=int)
    bits[rng.choice(n, k, replace=False)] = 1
    best, best_e = bits.copy(), energy(bits, q)
    cur_e = best_e
    for step in range(steps):
        ones, zeros = np.flatnonzero(bits), np.flatnonzero(1 - bits)
        proposal = bits.copy()
        proposal[rng.choice(ones)] = 0
        proposal[rng.choice(zeros)] = 1
        e = energy(proposal, q)
        temp = max(0.001, 0.1 * (1 - step / steps))
        if e < cur_e or rng.random() < math.exp((cur_e - e) / temp):
            bits, cur_e = proposal, e
        if cur_e < best_e:
            best, best_e = bits.copy(), cur_e
    return {"method": "simulated_annealing", "bits": best, "energy": float(best_e),
            "feasibility_rate": 1.0, "runtime_seconds": time.perf_counter() - start,
            "seed": seed}


def _optimize_qaoa_angles(evaluate, p: int, budget: int, seed: int) -> dict:
    """Deterministic multi-start COBYLA optimization with an auditable trace."""
    rng = np.random.default_rng(seed)
    starts = max(1, min(3, budget // 8))
    per_start = max(5, budget // starts)
    trace = []
    best = None
    bounds = [(0.0, 2 * np.pi)] * p + [(0.0, np.pi)] * p
    for start_id in range(starts):
        x0 = np.r_[rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)]
        evaluations = []

        def objective(params):
            expected, _ = evaluate(params)
            evaluations.append(float(expected))
            return expected

        result = minimize(
            objective, x0, method="COBYLA", bounds=bounds,
            options={"maxiter": per_start, "tol": 1e-7, "catol": 1e-7},
        )
        expected, probabilities = evaluate(result.x)
        trace.append({
            "start": start_id, "initial_parameters": x0.tolist(),
            "final_parameters": result.x.tolist(), "objective_trace": evaluations,
            "final_expected_energy": float(expected), "success": bool(result.success),
            "stopping_reason": str(result.message), "evaluations": int(result.nfev),
        })
        if best is None or expected < best["expected_energy"]:
            best = {
                "expected_energy": float(expected), "probabilities": probabilities,
                "parameters": result.x, "success": bool(result.success),
                "stopping_reason": str(result.message),
            }
    best["trace"] = trace
    best["optimizer"] = "COBYLA_multi_start"
    best["optimizer_budget"] = int(budget)
    return best


def xy_qaoa_statevector(
    q: np.ndarray, k: int, p: int, trials: int, shots: int, seed: int,
    uniform_probability_noise_proxy: float = 0.0,
) -> dict:
    """Ideal statevector simulation in the fixed-Hamming-weight subspace.

    The initial state is the Dicke state. The mixer connects feasible bitstrings that
    differ by swapping one selected and one unselected asset, equivalent to an all-to-all
    XY exchange mixer restricted to the feasible subspace.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    states = feasible_states(len(q), k)
    costs = np.array([energy(s, q) for s in states])
    dim = len(states)
    mixer = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            if np.abs(states[i] - states[j]).sum() == 2:
                mixer[i, j] = mixer[j, i] = 1.0
    mixer_eigenvalues, mixer_eigenvectors = np.linalg.eigh(mixer)
    initial = np.ones(dim, dtype=complex) / np.sqrt(dim)
    cost_scale = max(float(np.max(np.abs(costs))), 1e-12)
    phase_costs = costs / cost_scale

    def evaluate(params):
        gammas, betas = params[:p], params[p:]
        psi = initial.copy()
        for gamma, beta in zip(gammas, betas):
            psi *= np.exp(-1j * gamma * phase_costs)
            coefficients = mixer_eigenvectors.T.conj() @ psi
            psi = mixer_eigenvectors @ (
                np.exp(-1j * beta * mixer_eigenvalues) * coefficients
            )
        probs = np.abs(psi) ** 2
        return float(probs @ costs), probs / probs.sum()

    optimized = _optimize_qaoa_angles(evaluate, p, trials, seed)
    sample_probs = optimized["probabilities"].copy()
    if uniform_probability_noise_proxy:
        level = float(uniform_probability_noise_proxy)
        if not 0 <= level <= 1:
            raise ValueError("uniform_probability_noise_proxy must be in [0, 1].")
        sample_probs = (1 - level) * sample_probs + level * np.ones(dim) / dim
    counts_idx = rng.choice(dim, size=shots, p=sample_probs)
    counts = np.bincount(counts_idx, minlength=dim)
    primary = int(np.argmax(sample_probs))
    best_observed = int(np.argmin(np.where(counts > 0, costs, np.inf)))
    optimal_mask = np.isclose(costs, costs.min(), atol=1e-10, rtol=1e-8)
    bit_counts = {"".join(map(str, states[i])): int(c) for i, c in enumerate(counts) if c}
    bit_probs = {"".join(map(str, states[i])): float(prob)
                 for i, prob in enumerate(sample_probs) if prob > 1e-12}
    return {
        "method": "xy_qaoa_dicke_ideal_statevector", "bits": states[primary],
        "seed": seed,
        "energy": float(costs[primary]), "expected_energy": float(sample_probs @ costs),
        "mean_energy": float(sample_probs @ costs),
        "primary_probability": float(sample_probs[primary]),
        "success_probability": float(sample_probs[optimal_mask].sum()),
        "best_observed_bits": states[best_observed],
        "best_observed_energy": float(costs[best_observed]),
        "feasibility_rate": 1.0, "runtime_seconds": time.perf_counter() - start,
        "shots": shots, "depth_p": p, "two_qubit_gate_estimate": p * len(q) * (len(q) - 1) // 2,
        "bitstring_counts": bit_counts, "backend": "internal_ideal_statevector_fixed_weight",
        "bitstring_probabilities": bit_probs,
        "uniform_probability_noise_proxy": float(uniform_probability_noise_proxy),
        "optimizer": optimized["optimizer"], "optimizer_budget": optimized["optimizer_budget"],
        "optimal_parameters": optimized["parameters"].tolist(),
        "parameter_trace": optimized["trace"],
        "optimizer_success": optimized["success"],
        "stopping_reason": optimized["stopping_reason"],
    }


def penalty_qaoa_baseline(q: np.ndarray, k: int, seed: int, shots: int = 1024) -> dict:
    """Transparent stochastic penalty baseline; not labeled as a circuit simulation."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(q)
    samples = rng.integers(0, 2, size=(shots, n))
    penalty = max(1.0, float(np.abs(q).sum())) * (samples.sum(axis=1) - k) ** 2
    energies = np.array([energy(s, q) for s in samples]) + penalty
    idx = int(np.argmin(energies))
    feasible = samples.sum(axis=1) == k
    return {"method": "penalty_stochastic_baseline", "bits": samples[idx],
            "energy": energy(samples[idx], q), "feasibility_rate": float(feasible.mean()),
            "runtime_seconds": time.perf_counter() - start, "shots": shots, "seed": seed,
            "backend": "classical_stochastic_baseline_not_qaoa_circuit"}


def penalty_qaoa_statevector(
    q: np.ndarray, k: int, p: int, trials: int, shots: int, seed: int,
    penalty_strength: float | None = None,
) -> dict:
    """Ideal full-Hilbert-space penalty-QAOA circuit simulation.

    Uses |+> initialization, diagonal economic+cardinality cost Hamiltonian and
    transverse-X mixer. This is an actual statevector QAOA simulation, not hardware.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(q)
    states = np.array([
        [(index >> (n - 1 - bit)) & 1 for bit in range(n)]
        for index in range(2 ** n)
    ], dtype=int)
    economic = np.array([energy(s, q) for s in states])
    penalty_strength = penalty_strength or max(1.0, 2 * float(np.abs(q).sum()))
    total_cost = economic + penalty_strength * (states.sum(axis=1) - k) ** 2
    dim = len(states)
    initial = np.ones(dim, dtype=complex) / np.sqrt(dim)
    cost_scale = max(float(np.max(np.abs(total_cost))), 1e-12)
    phase_cost = total_cost / cost_scale

    def evaluate(params):
        gammas, betas = params[:p], params[p:]
        psi = initial.copy()
        for gamma, beta in zip(gammas, betas):
            psi *= np.exp(-1j * gamma * phase_cost)
            # exp(-i beta sum X_j) factorizes into independent single-qubit
            # rotations because all X_j terms commute.
            cosine, sine = np.cos(beta), -1j * np.sin(beta)
            for bit in range(n):
                stride = 1 << bit
                for base in range(0, dim, stride * 2):
                    lo = slice(base, base + stride)
                    hi = slice(base + stride, base + 2 * stride)
                    a, b = psi[lo].copy(), psi[hi].copy()
                    psi[lo] = cosine * a + sine * b
                    psi[hi] = sine * a + cosine * b
        probs = np.abs(psi) ** 2
        return float(probs @ total_cost), probs / probs.sum()

    optimized = _optimize_qaoa_angles(evaluate, p, trials, seed)
    probabilities = optimized["probabilities"]
    sampled = rng.choice(dim, size=shots, p=probabilities)
    counts = np.bincount(sampled, minlength=dim)
    feasible = states.sum(axis=1) == k
    feasible_indices = np.flatnonzero(feasible)
    chosen_idx = int(feasible_indices[np.argmax(probabilities[feasible_indices])])
    observed_pool = np.flatnonzero((counts > 0) & feasible)
    best_observed_idx = (
        int(observed_pool[np.argmin(economic[observed_pool])])
        if len(observed_pool) else int(np.argmin(np.where(counts > 0, total_cost, np.inf)))
    )
    bit_counts = {"".join(map(str, states[i])): int(c) for i, c in enumerate(counts) if c}
    sampled_feasible = sum(c for i, c in enumerate(counts) if feasible[i]) / shots
    optimum = economic[feasible].min()
    optimal_mask = feasible & np.isclose(economic, optimum, atol=1e-10, rtol=1e-8)
    return {
        "method": "penalty_qaoa_ideal_statevector", "bits": states[chosen_idx],
        "seed": seed,
        "energy": float(economic[chosen_idx]),
        "expected_energy": float(probabilities @ economic),
        "penalized_mean_energy": optimized["expected_energy"],
        "primary_probability": float(probabilities[chosen_idx]),
        "success_probability": float(probabilities[optimal_mask].sum()),
        "best_observed_bits": states[best_observed_idx],
        "best_observed_energy": float(economic[best_observed_idx]),
        "feasibility_rate": float(sampled_feasible),
        "runtime_seconds": time.perf_counter() - start, "shots": shots, "depth_p": p,
        "two_qubit_gate_estimate": p * (n * (n - 1) // 2 + n),
        "bitstring_counts": bit_counts, "backend": "internal_ideal_statevector_full_hilbert",
        "penalty_strength": penalty_strength,
        "optimizer": optimized["optimizer"], "optimizer_budget": optimized["optimizer_budget"],
        "optimal_parameters": optimized["parameters"].tolist(),
        "parameter_trace": optimized["trace"],
        "optimizer_success": optimized["success"],
        "stopping_reason": optimized["stopping_reason"],
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def paired_block_bootstrap_test(
    a: pd.Series, b: pd.Series, seed: int, samples: int = 500, block: int = 10
) -> dict:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = (joined["a"] - joined["b"]).to_numpy()
    if len(diff) < block * 2:
        return {"mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        starts = rng.integers(0, len(diff) - block + 1, math.ceil(len(diff) / block))
        sample = np.concatenate([diff[s:s + block] for s in starts])[:len(diff)]
        means.append(sample.mean())
    means = np.asarray(means)
    centered = diff - diff.mean()
    null_means = []
    for _ in range(samples):
        starts = rng.integers(0, len(centered) - block + 1, math.ceil(len(centered) / block))
        sample = np.concatenate([centered[s:s + block] for s in starts])[:len(centered)]
        null_means.append(sample.mean())
    null_means = np.asarray(null_means)
    p_value = float((np.abs(null_means) >= abs(diff.mean())).mean())
    return {
        "mean_difference": float(diff.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_value": float(min(1.0, p_value)),
        "effect_size_daily": float(diff.mean()),
        "bootstrap_centered_under_null": True,
    }


def optimize_weights(mu: np.ndarray, cov: np.ndarray, lower: float, upper: float,
                     risk_aversion: float, previous: np.ndarray | None, turnover_penalty: float) -> np.ndarray:
    n = len(mu)
    mu = np.nan_to_num(np.asarray(mu, dtype=float))
    cov = np.nan_to_num(np.asarray(cov, dtype=float))
    cov = (cov + cov.T) / 2 + np.eye(n) * 1e-10
    if n * lower > 1 + 1e-12 or n * upper < 1 - 1e-12:
        raise ValueError(
            f"Infeasible weight bounds for selected cardinality n={n}, "
            f"lower={lower}, upper={upper}"
        )
    prev = np.ones(n) / n if previous is None or len(previous) != n else previous
    def objective(w):
        # Scaling avoids SLSQP terminating on the very small daily-return objective.
        return 1000.0 * (
            risk_aversion * (w @ cov @ w) - mu @ w
            + turnover_penalty * np.sqrt((w - prev) ** 2 + 1e-12).sum()
        )
    result = minimize(objective, np.ones(n) / n, method="SLSQP",
                      bounds=[(lower, upper)] * n,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                      options={"maxiter": 1000, "ftol": 1e-10})
    if result.success and np.isfinite(result.x).all():
        return result.x
    # Deterministic projected-gradient fallback remains a genuine convex classical
    # optimizer and is more stable than silently returning arbitrary weights.
    def project_box_simplex(v):
        lo, hi = np.min(v - upper), np.max(v - lower)
        for _ in range(100):
            mid = (lo + hi) / 2
            w = np.clip(v - mid, lower, upper)
            if w.sum() > 1:
                lo = mid
            else:
                hi = mid
        return np.clip(v - (lo + hi) / 2, lower, upper)
    w = np.ones(n) / n
    lipschitz = max(1e-6, 2 * risk_aversion * np.linalg.eigvalsh(cov).max())
    step = min(0.1, 1 / lipschitz)
    for _ in range(3000):
        smooth_turnover_grad = turnover_penalty * (w - prev) / np.sqrt((w - prev) ** 2 + 1e-12)
        grad = 2 * risk_aversion * cov @ w - mu + smooth_turnover_grad
        updated = project_box_simplex(w - step * grad)
        if np.linalg.norm(updated - w) < 1e-10:
            break
        w = updated
    return w


def financial_metrics(returns: pd.Series, rf_annual: float) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    ann_ret = equity.iloc[-1] ** (252 / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    downside = r[r < 0].std(ddof=1) * np.sqrt(252)
    drawdown = equity / equity.cummax() - 1
    return {
        "cumulative_return": float(equity.iloc[-1] - 1), "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe": float((ann_ret - rf_annual) / ann_vol) if ann_vol else np.nan,
        "sortino": float((ann_ret - rf_annual) / downside) if downside else np.nan,
        "max_drawdown": float(drawdown.min()),
        "calmar": float(ann_ret / abs(drawdown.min())) if drawdown.min() else np.nan,
        "positive_day_ratio": float((r > 0).mean()), "observations": int(len(r)),
    }


def block_bootstrap_sharpe(returns: pd.Series, rf: float, seed: int, samples: int = 300,
                           block: int = 10) -> tuple[float, float]:
    r = returns.dropna().to_numpy()
    if len(r) < block * 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(samples):
        starts = rng.integers(0, len(r) - block + 1, math.ceil(len(r) / block))
        b = np.concatenate([r[s:s + block] for s in starts])[:len(r)]
        vol = b.std(ddof=1) * np.sqrt(252)
        vals.append(((b.mean() * 252 - rf) / vol) if vol else np.nan)
    return tuple(np.nanquantile(vals, [0.025, 0.975]))


def _blocked_experiment(
    project_root: Path, config_path: Path, cfg: dict, quality: dict, leak: dict,
    blockers: list[str], message: str,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    cfg_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:10]
    experiment_id = f"{stamp}-{cfg_hash}-blocked"
    out = project_root / "outputs" / "experiments" / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (out / "leakage_audit.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id, "status": "blocked", "label": cfg.get("label"),
        "mode": cfg.get("mode"), "created_at": datetime.now(timezone.utc).isoformat(),
        "blockers": blockers, "message": message, "config": str(config_path),
        "artifacts": ["RESEARCH_BLOCKED.md", "data_quality.json", "leakage_audit.json",
                      "resolved_config.yaml"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "RESEARCH_BLOCKED.md").write_text(
        "# Research run blocked\n\n"
        f"- Experiment: `{experiment_id}`\n"
        f"- Reason: {message}\n"
        f"- Blockers: `{blockers}`\n\n"
        "No model, backtest, or research metric was produced. The system did not infer or "
        "fabricate historical universe records. Supply verified point-in-time source data, "
        "rebuild the universe snapshots, rerun the leakage audit, and then rerun this config.\n",
        encoding="utf-8",
    )
    return out


def run_experiment(project_root: Path, config_path: Path) -> Path:
    cfg = load_config(config_path)
    paths = Paths(project_root)
    quality, _ = validate_data(paths)
    if cfg.get("mode") != "research" and not (paths.curated / "universe_monthly.parquet").exists():
        build_universe(paths, cfg.get("data", {}).get("rebalance", "monthly"))
    leak = leakage_audit(paths)
    if quality["status"] != "pass":
        message = "Data quality failed; refusing to run."
        blocked = _blocked_experiment(project_root, config_path, cfg, quality, leak,
                                      ["data_quality"], message)
        raise ResearchRunBlocked(message, blocked)
    if cfg.get("mode") == "research" and "fixture" in quality["data_class"]:
        message = ("Research mode refuses fixture data. Supply verified real point-in-time "
                   "data and pass the leakage audit before using a research config.")
        blocked = _blocked_experiment(project_root, config_path, cfg, quality, leak,
                                      ["fixture_data_in_research_mode"], message)
        raise ResearchRunBlocked(message, blocked)
    if cfg.get("mode") == "research" and leak["status"] not in {"pass", "pass_with_limitations"}:
        message = "Historical point-in-time leakage audit blocked the research run."
        blocked = _blocked_experiment(project_root, config_path, cfg, quality, leak,
                                      leak.get("blockers", ["leakage_audit"]), message)
        raise ResearchRunBlocked(message, blocked)
    if leak["status"] == "blocked":
        raise RuntimeError("Leakage audit blocked; refusing to label results.")
    if cfg["reduction"]["candidate_size"] > 8 and cfg.get("mode") != "research":
        raise RuntimeError("Internal exact statevector demo is limited to 8 candidate qubits.")
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["ret1"] = prices.groupby("ticker")["adjusted_close"].pct_change()
    target_horizon = int(cfg.get("target", {}).get(
        "horizon_days", cfg.get("covariance", {}).get("horizon_days", 20)
    ))
    features = attach_point_in_time_features(build_features(prices, target_horizon), paths)
    wf = cfg["walk_forward"]
    folds = make_folds(features["date"], wf["train_months"], wf["validation_months"],
                       wf["test_months"], wf.get("max_folds"),
                       wf.get("embargo_days", cfg.get("target", {}).get("horizon_days", 20)))
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    cfg_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:10]
    experiment_id = f"{stamp}-{cfg_hash}"
    out = project_root / "outputs" / "experiments" / experiment_id
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out / "features.parquet", index=False)
    universe = pd.read_parquet(paths.curated / "universe_monthly.parquet")
    universe["decision_time"] = pd.to_datetime(universe["decision_time"])
    ranking_rows, selection_rows, instance_rows = [], [], []
    solver_rows, weight_rows, trade_rows, return_rows = [], [], [], []
    ablation_rows, sensitivity_rows, fold_audit_rows = [], [], []
    feature_coverage_rows, tuning_rows, aur_diagnostic_rows = [], [], []
    previous_weights: dict[str, dict[str, float]] = {
        "full_pipeline_xy_qaoa": {},
    }
    sensitivity_cfg = cfg.get("sensitivity", {})
    representative_count = min(
        len(folds), int(sensitivity_cfg.get("representative_folds", 1))
    )
    sensitivity_fold_ids = set(
        int(index) for index in np.linspace(0, max(0, len(folds) - 1),
                                           max(1, representative_count)).round()
    )
    sensitivity_seeds = sensitivity_cfg.get("seeds", cfg["solver"]["seeds"][:1])
    for fold in folds:
        train, val, test, fold_audit = purged_fold_frames(features, fold)
        fold_audit_rows.append(fold_audit)
        eligible_snapshots = universe[universe["decision_time"] <= fold["test_start"]]
        if eligible_snapshots.empty:
            continue
        universe_time = eligible_snapshots["decision_time"].max()
        eligible_tickers = set(eligible_snapshots[
            (eligible_snapshots["decision_time"] == universe_time)
            & eligible_snapshots["eligible"].astype(bool)
        ]["ticker"])
        available_features = features[
            (features.date <= fold["test_start"])
            & (features.feature_available_at <= fold["test_start"])
            & features.ticker.isin(eligible_tickers)
        ]
        if available_features.empty:
            continue
        snapshot_date = available_features["date"].max()
        market_features = [
            name for name in FEATURES
            if name not in {"roe_pit", "revenue_growth_yoy_pit", "policy_rate_pit"}
        ]
        snap = available_features[available_features.date == snapshot_date].dropna(
            subset=market_features
        ).copy()
        if len(snap) < cfg["reduction"]["candidate_size"] or test.empty:
            continue
        model_cfg = {**cfg["model"], "seed": cfg.get("seed", 42)}
        bundle = fit_ranker(train, val, model_cfg)
        snap["signal"] = predict(bundle, snap)
        for feature, coverage in bundle["feature_coverage"].items():
            feature_coverage_rows.append({
                "fold": fold["fold"], "feature": feature, "training_coverage": coverage,
                "active": feature in bundle["active_features"],
            })
        for tuning in bundle["tuning"]:
            tuning_rows.append({"fold": fold["fold"], **tuning,
                                "selected": all(tuning[key] == bundle["selected_params"][key]
                                                for key in ("n_estimators", "max_depth"))})
        history = features[
            (features.date <= snapshot_date) & features.ticker.isin(eligible_tickers)
        ].copy()
        history["ret1"] = history.sort_values(["ticker", "date"]).groupby("ticker")[
            "adjusted_close"
        ].pct_change()
        ewma_panel = history.pivot(index="date", columns="ticker", values="ret1").tail(252)
        ewma_signal = ewma_panel.ewm(
            span=int(cfg.get("covariance", {}).get("span", 60)), adjust=False
        ).mean().iloc[-1]
        snap["ewma_signal"] = snap["ticker"].map(ewma_signal)
        known = snap.dropna(subset=["target_rank"])
        ic = spearmanr(known["signal"], known["target_rank"]).statistic if len(known) > 2 else np.nan
        ewma_known = known.dropna(subset=["ewma_signal"])
        ewma_ic = (spearmanr(ewma_known["ewma_signal"], ewma_known["target_rank"]).statistic
                   if len(ewma_known) > 2 else np.nan)
        for row in snap.itertuples():
            ranking_rows.append({"fold": fold["fold"], "decision_time": snapshot_date,
                                 "ticker": row.ticker, "signal": row.signal,
                                 "ewma_signal": row.ewma_signal, "fold_rank_ic": ic,
                                 "xgboost_rank_ic": ic, "ewma_rank_ic": ewma_ic,
                                 "universe_snapshot_time": universe_time})
        reduced = adaptive_reduce(snap, history, cfg["reduction"])
        reduced["fold"] = fold["fold"]
        reduced["decision_time"] = snapshot_date
        selected_now = set(reduced.loc[reduced.selected_candidate, "ticker"])
        previous_candidates = set(aur_diagnostic_rows[-1]["selected_tickers"].split("|")) if aur_diagnostic_rows else set()
        candidate_turnover = (len(selected_now.symmetric_difference(previous_candidates))
                              / max(1, len(selected_now | previous_candidates)))
        aur_diagnostic_rows.append({
            "fold": fold["fold"], "decision_time": snapshot_date,
            "eligible_count": int(reduced["eligible_count"].iloc[0]),
            "selected_m": int(reduced["selected_m"].iloc[0]),
            "signal_dispersion": float(reduced["signal_dispersion"].iloc[0]),
            "relative_signal_dispersion": float(reduced["relative_signal_dispersion"].iloc[0]),
            "average_abs_correlation": float(reduced["average_abs_correlation"].iloc[0]),
            "candidate_size_reason": reduced["candidate_size_reason"].iloc[0],
            "candidate_turnover": candidate_turnover,
            "selected_tickers": "|".join(sorted(selected_now)),
        })
        selection_columns = [
            "fold", "decision_time", "ticker", "signal", "ewma_signal", "liquidity_20d",
            "volatility_20d", "signal_z", "liquidity_z", "risk_z", "base_score",
            "selected_candidate", "decision_reason", "eligible_count", "selected_m",
            "signal_dispersion", "relative_signal_dispersion", "average_abs_correlation",
            "candidate_size_reason",
        ]
        selection_rows.extend(reduced[selection_columns].to_dict("records"))
        candidates = reduced[reduced.selected_candidate]["ticker"].tolist()
        hist_returns = history[history.ticker.isin(candidates)].pivot(
            index="date", columns="ticker", values="ret1").tail(252).dropna()
        candidates = [c for c in candidates if c in hist_returns.columns]
        hist_returns = hist_returns[candidates]
        covariance_cfg = cfg.get("covariance", {})
        covariance_method = covariance_cfg.get("method", "ewma")
        covariance_span = int(covariance_cfg.get("span", 60))
        holding_horizon = int(covariance_cfg.get("horizon_days", 20))
        if covariance_method == "ewma":
            mu, cov = ewma_mean_cov(hist_returns, covariance_span, holding_horizon)
        elif covariance_method == "ledoit_wolf":
            mu = hist_returns.mean().to_numpy() * holding_horizon
            cov = LedoitWolf().fit(hist_returns.to_numpy()).covariance_ * holding_horizon
        else:
            raise ValueError(f"Unsupported covariance method: {covariance_method}")
        q = qubo_instance(mu, cov, cfg["qubo"]["risk_aversion"])
        ising_offset, ising_h, ising_j = qubo_to_ising(q)
        k = min(cfg["reduction"]["cardinality"], len(candidates))
        instance_rows.append({
            "fold": fold["fold"], "decision_time": str(snapshot_date),
            "tickers": candidates, "cardinality": k,
            "expected_return": mu.tolist(), "covariance": cov.tolist(), "qubo_matrix": q.tolist(),
            "ising_offset": ising_offset, "ising_h": ising_h.tolist(),
            "ising_j": ising_j.tolist(), "binary_spin_convention": "x=(1-z)/2",
            "covariance_method": covariance_method, "covariance_span": covariance_span,
            "holding_horizon_days": holding_horizon,
        })
        exact = exact_solver(q, k)
        xy_runs = [
            xy_qaoa_statevector(
                q, k, cfg["solver"]["qaoa_depth"], cfg["solver"]["parameter_trials"],
                cfg["solver"]["shots"], seed,
            ) for seed in cfg["solver"]["seeds"]
        ]
        penalty_runs = [
            penalty_qaoa_statevector(
                q, k, cfg["solver"]["qaoa_depth"], cfg["solver"]["parameter_trials"],
                cfg["solver"]["shots"], seed,
            ) for seed in cfg["solver"]["seeds"]
        ]
        annealing_runs = [
            simulated_annealing(q, k, seed) for seed in cfg["solver"]["seeds"]
        ]
        runs = [
            exact,
            *annealing_runs,
            penalty_qaoa_baseline(q, k, cfg["solver"]["seeds"][0], cfg["solver"]["shots"]),
            *penalty_runs,
            *xy_runs,
        ]
        chosen_xy = min(xy_runs, key=lambda result: result["expected_energy"])
        chosen_xy_bits = np.asarray(chosen_xy["bits"]).copy()
        for run in runs:
            primary_bits = np.asarray(run["bits"]).copy()
            run["fold"] = fold["fold"]
            run["decision_time"] = str(snapshot_date)
            run["optimality_gap"] = float((run["energy"] - exact["energy"]) / (abs(exact["energy"]) + 1e-12))
            run["selected_tickers"] = [candidates[i] for i in np.flatnonzero(primary_bits)]
            run["bits"] = "".join(map(str, primary_bits))
            if "best_observed_bits" in run:
                run["best_observed_bits"] = "".join(map(str, run["best_observed_bits"]))
                run["best_observed_gap"] = float(
                    (run["best_observed_energy"] - exact["energy"])
                    / (abs(exact["energy"]) + 1e-12)
                )
            run["bitstring_counts"] = json.dumps(run.get("bitstring_counts", {}))
            run["bitstring_probabilities"] = json.dumps(run.get("bitstring_probabilities", {}))
            run["parameter_trace"] = json.dumps(run.get("parameter_trace", []))
            run["optimal_parameters"] = json.dumps(run.get("optimal_parameters", []))
            solver_rows.append(run)
        chosen = [candidates[i] for i in np.flatnonzero(chosen_xy_bits)]
        idx = [candidates.index(c) for c in chosen]
        strategy_name = "full_pipeline_xy_qaoa"
        previous = previous_weights[strategy_name]
        previous_selected = aligned_previous_weights(chosen, previous)
        weights = optimize_weights(mu[idx], cov[np.ix_(idx, idx)], cfg["weights"]["lower"],
            cfg["weights"]["upper"], cfg["weights"]["risk_aversion"], previous_selected,
            cfg["weights"]["turnover_penalty"])
        cost = cfg["backtest"]["transaction_cost_bps"] / 10000
        target = {ticker: float(w) for ticker, w in zip(chosen, weights)}
        turnover, trade_changes = portfolio_turnover(previous, target)
        transaction_cost = cost * turnover
        for ticker in sorted(set(previous) | set(target)):
            w = target.get(ticker, 0.0)
            weight_rows.append({"fold": fold["fold"], "decision_time": snapshot_date,
                                "strategy": strategy_name, "ticker": ticker, "weight": w,
                                "pre_trade_weight": previous.get(ticker, 0.0)})
            trade_rows.append({"fold": fold["fold"], "trade_time": test.date.min(),
                               "strategy": strategy_name, "ticker": ticker,
                               "pre_trade_weight": previous.get(ticker, 0.0),
                               "target_weight": w, "trade_weight": trade_changes[ticker],
                               "turnover": abs(trade_changes[ticker]),
                               "transaction_cost": abs(trade_changes[ticker]) * cost})
        test_ret = test[test.ticker.isin(chosen)].pivot(index="date", columns="ticker",
                                                        values="ret1").reindex(columns=chosen).fillna(0)
        simulation = simulate_buy_and_hold(target, test_ret, transaction_cost)
        previous_weights[strategy_name] = simulation["ending_weights"]
        for date in simulation["net_returns"].index:
            return_rows.append({
                "fold": fold["fold"], "date": date, "strategy": strategy_name,
                "gross_return": simulation["gross_returns"].loc[date],
                "net_return": simulation["net_returns"].loc[date],
                "return": simulation["net_returns"].loc[date],
            })
        equal_name = "equal_weight_candidates"
        equal_previous = previous_weights.setdefault(equal_name, {})
        equal_target = {ticker: 1 / len(candidates) for ticker in candidates}
        equal_turnover, equal_trades = portfolio_turnover(equal_previous, equal_target)
        equal_test = test[test.ticker.isin(candidates)].pivot(
            index="date", columns="ticker", values="ret1"
        ).reindex(columns=candidates).fillna(0)
        equal_sim = simulate_buy_and_hold(equal_target, equal_test, cost * equal_turnover)
        previous_weights[equal_name] = equal_sim["ending_weights"]
        for ticker, change in equal_trades.items():
            weight_rows.append({
                "fold": fold["fold"], "decision_time": snapshot_date,
                "strategy": equal_name, "ticker": ticker,
                "weight": equal_target.get(ticker, 0.0),
                "pre_trade_weight": equal_previous.get(ticker, 0.0),
            })
            trade_rows.append({
                "fold": fold["fold"], "trade_time": test.date.min(), "strategy": equal_name,
                "ticker": ticker, "pre_trade_weight": equal_previous.get(ticker, 0.0),
                "target_weight": equal_target.get(ticker, 0.0), "trade_weight": change,
                "turnover": abs(change), "transaction_cost": abs(change) * cost,
            })
        for date in equal_sim["net_returns"].index:
            return_rows.append({
                "fold": fold["fold"], "date": date, "strategy": equal_name,
                "gross_return": equal_sim["gross_returns"].loc[date],
                "net_return": equal_sim["net_returns"].loc[date],
                "return": equal_sim["net_returns"].loc[date],
            })
        # Eight pre-declared ablations use their own candidate construction and solver.
        ablation_specs = [
            ("liquidity_topk_exact", "liquidity", "exact"),
            ("ewma_topk_exact", "ewma", "exact"),
            ("xgboost_topk_exact", "xgboost", "exact"),
            ("adaptive_exact", "adaptive", "exact"),
            ("xgboost_penalty_qaoa", "xgboost", "penalty"),
            ("xgboost_xy_qaoa", "xgboost", "xy"),
            ("adaptive_penalty_qaoa", "adaptive", "penalty"),
            ("adaptive_xy_qaoa", "adaptive", "xy"),
            ("adaptive_simulated_annealing", "adaptive", "sa"),
        ]
        m = min(cfg["reduction"]["candidate_size"], len(snap))
        for ablation_name, selector, solver_name in ablation_specs:
            if selector == "liquidity":
                pool = snap.nlargest(m, "liquidity_20d")["ticker"].tolist()
            elif selector == "ewma":
                ewma_universe = history[history.ticker.isin(snap["ticker"])].pivot(
                    index="date", columns="ticker", values="ret1"
                ).tail(252)
                ewma_scores = ewma_universe.ewm(
                    span=covariance_span, adjust=False
                ).mean().iloc[-1]
                pool = ewma_scores.nlargest(m).index.tolist()
            elif selector == "xgboost":
                pool = snap.nlargest(m, "signal")["ticker"].tolist()
            else:
                pool = candidates
            variant_hist = history[history.ticker.isin(pool)].pivot(
                index="date", columns="ticker", values="ret1"
            ).tail(252).dropna()
            pool = [ticker for ticker in pool if ticker in variant_hist.columns]
            variant_hist = variant_hist[pool]
            if len(pool) < 2 or variant_hist.empty:
                continue
            if covariance_method == "ewma":
                variant_mu, variant_cov = ewma_mean_cov(
                    variant_hist, covariance_span, holding_horizon
                )
            else:
                variant_mu = variant_hist.mean().to_numpy() * holding_horizon
                variant_cov = LedoitWolf().fit(
                    variant_hist.to_numpy()
                ).covariance_ * holding_horizon
            variant_q = qubo_instance(variant_mu, variant_cov, cfg["qubo"]["risk_aversion"])
            variant_k = min(cfg["reduction"]["cardinality"], len(pool))
            exact_variant = exact_solver(variant_q, variant_k)
            if solver_name == "penalty":
                chosen_run = penalty_qaoa_statevector(
                    variant_q, variant_k, cfg["solver"]["qaoa_depth"],
                    max(8, cfg["solver"]["parameter_trials"] // 2),
                    cfg["solver"]["shots"], cfg["solver"]["seeds"][0],
                )
            elif solver_name == "xy":
                chosen_run = xy_qaoa_statevector(
                    variant_q, variant_k, cfg["solver"]["qaoa_depth"],
                    max(8, cfg["solver"]["parameter_trials"] // 2),
                    cfg["solver"]["shots"], cfg["solver"]["seeds"][0],
                )
            elif solver_name == "sa":
                chosen_run = simulated_annealing(
                    variant_q, variant_k, cfg["solver"]["seeds"][0]
                )
            else:
                chosen_run = exact_variant
            selected = [pool[i] for i in np.flatnonzero(chosen_run["bits"])]
            selected_idx = [pool.index(ticker) for ticker in selected]
            strategy_previous = previous_weights.setdefault(ablation_name, {})
            previous_selected = aligned_previous_weights(selected, strategy_previous)
            variant_weights = optimize_weights(
                variant_mu[selected_idx], variant_cov[np.ix_(selected_idx, selected_idx)],
                cfg["weights"]["lower"], cfg["weights"]["upper"],
                cfg["weights"]["risk_aversion"], previous_selected,
                cfg["weights"]["turnover_penalty"],
            )
            variant_test = test[test.ticker.isin(selected)].pivot(
                index="date", columns="ticker", values="ret1"
            ).reindex(columns=selected).fillna(0)
            variant_target = {
                ticker: float(weight) for ticker, weight in zip(selected, variant_weights)
            }
            variant_turnover, variant_trades = portfolio_turnover(strategy_previous, variant_target)
            variant_cost = cost * variant_turnover
            variant_sim = simulate_buy_and_hold(variant_target, variant_test, variant_cost)
            previous_weights[ablation_name] = variant_sim["ending_weights"]
            for ticker, change in variant_trades.items():
                weight_rows.append({
                    "fold": fold["fold"], "decision_time": snapshot_date,
                    "strategy": ablation_name, "ticker": ticker,
                    "weight": variant_target.get(ticker, 0.0),
                    "pre_trade_weight": strategy_previous.get(ticker, 0.0),
                })
                trade_rows.append({
                    "fold": fold["fold"], "trade_time": test.date.min(),
                    "strategy": ablation_name, "ticker": ticker,
                    "pre_trade_weight": strategy_previous.get(ticker, 0.0),
                    "target_weight": variant_target.get(ticker, 0.0),
                    "trade_weight": change, "turnover": abs(change),
                    "transaction_cost": abs(change) * cost,
                })
            gap = (chosen_run["energy"] - exact_variant["energy"]) / (
                abs(exact_variant["energy"]) + 1e-12
            )
            ablation_rows.append({
                "fold": fold["fold"], "configuration": ablation_name,
                "selector": selector, "solver": chosen_run["method"],
                "selected_tickers": "|".join(selected), "objective": chosen_run["energy"],
                "optimality_gap": gap, "feasibility_rate": chosen_run["feasibility_rate"],
                "turnover": variant_turnover, "transaction_cost": variant_cost,
                "covariance_method": covariance_method,
            })
            for date in variant_sim["net_returns"].index:
                return_rows.append({
                    "fold": fold["fold"], "date": date,
                    "strategy": ablation_name,
                    "gross_return": variant_sim["gross_returns"].loc[date],
                    "net_return": variant_sim["net_returns"].loc[date],
                    "return": variant_sim["net_returns"].loc[date],
                })
        # Independent classical benchmarks use the complete eligible universe at the
        # same decision time, schedule, holding convention, and transaction-cost model.
        benchmark_hist = history[history.ticker.isin(snap["ticker"])].pivot(
            index="date", columns="ticker", values="ret1"
        ).tail(252)
        minimum_observations = min(60, max(2, len(benchmark_hist) // 2))
        benchmark_hist = benchmark_hist.dropna(axis=1, thresh=minimum_observations).fillna(0.0)
        benchmark_tickers = benchmark_hist.columns.tolist()
        if len(benchmark_tickers) >= 2:
            benchmark_mu, benchmark_cov = ewma_mean_cov(
                benchmark_hist, covariance_span, holding_horizon
            )
            benchmark_test = test[test.ticker.isin(benchmark_tickers)].pivot(
                index="date", columns="ticker", values="ret1"
            ).reindex(columns=benchmark_tickers).fillna(0.0)
            benchmark_targets = {
                "equal_weight_universe": np.ones(len(benchmark_tickers)) / len(benchmark_tickers),
                "markowitz_mean_variance": optimize_weights(
                    benchmark_mu, benchmark_cov, cfg["weights"]["lower"],
                    cfg["weights"]["upper"], cfg["weights"]["risk_aversion"],
                    aligned_previous_weights(
                        benchmark_tickers, previous_weights.get("markowitz_mean_variance", {})
                    ), cfg["weights"]["turnover_penalty"],
                ),
                "minimum_variance": optimize_weights(
                    np.zeros(len(benchmark_tickers)), benchmark_cov, cfg["weights"]["lower"],
                    cfg["weights"]["upper"], 1.0,
                    aligned_previous_weights(
                        benchmark_tickers, previous_weights.get("minimum_variance", {})
                    ), cfg["weights"]["turnover_penalty"],
                ),
            }
            for benchmark_name, benchmark_weights in benchmark_targets.items():
                record_rebalanced_strategy(
                    benchmark_name, fold,
                    {ticker: float(weight) for ticker, weight in zip(
                        benchmark_tickers, benchmark_weights
                    )}, benchmark_test, previous_weights, cost,
                    weight_rows, trade_rows, return_rows,
                )
        if fold["fold"] in sensitivity_fold_ids:
            sensitivity_seed = int(
                sensitivity_seeds[fold["fold"] % len(sensitivity_seeds)]
            )
            for depth in sorted({1, cfg["solver"]["qaoa_depth"], 2}):
                for shots in sorted({256, cfg["solver"]["shots"]}):
                    for sensitivity_k in sorted({max(1, k - 1), k}):
                        for noise in (0.0, 0.02):
                            sensitivity = xy_qaoa_statevector(
                                q, sensitivity_k, depth,
                                max(6, cfg["solver"]["parameter_trials"] // 3),
                                shots, sensitivity_seed,
                                uniform_probability_noise_proxy=noise,
                            )
                            exact_sensitivity = exact_solver(q, sensitivity_k)
                            sensitivity_selected = [
                                candidates[i] for i in np.flatnonzero(sensitivity["bits"])
                            ]
                            sensitivity_test = test[test.ticker.isin(sensitivity_selected)].pivot(
                                index="date", columns="ticker", values="ret1"
                            ).reindex(columns=sensitivity_selected).fillna(0)
                            sensitivity_target = {
                                ticker: 1 / len(sensitivity_selected)
                                for ticker in sensitivity_selected
                            }
                            for sensitivity_cost in (0, cfg["backtest"]["transaction_cost_bps"], 25):
                                sensitivity_sim = simulate_buy_and_hold(
                                    sensitivity_target, sensitivity_test,
                                    sensitivity_cost / 10000,
                                )
                                sensitivity_rows.append({
                                    "sensitivity_factor": "core_partial_factorial",
                                    "fold": fold["fold"], "depth_p": depth, "shots": shots,
                                    "seed": sensitivity_seed,
                                    "cardinality": sensitivity_k, "candidate_size": len(candidates),
                                    "qubit_budget": cfg["reduction"]["qubit_budget"],
                                    "uniform_probability_noise_proxy": noise,
                                    "transaction_cost_bps": sensitivity_cost,
                                    "energy": sensitivity["energy"],
                                    "optimality_gap": (sensitivity["energy"] - exact_sensitivity["energy"]) /
                                                      (abs(exact_sensitivity["energy"]) + 1e-12),
                                    "feasibility_rate": sensitivity["feasibility_rate"],
                                    "runtime_seconds": sensitivity["runtime_seconds"],
                                    "net_cumulative_return": float(
                                        (1 + sensitivity_sim["net_returns"]).prod() - 1
                                    ),
                                })
            for sensitivity_n in sorted({max(k, len(candidates) - 2), len(candidates)}):
                q_reduced = q[:sensitivity_n, :sensitivity_n]
                k_reduced = min(k, sensitivity_n)
                size_run = xy_qaoa_statevector(
                    q_reduced, k_reduced, cfg["solver"]["qaoa_depth"],
                    max(6, cfg["solver"]["parameter_trials"] // 3),
                    cfg["solver"]["shots"], sensitivity_seed,
                )
                size_exact = exact_solver(q_reduced, k_reduced)
                size_selected = [
                    candidates[:sensitivity_n][i] for i in np.flatnonzero(size_run["bits"])
                ]
                size_test = test[test.ticker.isin(size_selected)].pivot(
                    index="date", columns="ticker", values="ret1"
                ).reindex(columns=size_selected).fillna(0)
                size_target = {ticker: 1 / len(size_selected) for ticker in size_selected}
                size_sim = simulate_buy_and_hold(
                    size_target, size_test, cfg["backtest"]["transaction_cost_bps"] / 10000
                )
                sensitivity_rows.append({
                    "sensitivity_factor": "candidate_size_and_qubit_budget",
                    "fold": fold["fold"], "depth_p": cfg["solver"]["qaoa_depth"],
                    "shots": cfg["solver"]["shots"], "seed": sensitivity_seed,
                    "cardinality": k_reduced, "candidate_size": sensitivity_n,
                    "qubit_budget": sensitivity_n,
                    "uniform_probability_noise_proxy": 0.0,
                    "transaction_cost_bps": cfg["backtest"]["transaction_cost_bps"],
                    "energy": size_run["energy"],
                    "optimality_gap": (size_run["energy"] - size_exact["energy"])
                    / (abs(size_exact["energy"]) + 1e-12),
                    "feasibility_rate": size_run["feasibility_rate"],
                    "runtime_seconds": size_run["runtime_seconds"],
                    "net_cumulative_return": float((1 + size_sim["net_returns"]).prod() - 1),
                })
    rankings = pd.DataFrame(ranking_rows)
    selections = pd.DataFrame(selection_rows)
    solvers = pd.DataFrame(solver_rows)
    weights_df, trades, returns = pd.DataFrame(weight_rows), pd.DataFrame(trade_rows), pd.DataFrame(return_rows)
    rankings.to_csv(out / "rankings.csv", index=False)
    selections.to_csv(out / "selected_universe.csv", index=False)
    solvers.to_csv(out / "solver_runs.csv", index=False)
    (out / "optimization_instances.json").write_text(
        json.dumps(instance_rows, indent=2), encoding="utf-8"
    )
    weights_df.to_csv(out / "weights.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    cost_ledger = trades.groupby(["fold", "strategy"], as_index=False).agg(
        turnover=("turnover", "sum"), transaction_cost=("transaction_cost", "sum")
    ) if not trades.empty else pd.DataFrame(
        columns=["fold", "strategy", "turnover", "transaction_cost"]
    )
    cost_ledger.to_csv(out / "cost_ledger.csv", index=False)
    returns.to_csv(out / "portfolio_returns.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out / "ablation_results.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out / "sensitivity_results.csv", index=False)
    pd.DataFrame(fold_audit_rows).to_csv(out / "fold_manifest.csv", index=False)
    pd.DataFrame(feature_coverage_rows).to_csv(out / "feature_coverage_by_fold.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out / "model_tuning.csv", index=False)
    pd.DataFrame(aur_diagnostic_rows).to_csv(out / "aur_diagnostics.csv", index=False)
    metric_rows = []
    for strategy, g in returns.groupby("strategy"):
        metrics = financial_metrics(g.sort_values("date")["return"], cfg["backtest"]["risk_free_annual"])
        lo, hi = block_bootstrap_sharpe(g["return"], cfg["backtest"]["risk_free_annual"], cfg["seed"])
        metrics.update({"strategy": strategy, "sharpe_ci_low": lo, "sharpe_ci_high": hi})
        metric_rows.append(metrics)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out / "metrics_long.csv", index=False)
    # Descriptive regime analysis uses only trailing market information.
    market_daily = (
        prices.pivot(index="date", columns="ticker", values="ret1").mean(axis=1).sort_index()
    )
    trailing_60 = (1 + market_daily).rolling(60).apply(np.prod, raw=True) - 1
    trailing_vol = market_daily.rolling(60).std()
    vol_median = trailing_vol.expanding(min_periods=60).median()
    regime = pd.Series("sideway", index=market_daily.index)
    regime[trailing_60 > 0.05] = "bull"
    regime[trailing_60 < -0.05] = "bear"
    regime = regime + np.where(trailing_vol > vol_median, "_high_vol", "_low_vol")
    regime_rows = []
    returns_with_regime = returns.copy()
    returns_with_regime["regime"] = pd.to_datetime(returns_with_regime["date"]).map(regime)
    for (strategy, label), group in returns_with_regime.groupby(["strategy", "regime"], dropna=False):
        row = financial_metrics(group["return"], cfg["backtest"]["risk_free_annual"])
        row.update({"strategy": strategy, "regime": label})
        regime_rows.append(row)
    pd.DataFrame(regime_rows).to_csv(out / "regime_metrics.csv", index=False)
    comparisons = solvers.groupby("method").agg(
        energy_mean=("energy", "mean"), feasibility_rate=("feasibility_rate", "mean"),
        optimality_gap_mean=("optimality_gap", "mean"), runtime_seconds=("runtime_seconds", "mean"),
        runs=("method", "size")).reset_index()
    comparisons.to_csv(out / "comparisons.csv", index=False)
    tests = []
    returns_wide = returns.pivot_table(index="date", columns="strategy", values="return", aggfunc="mean")
    if "full_pipeline_xy_qaoa" in returns_wide:
        for baseline in ["equal_weight_universe", "markowitz_mean_variance", "minimum_variance",
                         "liquidity_topk_exact", "ewma_topk_exact", "adaptive_exact",
                         "adaptive_simulated_annealing", "adaptive_penalty_qaoa"]:
            if baseline in returns_wide:
                result = paired_block_bootstrap_test(
                    returns_wide["full_pipeline_xy_qaoa"], returns_wide[baseline], cfg["seed"]
                )
                result.update({"test": f"full_pipeline_xy_qaoa_vs_{baseline}",
                               "hypothesis": "H5"})
                tests.append(result)
    fold_ic = rankings.groupby("fold")[["xgboost_rank_ic", "ewma_rank_ic"]].first().dropna()
    if len(fold_ic) >= 2:
        result = paired_block_bootstrap_test(
            fold_ic["xgboost_rank_ic"], fold_ic["ewma_rank_ic"], cfg["seed"],
            samples=1000, block=max(1, min(3, len(fold_ic) // 2)),
        )
        result.update({"test": "xgboost_rank_ic_vs_ewma_rank_ic", "hypothesis": "H1"})
        tests.append(result)
    finite_indices = [i for i, row in enumerate(tests) if np.isfinite(row["p_value"])]
    adjusted = holm_adjust([tests[i]["p_value"] for i in finite_indices])
    for i, value in zip(finite_indices, adjusted):
        tests[i]["p_value_holm"] = value
        tests[i]["conclusion"] = "significant" if value < 0.05 else "not_significant"
    for row in tests:
        row.setdefault("p_value_holm", np.nan)
        row.setdefault("conclusion", "insufficient_observations")
    pd.DataFrame(tests or [{"test": "no_valid_pair", "p_value": np.nan,
                            "conclusion": "insufficient_observations"}]).to_csv(
        out / "statistical_tests.csv", index=False
    )
    if not returns.empty:
        pivot = returns.pivot_table(index="date", columns="strategy", values="return", aggfunc="mean")
        chart_title = (
            "Walk-forward cumulative wealth — real HOSE data"
            if cfg.get("mode") == "research"
            else "Demo cumulative wealth — fixture"
        )
        (1 + pivot).cumprod().plot(title=chart_title)
        plt.ylabel("Growth of 1 unit")
        plt.tight_layout()
        plt.savefig(fig_dir / "equity_curve.png", dpi=160)
        plt.close()
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (out / "leakage_audit.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")
    (out / "data_quality.md").write_text(
        "# Data quality\n\n"
        f"- Status: `{quality['status']}`\n- Class: `{quality['data_class']}`\n"
        f"- Records: {quality['records']}\n- Tickers: {quality['tickers']}\n"
        f"- Issues: `{quality['issues']}`\n", encoding="utf-8"
    )
    (out / "leakage_audit.md").write_text(
        "# Leakage audit\n\n"
        f"- Status: `{leak['status']}`\n- Blockers: `{leak['blockers']}`\n\n"
        f"{leak['note']}\n", encoding="utf-8"
    )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    env = f"python={sys.version}\nplatform={platform.platform()}\ngit_commit={git_commit}\n"
    (out / "environment.txt").write_text(env, encoding="utf-8")
    provenance = {
        "price_sources": prices.groupby(
            ["source", "source_url", "data_class"], dropna=False
        ).size().reset_index(name="records").to_dict("records"),
        "price_dataset_sha256": sha256_file(paths.normalized / "prices.parquet"),
        "universe_dataset_sha256": sha256_file(paths.curated / "universe_monthly.parquet"),
        "source_manifest": str(paths.raw / "manifest.json"),
    }
    (out / "data_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    actual_oos_start = str(pd.to_datetime(returns["date"]).min().date()) if not returns.empty else None
    actual_oos_end = str(pd.to_datetime(returns["date"]).max().date()) if not returns.empty else None
    manifest = {
        "experiment_id": experiment_id, "status": "success", "mode": cfg.get("mode"),
        "label": cfg["label"],
        "data_class": quality["data_class"], "started_from_config": str(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(), "config_hash": cfg_hash,
        "dataset_hash": sha256_file(paths.normalized / "prices.parquet"),
        "universe_hash": sha256_file(paths.curated / "universe_monthly.parquet"),
        "git_commit": git_commit,
        "requested_data_start": cfg.get("data", {}).get("start"),
        "requested_data_end": cfg.get("data", {}).get("end"),
        "actual_data_start": quality.get("start"), "actual_data_end": quality.get("end"),
        "actual_oos_start": actual_oos_start, "actual_oos_end": actual_oos_end,
        "folds_requested": len(folds), "folds_completed": int(returns["fold"].nunique()) if not returns.empty else 0,
        "artifacts": [],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    create_report(out, cfg, quality, leak, metrics_df, comparisons, rankings)
    report_kind = "Research run report" if cfg.get("mode") == "research" else "Demo run report"
    run_report = [
        f"# {report_kind}", "", f"- Experiment: `{experiment_id}`",
        f"- Status: `success`", f"- Label: **{cfg['label']}**",
        f"- Data records: {quality['records']}", f"- Tickers: {quality['tickers']}",
        f"- Folds requested/completed: {len(folds)}/{manifest['folds_completed']}",
        f"- Data quality: `{quality['status']}`", f"- Leakage audit: `{leak['status']}`",
        f"- Command: `python -m src.cli run-experiment --config {config_path}`",
        "- Quantum backend: internal ideal fixed-Hamming-weight statevector simulator; not hardware.",
        f"- Limitations: {', '.join(leak.get('limitations', [])) or 'none reported'}.",
        "", "## Artifact index", "",
    ]
    current = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    run_report.extend(f"- `{name}`" for name in current)
    run_report_name = "RUN_REPORT.md" if cfg.get("mode") == "research" else "DEMO_RUN_REPORT.md"
    (out / run_report_name).write_text("\n".join(run_report) + "\n", encoding="utf-8")
    manifest["artifacts"] = sorted(
        p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def create_report(out: Path, cfg: dict, quality: dict, leak: dict, metrics: pd.DataFrame,
                  comparisons: pd.DataFrame, rankings: pd.DataFrame) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "No results."
        clean = df.copy()
        for col in clean.select_dtypes(include=[np.number]).columns:
            clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        header = "| " + " | ".join(map(str, clean.columns)) + " |"
        rule = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in clean.astype(str).to_numpy()]
        return "\n".join([header, rule, *rows])
    label = cfg["label"]
    ic = rankings.groupby("fold")["fold_rank_ic"].first().mean() if not rankings.empty else np.nan
    ablations = pd.read_csv(out / "ablation_results.csv") if (out / "ablation_results.csv").exists() else pd.DataFrame()
    sensitivity = pd.read_csv(out / "sensitivity_results.csv") if (out / "sensitivity_results.csv").exists() else pd.DataFrame()
    statistics = pd.read_csv(out / "statistical_tests.csv") if (out / "statistical_tests.csv").exists() else pd.DataFrame()
    regimes = pd.read_csv(out / "regime_metrics.csv") if (out / "regime_metrics.csv").exists() else pd.DataFrame()
    is_research = cfg.get("mode") == "research"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    h1_test = (statistics[statistics["hypothesis"] == "H1"]
               if "hypothesis" in statistics else pd.DataFrame())
    h5_tests = (statistics[statistics["hypothesis"] == "H5"]
                if "hypothesis" in statistics else pd.DataFrame())
    interpretation_prefix = "research" if is_research else "demo-only"
    hypotheses = [
        ("H1", interpretation_prefix,
         f"Mean XGBoost walk-forward Rank IC={ic:.4f}; paired XGBoost–EWMA test rows={len(h1_test)}."),
        ("H2", interpretation_prefix,
         "AUR diagnostics report signal, liquidity, risk, correlation, selected M and candidate turnover; causal superiority is not inferred."),
        ("H3", "implementation-supported" if is_research else "demo-only",
         "Fixed-weight XY simulation preserves cardinality by construction; penalty feasibility is measured from samples."),
        ("H4", interpretation_prefix,
         "Primary-solution and best-observed gaps are separated against the exact small-instance oracle."),
        ("H5", interpretation_prefix,
         f"Net buy-and-hold performance uses common costs; paired benchmark comparisons={len(h5_tests)}."),
        ("H6", interpretation_prefix,
         "Sensitivity reruns solver/accounting on the declared grid and representative folds; inference is conditional on that grid."),
    ]
    title = "AI–Quantum Portfolio Research Report" if is_research else "AI–Quantum Portfolio Demo Report"
    scope = (
        "This run evaluates the complete walk-forward pipeline on the normalized real-market price panel."
        if is_research
        else "This run verifies the software path end-to-end. It is not evidence for the 2015–2025 HOSE study."
    )
    lines = [
        f"# {title}", "", f"> **{label}**", "",
        "## Scope", "",
        scope,
        "", "## Data validation", "", f"- Quality: `{quality['status']}`",
        f"- Leakage audit: `{leak['status']}`", f"- Records: {quality['records']}",
        f"- Tickers: {quality['tickers']}",
        f"- Requested range: {manifest.get('requested_data_start')} to {manifest.get('requested_data_end')}",
        f"- Actual data range: {manifest.get('actual_data_start')} to {manifest.get('actual_data_end')}",
        f"- Actual OOS range: {manifest.get('actual_oos_start')} to {manifest.get('actual_oos_end')}",
        f"- Folds completed/requested: {manifest.get('folds_completed')}/{manifest.get('folds_requested')}",
        "", "## Predictive ranking", "",
        f"- Mean fold rank IC: {ic:.6f}", "", "## Solver comparison", "",
        markdown_table(comparisons),
        "", "## Portfolio metrics", "",
        markdown_table(metrics),
        "", "## Ablation study", "",
        markdown_table(
            ablations.groupby(["configuration", "selector", "solver"]).agg(
                objective_mean=("objective", "mean"),
                optimality_gap_mean=("optimality_gap", "mean"),
                feasibility_rate=("feasibility_rate", "mean"),
                folds=("fold", "nunique"),
            ).reset_index()
        ) if not ablations.empty else "No ablation results.",
        "", "## Robustness and sensitivity", "",
        markdown_table(
            sensitivity.groupby(["depth_p", "shots", "cardinality",
                                 "uniform_probability_noise_proxy", "transaction_cost_bps"]).agg(
                optimality_gap=("optimality_gap", "mean"),
                feasibility_rate=("feasibility_rate", "mean"),
                runtime_seconds=("runtime_seconds", "mean"),
            ).reset_index()
        ) if not sensitivity.empty else "No sensitivity results.",
        "", "## Statistical comparison", "",
        markdown_table(statistics),
        "", "## Market-regime description", "",
        markdown_table(regimes),
        "", "## H1–H6 interpretation", "",
    ]
    for h, status, reason in hypotheses:
        lines.append(f"- **{h}: {status}.** {reason}")
    if is_research:
        limitations = [
            "- Research execution requires a verified historical universe; current-listing or first-price proxies are blocked before this report can be produced.",
            "- Optional point-in-time fundamentals, macroeconomic data, foreign flow and corporate actions are excluded when verified real tables are unavailable.",
            "- The XY-QAOA implementation is an ideal fixed-Hamming-weight statevector simulator, not quantum hardware.",
            "- Penalty-QAOA and XY-QAOA are ideal internal statevector simulations, not quantum hardware.",
            "- Statistical results are conditional on the selected period, universe, costs and model specification; they are not investment advice or proof of quantum advantage.",
        ]
        reproduce = "python -m src.cli run-experiment --config configs/hose300_real.yaml"
    else:
        limitations = [
            "- Data are deterministic fixtures, explicitly not real HOSE observations.",
            "- The XY-QAOA implementation is an ideal fixed-Hamming-weight statevector simulator, not quantum hardware.",
            "- Penalty-QAOA and XY-QAOA are ideal internal statevector simulations, not quantum hardware.",
            "- Robustness, regimes and statistical tests are implemented, but fixture results are not confirmatory evidence.",
        ]
        reproduce = "python -m src.cli run-experiment --config configs/quick.yaml"
    lines += ["", "## Limitations", ""] + limitations + [
        "", "## Reproduce", "", "```powershell", reproduce, "```", "",
    ]
    md = "\n".join(lines)
    (out / "RESEARCH_REPORT.md").write_text(md, encoding="utf-8")
    html = "<html><meta charset='utf-8'><body>" + (
        md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n")
    ) + "</body></html>"
    (out / "report.html").write_text(html, encoding="utf-8")
