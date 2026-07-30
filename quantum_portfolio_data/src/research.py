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

from .data_pipeline import Paths, leakage_audit, sha256_file, validate_data


FEATURES = [
    "return_5d", "return_20d", "return_60d", "return_120d", "sma_ratio_20",
    "ema_ratio_20", "rsi_14", "macd", "atr_14", "volatility_20d",
    "downside_volatility_20d", "drawdown_60d", "liquidity_20d", "beta_60d",
    "roe_pit", "revenue_growth_yoy_pit", "policy_rate_pit",
]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = -d.clip(upper=0).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
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
        x["target_return_20d"] = p.shift(-20) / p - 1
        x["target_rank"] = np.nan
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    out["target_rank"] = out.groupby("date")["target_return_20d"].rank(pct=True)
    out["feature_available_at"] = pd.to_datetime(out["date"])
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
            left = group.drop(columns=["roe_pit", "revenue_growth_yoy_pit"]).sort_values("date")
            if right.empty:
                left["roe_pit"] = np.nan
                left["revenue_growth_yoy_pit"] = np.nan
            else:
                left = pd.merge_asof(
                    left, right, left_on="date", right_on="financial_available_at",
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
                features.sort_values("date"), policy,
                left_on="date", right_on="macro_available_at", direction="backward",
            ).drop(columns=["macro_available_at"])
            features["policy_rate_pit"] = features["policy_rate_new"]
            features = features.drop(columns=["policy_rate_new"])
    return features


def make_folds(dates: pd.Series, train_months: int, validation_months: int,
               test_months: int, max_folds: int | None) -> list[dict]:
    unique = pd.Series(pd.to_datetime(dates).sort_values().unique())
    first = unique.min() + pd.DateOffset(months=train_months + validation_months)
    last = unique.max() - pd.DateOffset(months=test_months)
    anchors = pd.date_range(first, last, freq="ME")
    if max_folds:
        anchors = anchors[-max_folds:]
    folds = []
    for i, test_start in enumerate(anchors):
        train_start = test_start - pd.DateOffset(months=train_months + validation_months)
        train_end = test_start - pd.DateOffset(months=validation_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        folds.append({
            "fold": i, "train_start": train_start, "train_end": train_end,
            "validation_end": test_start, "test_start": test_start, "test_end": test_end,
        })
    return folds


def fit_ranker(train: pd.DataFrame, validation: pd.DataFrame, cfg: dict):
    usable = train.dropna(subset=["target_rank"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(usable[FEATURES])
    scaler = StandardScaler().fit(imputer.transform(usable[FEATURES]))
    x_train = scaler.transform(imputer.transform(usable[FEATURES]))
    model = XGBRegressor(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"], objective="reg:squarederror",
        random_state=42, n_jobs=1,
    )
    model.fit(x_train, usable["target_rank"])
    return imputer, scaler, model


def predict(model_bundle, df: pd.DataFrame) -> np.ndarray:
    imputer, scaler, model = model_bundle
    return model.predict(scaler.transform(imputer.transform(df[FEATURES])))


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
    m = min(cfg["candidate_size"], cfg["qubit_budget"], len(snap))
    selected: list[str] = []
    returns = history.pivot(index="date", columns="ticker", values="ret1").tail(120)
    corr = returns.corr().fillna(0)
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
    return snap.sort_values(["selected_candidate", "base_score"], ascending=[False, False])


def qubo_instance(mu: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    return risk_aversion * cov - (1 - risk_aversion) * np.diag(mu)


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
            "feasibility_rate": 1.0, "runtime_seconds": time.perf_counter() - start}


def xy_qaoa_statevector(
    q: np.ndarray, k: int, p: int, trials: int, shots: int, seed: int,
    noise: float = 0.0,
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
    best = None
    for _ in range(trials):
        gammas = rng.uniform(0, 2 * np.pi, p)
        betas = rng.uniform(0, np.pi, p)
        psi = initial.copy()
        for gamma, beta in zip(gammas, betas):
            psi *= np.exp(-1j * gamma * costs)
            coefficients = mixer_eigenvectors.T.conj() @ psi
            psi = mixer_eigenvectors @ (
                np.exp(-1j * beta * mixer_eigenvalues) * coefficients
            )
        probs = np.abs(psi) ** 2
        expected = float(probs @ costs)
        if best is None or expected < best["expected"]:
            best = {"expected": expected, "probs": probs, "gammas": gammas, "betas": betas}
    sample_probs = best["probs"] / best["probs"].sum()
    if noise:
        sample_probs = (1 - noise) * sample_probs + noise * np.ones(dim) / dim
    counts_idx = rng.choice(dim, size=shots, p=sample_probs)
    counts = np.bincount(counts_idx, minlength=dim)
    observed = int(np.argmin(np.where(counts > 0, costs, np.inf)))
    bit_counts = {"".join(map(str, states[i])): int(c) for i, c in enumerate(counts) if c}
    return {
        "method": "xy_qaoa_dicke_ideal_statevector", "bits": states[observed],
        "energy": float(costs[observed]), "mean_energy": best["expected"],
        "feasibility_rate": 1.0, "runtime_seconds": time.perf_counter() - start,
        "shots": shots, "depth_p": p, "two_qubit_gate_estimate": p * len(q) * (len(q) - 1) // 2,
        "bitstring_counts": bit_counts, "backend": "internal_ideal_statevector_fixed_weight",
        "noise_proxy": noise,
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
            "runtime_seconds": time.perf_counter() - start, "shots": shots,
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
    best = None
    for _ in range(trials):
        gammas = rng.uniform(0, 2 * np.pi, p)
        betas = rng.uniform(0, np.pi, p)
        psi = initial.copy()
        for gamma, beta in zip(gammas, betas):
            psi *= np.exp(-1j * gamma * total_cost)
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
        expected = float(probs @ total_cost)
        if best is None or expected < best["expected"]:
            best = {"expected": expected, "probs": probs}
    sampled = rng.choice(dim, size=shots, p=best["probs"] / best["probs"].sum())
    counts = np.bincount(sampled, minlength=dim)
    feasible = states.sum(axis=1) == k
    observed_pool = np.flatnonzero((counts > 0) & feasible)
    if len(observed_pool):
        chosen_idx = int(observed_pool[np.argmin(economic[observed_pool])])
    else:
        chosen_idx = int(np.argmin(np.where(counts > 0, total_cost, np.inf)))
    bit_counts = {"".join(map(str, states[i])): int(c) for i, c in enumerate(counts) if c}
    sampled_feasible = sum(c for i, c in enumerate(counts) if feasible[i]) / shots
    return {
        "method": "penalty_qaoa_ideal_statevector", "bits": states[chosen_idx],
        "energy": float(economic[chosen_idx]), "penalized_mean_energy": best["expected"],
        "feasibility_rate": float(sampled_feasible),
        "runtime_seconds": time.perf_counter() - start, "shots": shots, "depth_p": p,
        "two_qubit_gate_estimate": p * (n * (n - 1) // 2 + n),
        "bitstring_counts": bit_counts, "backend": "internal_ideal_statevector_full_hilbert",
        "penalty_strength": penalty_strength,
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
    p_value = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "mean_difference": float(diff.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_value": float(min(1.0, p_value)),
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


def run_experiment(project_root: Path, config_path: Path) -> Path:
    cfg = load_config(config_path)
    paths = Paths(project_root)
    quality, _ = validate_data(paths)
    leak = leakage_audit(paths)
    if quality["status"] != "pass":
        raise RuntimeError("Data quality failed; refusing to run.")
    if leak["status"] == "blocked":
        raise RuntimeError("Leakage audit blocked; refusing to label results.")
    if cfg.get("mode") == "research" and "fixture" in quality["data_class"]:
        raise RuntimeError(
            "Research mode refuses fixture data. Supply verified real point-in-time data "
            "and pass the leakage audit before using configs/full.yaml."
        )
    if cfg["reduction"]["candidate_size"] > 8 and cfg.get("mode") != "research":
        raise RuntimeError("Internal exact statevector demo is limited to 8 candidate qubits.")
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["ret1"] = prices.groupby("ticker")["adjusted_close"].pct_change()
    features = attach_point_in_time_features(build_features(prices), paths)
    wf = cfg["walk_forward"]
    folds = make_folds(features["date"], wf["train_months"], wf["validation_months"],
                       wf["test_months"], wf.get("max_folds"))
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    cfg_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:10]
    experiment_id = f"{stamp}-{cfg_hash}"
    out = project_root / "outputs" / "experiments" / experiment_id
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out / "features.parquet", index=False)
    pd.DataFrame(folds).to_csv(out / "fold_manifest.csv", index=False)
    ranking_rows, selection_rows, instance_rows = [], [], []
    solver_rows, weight_rows, trade_rows, return_rows = [], [], [], []
    ablation_rows, sensitivity_rows = [], []
    previous_weights: dict[str, dict[str, float]] = {
        "full_pipeline_exact_selection": {},
    }
    for fold in folds:
        train = features[(features.date >= fold["train_start"]) & (features.date < fold["train_end"])].copy()
        val = features[(features.date >= fold["train_end"]) & (features.date < fold["validation_end"])].copy()
        test = features[(features.date > fold["test_start"]) & (features.date <= fold["test_end"])].copy()
        snapshot_date = features.loc[features.date <= fold["test_start"], "date"].max()
        market_features = [
            name for name in FEATURES
            if name not in {"roe_pit", "revenue_growth_yoy_pit", "policy_rate_pit"}
        ]
        snap = features[features.date == snapshot_date].dropna(subset=market_features).copy()
        if len(snap) < cfg["reduction"]["candidate_size"] or test.empty:
            continue
        bundle = fit_ranker(train, val, cfg["model"])
        snap["signal"] = predict(bundle, snap)
        known = snap.dropna(subset=["target_rank"])
        ic = spearmanr(known["signal"], known["target_rank"]).statistic if len(known) > 2 else np.nan
        for row in snap.itertuples():
            ranking_rows.append({"fold": fold["fold"], "decision_time": snapshot_date,
                                 "ticker": row.ticker, "signal": row.signal, "fold_rank_ic": ic})
        history = features[features.date <= snapshot_date].copy()
        history["ret1"] = history.sort_values(["ticker", "date"]).groupby("ticker")["adjusted_close"].pct_change()
        reduced = adaptive_reduce(snap, history, cfg["reduction"])
        reduced["fold"] = fold["fold"]
        reduced["decision_time"] = snapshot_date
        selection_rows.extend(reduced[["fold", "decision_time", "ticker", "base_score",
            "selected_candidate", "decision_reason"]].to_dict("records"))
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
        k = min(cfg["reduction"]["cardinality"], len(candidates))
        instance_rows.append({
            "fold": fold["fold"], "decision_time": str(snapshot_date),
            "tickers": candidates, "cardinality": k,
            "expected_return": mu.tolist(), "covariance": cov.tolist(), "qubo_matrix": q.tolist(),
            "covariance_method": covariance_method, "covariance_span": covariance_span,
            "holding_horizon_days": holding_horizon,
        })
        exact = exact_solver(q, k)
        exact_bits = exact["bits"].copy()
        runs = [
            exact,
            simulated_annealing(q, k, cfg["solver"]["seeds"][0]),
            penalty_qaoa_baseline(q, k, cfg["solver"]["seeds"][0], cfg["solver"]["shots"]),
            penalty_qaoa_statevector(
                q, k, cfg["solver"]["qaoa_depth"], cfg["solver"]["parameter_trials"],
                cfg["solver"]["shots"], cfg["solver"]["seeds"][0],
            ),
        ]
        for seed in cfg["solver"]["seeds"]:
            runs.append(xy_qaoa_statevector(q, k, cfg["solver"]["qaoa_depth"],
                        cfg["solver"]["parameter_trials"], cfg["solver"]["shots"], seed))
        for run in runs:
            run["fold"] = fold["fold"]
            run["decision_time"] = str(snapshot_date)
            run["optimality_gap"] = float((run["energy"] - exact["energy"]) / (abs(exact["energy"]) + 1e-12))
            run["selected_tickers"] = [candidates[i] for i in np.flatnonzero(run["bits"])]
            run["bits"] = "".join(map(str, run["bits"]))
            run["bitstring_counts"] = json.dumps(run.get("bitstring_counts", {}))
            solver_rows.append(run)
        chosen = [candidates[i] for i in np.flatnonzero(exact_bits)]
        idx = [candidates.index(c) for c in chosen]
        strategy_name = "full_pipeline_exact_selection"
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
                                "ticker": ticker, "weight": w, "pre_trade_weight": previous.get(ticker, 0.0)})
            trade_rows.append({"fold": fold["fold"], "trade_time": test.date.min(),
                               "strategy": strategy_name, "ticker": ticker,
                               "pre_trade_weight": previous.get(ticker, 0.0),
                               "target_weight": w, "trade_weight": trade_changes[ticker],
                               "turnover": abs(trade_changes[ticker]),
                               "transaction_cost": abs(trade_changes[ticker]) * cost})
        test_ret = test[test.ticker.isin(chosen)].pivot(index="date", columns="ticker",
                                                        values="ret1").reindex(columns=chosen).fillna(0)
        port = test_ret @ weights
        if len(port):
            port.iloc[0] -= transaction_cost
        previous_weights[strategy_name] = drift_weights(target, test_ret)
        equal = test_ret.mean(axis=1)
        for date, value in port.items():
            return_rows.append({"fold": fold["fold"], "date": date, "strategy": "full_pipeline_exact_selection",
                                "return": value})
        for date, value in equal.items():
            return_rows.append({"fold": fold["fold"], "date": date, "strategy": "equal_weight_selected",
                                "return": value})
        # Eight pre-declared ablations use their own candidate construction and solver.
        ablation_specs = [
            ("01_no_ai_no_quantum", "liquidity", "exact", False),
            ("02_ewma_classical", "ewma", "exact", False),
            ("03_xgboost_classical", "xgboost", "exact", False),
            ("04_adaptive_classical", "adaptive", "exact", False),
            ("05_xgboost_penalty_qaoa", "xgboost", "penalty", False),
            ("06_xgboost_xy_qaoa", "xgboost", "xy", False),
            ("07_adaptive_xy_qaoa", "adaptive", "xy", False),
            ("08_full_pipeline_costs", "adaptive", "xy", True),
        ]
        m = min(cfg["reduction"]["candidate_size"], len(snap))
        for ablation_name, selector, solver_name, apply_cost in ablation_specs:
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
            variant_returns = variant_test @ variant_weights
            variant_target = {
                ticker: float(weight) for ticker, weight in zip(selected, variant_weights)
            }
            variant_turnover, _ = portfolio_turnover(strategy_previous, variant_target)
            variant_cost = cost * variant_turnover if apply_cost and len(variant_returns) else 0
            if len(variant_returns):
                variant_returns.iloc[0] -= variant_cost
            previous_weights[ablation_name] = drift_weights(variant_target, variant_test)
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
            for date, value in variant_returns.items():
                return_rows.append({
                    "fold": fold["fold"], "date": date,
                    "strategy": ablation_name, "return": value,
                })
        if fold["fold"] == 0:
            for depth in sorted({1, cfg["solver"]["qaoa_depth"], 2}):
                for shots in sorted({256, cfg["solver"]["shots"]}):
                    for sensitivity_k in sorted({max(1, k - 1), k}):
                        for noise in (0.0, 0.02):
                            sensitivity = xy_qaoa_statevector(
                                q, sensitivity_k, depth,
                                max(6, cfg["solver"]["parameter_trials"] // 3),
                                shots, cfg["solver"]["seeds"][0], noise=noise,
                            )
                            exact_sensitivity = exact_solver(q, sensitivity_k)
                            for sensitivity_cost in (0, cfg["backtest"]["transaction_cost_bps"], 25):
                                sensitivity_rows.append({
                                    "fold": fold["fold"], "depth_p": depth, "shots": shots,
                                    "cardinality": sensitivity_k, "candidate_size": len(candidates),
                                    "qubit_budget": cfg["reduction"]["qubit_budget"],
                                    "noise": noise, "transaction_cost_bps": sensitivity_cost,
                                    "energy": sensitivity["energy"],
                                    "optimality_gap": (sensitivity["energy"] - exact_sensitivity["energy"]) /
                                                      (abs(exact_sensitivity["energy"]) + 1e-12),
                                    "feasibility_rate": sensitivity["feasibility_rate"],
                                    "runtime_seconds": sensitivity["runtime_seconds"],
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
    returns.to_csv(out / "portfolio_returns.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out / "ablation_results.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out / "sensitivity_results.csv", index=False)
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
    if "08_full_pipeline_costs" in returns_wide:
        for baseline in ["01_no_ai_no_quantum", "02_ewma_classical", "03_xgboost_classical",
                         "04_adaptive_classical"]:
            if baseline in returns_wide:
                result = paired_block_bootstrap_test(
                    returns_wide["08_full_pipeline_costs"], returns_wide[baseline], cfg["seed"]
                )
                result.update({"test": f"08_full_pipeline_costs_vs_{baseline}"})
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
    env = f"python={sys.version}\nplatform={platform.platform()}\n"
    (out / "environment.txt").write_text(env, encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id, "status": "success", "label": cfg["label"],
        "data_class": quality["data_class"], "started_from_config": str(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(), "config_hash": cfg_hash,
        "dataset_hash": sha256_file(paths.normalized / "prices.parquet"),
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
    current = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    run_report.extend(f"- `{name}`" for name in current)
    run_report_name = "RUN_REPORT.md" if cfg.get("mode") == "research" else "DEMO_RUN_REPORT.md"
    (out / run_report_name).write_text("\n".join(run_report) + "\n", encoding="utf-8")
    manifest["artifacts"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
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
    hypotheses = [
        ("H1", "descriptive", f"Mean walk-forward fold rank IC={ic:.4f}."),
        ("H2", "inconclusive", "Adaptive selection executed, but superiority requires independent replication."),
        ("H3", "implementation-supported", "Fixed-weight XY simulation feasibility is one by construction."),
        ("H4", "descriptive", "Optimization gaps are reported for the configured reduced instances."),
        ("H5", "descriptive", "Walk-forward performance includes configured costs; it is not investment advice."),
        ("H6", "descriptive", "Sensitivity is limited to the pre-declared grid in the resolved config."),
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
        f"- Tickers: {quality['tickers']}", "", "## Predictive ranking", "",
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
            sensitivity.groupby(["depth_p", "shots", "cardinality", "noise", "transaction_cost_bps"]).agg(
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
            "- The universe is selected from the current HOSE listing and therefore may contain survivorship bias for historical dates.",
            "- Optional point-in-time fundamentals, macroeconomic data, foreign flow and corporate actions are excluded when verified real tables are unavailable.",
            "- The XY-QAOA implementation is an ideal fixed-Hamming-weight statevector simulator, not quantum hardware.",
            "- The penalty comparator is a transparent stochastic penalty baseline, not a QAOA circuit.",
            "- Statistical results are conditional on the selected period, universe, costs and model specification; they are not investment advice or proof of quantum advantage.",
        ]
        reproduce = "python -m src.cli run-experiment --config configs/hose300_real.yaml"
    else:
        limitations = [
            "- Data are deterministic fixtures, explicitly not real HOSE observations.",
            "- The XY-QAOA implementation is an ideal fixed-Hamming-weight statevector simulator, not quantum hardware.",
            "- The penalty comparator is a transparent stochastic penalty baseline, not a QAOA circuit.",
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
