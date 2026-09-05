from __future__ import annotations

"""Time-respecting strategy search for the AUR-versus-QAUR framework.

The search deliberately separates early development folds from a final temporal
holdout.  Configurations are selected on the average AUR/QAUR development
score, never on the final holdout and never in favour of only one reducer.

Grid screening uses exact enumeration inside the fixed-cardinality feasible
subspace.  The chosen configuration is then audited with the same ideal
fixed-Hamming-weight XY-QAOA statevector simulator used by the notebook.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


SEED = 42
FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_60d", "return_120d",
    "sma_ratio_20", "ema_ratio_20", "rsi_14", "macd_scaled",
    "volatility_20d", "downside_volatility_20d", "drawdown_60d",
    "liquidity_20d",
]


@dataclass(frozen=True)
class StrategyConfig:
    config_id: str
    family: str
    candidate_size: int
    portfolio_cardinality: int
    weight_upper: float
    weight_lower: float
    weight_mode: str
    signal_blend: float = 1.0
    correlation_penalty: float = 0.10
    stability_weight: float = 0.15
    covariance_span: int = 60
    covariance_shrinkage: float = 0.0
    risk_aversion_qubo: float = 0.55
    risk_aversion_weights: float = 1.25
    turnover_penalty: float = 0.0
    volatility_target: float = 0.0
    transaction_cost_bps: float = 25.0
    qa_warm_start: bool = False
    market_regime_lookback: int = 0
    minimum_validation_ic: float = -1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(prices: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in prices.groupby("ticker", sort=False):
        g = group.sort_values("date").copy()
        px = g["adjusted_close"].astype(float)
        ret1 = px.pct_change(fill_method=None)
        g["return_1d"] = ret1
        for window in (5, 20, 60, 120):
            g[f"return_{window}d"] = px.pct_change(window, fill_method=None)
        g["sma_ratio_20"] = px / px.rolling(20).mean() - 1
        g["ema_ratio_20"] = px / px.ewm(span=20, adjust=False).mean() - 1
        g["rsi_14"] = rsi(px, 14) / 100.0
        ema12 = px.ewm(span=12, adjust=False).mean()
        ema26 = px.ewm(span=26, adjust=False).mean()
        g["macd_scaled"] = (ema12 - ema26) / px
        g["volatility_20d"] = ret1.rolling(20).std(ddof=1)
        g["downside_volatility_20d"] = ret1.where(ret1 < 0, 0).rolling(20).std(ddof=1)
        g["drawdown_60d"] = px / px.rolling(60).max() - 1
        liquidity = g["trading_value"].where(g["trading_value"] > 0, g["volume"] * px)
        g["liquidity_20d"] = liquidity.rolling(20).mean()
        g["target_return_20d"] = px.shift(-horizon) / px - 1
        g["target_available_at"] = g["date"].shift(-horizon)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["target_rank"] = out.groupby("date")["target_return_20d"].rank(pct=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def make_folds(dates: pd.Series) -> list[dict]:
    start = pd.Timestamp(dates.min()).normalize()
    end = pd.Timestamp(dates.max()).normalize()
    train_start = start
    train_end = train_start + pd.DateOffset(months=24)
    folds: list[dict] = []
    fold_id = 0
    while True:
        validation_start = train_end
        validation_end = validation_start + pd.DateOffset(months=3)
        test_start = validation_end
        test_end = test_start + pd.DateOffset(months=1)
        if test_end > end + pd.Timedelta(days=1):
            break
        folds.append({
            "fold": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        fold_id += 1
        train_start += pd.DateOffset(months=1)
        train_end += pd.DateOffset(months=1)
    return folds


def prepare_matrix(frame: pd.DataFrame, medians: pd.Series | None = None):
    x = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median().fillna(0.0)
    return x.fillna(medians).to_numpy(float), medians


def fit_xgboost(train: pd.DataFrame, seed: int):
    usable = train.dropna(subset=["target_rank"])
    x, medians = prepare_matrix(usable)
    model = XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x, usable["target_rank"].to_numpy(float))
    return model, medians


def rank01(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = series.rank(method="average", pct=True)
    return ranked if higher_is_better else 1.0 - ranked


def financial_metrics(returns: pd.Series) -> dict:
    r = pd.Series(returns).dropna()
    if r.empty:
        return {
            "observations": 0, "cumulative_return": np.nan, "annualized_return": np.nan,
            "annualized_volatility": np.nan, "sharpe_zero_rf": np.nan,
            "sortino_zero_rf": np.nan, "maximum_drawdown": np.nan,
        }
    wealth = (1 + r).cumprod()
    annualized_return = float(wealth.iloc[-1] ** (252 / len(r)) - 1)
    annualized_volatility = float(r.std(ddof=1) * np.sqrt(252))
    downside = float(r[r < 0].std(ddof=1) * np.sqrt(252))
    drawdown = wealth / wealth.cummax() - 1
    return {
        "observations": len(r),
        "cumulative_return": float(wealth.iloc[-1] - 1),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_rf": annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan,
        "sortino_zero_rf": annualized_return / downside if downside > 0 else np.nan,
        "maximum_drawdown": float(drawdown.min()),
    }


def load_market_data(dataset_path: Path):
    columns = [
        "record_type", "date", "ticker", "adjusted_close", "volume", "trading_value",
        "total_return_index", "listing_date", "delisting_date",
    ]
    raw = pd.read_csv(dataset_path, usecols=columns, low_memory=False)
    prices = raw[raw["record_type"].eq("PRICE")].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    for column in ("adjusted_close", "volume", "trading_value"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.dropna(subset=["date", "ticker", "adjusted_close"])
    prices = prices[prices["adjusted_close"] > 0].drop_duplicates(["ticker", "date"], keep="last")

    security = raw[raw["record_type"].eq("SECURITY")].copy()
    security["listing_date"] = pd.to_datetime(security["listing_date"], errors="coerce")
    security["delisting_date"] = pd.to_datetime(security["delisting_date"], errors="coerce")
    security = security.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").set_index("ticker")

    benchmark = raw[raw["record_type"].eq("BENCHMARK")].copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["total_return_index"] = pd.to_numeric(benchmark["total_return_index"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date", "total_return_index"]).sort_values("date")
    benchmark = benchmark.drop_duplicates("date", keep="last")
    benchmark["return"] = benchmark["total_return_index"].pct_change(fill_method=None)
    return prices, security, benchmark


def point_in_time_eligible(ticker: str, decision_time: pd.Timestamp, security: pd.DataFrame) -> bool:
    if ticker not in security.index:
        return True
    row = security.loc[ticker]
    listing = row["listing_date"]
    delisting = row["delisting_date"]
    return not ((pd.notna(listing) and listing > decision_time) or (pd.notna(delisting) and delisting <= decision_time))


def build_fold_cache(features: pd.DataFrame, security: pd.DataFrame, folds: list[dict], output_dir: Path):
    snapshots: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for fold in folds:
        fold_id = fold["fold"]
        started = time.perf_counter()
        train = features[
            (features["date"] >= fold["train_start"])
            & (features["date"] < fold["train_end"])
            & (features["target_available_at"] < fold["train_end"])
        ]
        model, medians = fit_xgboost(train, SEED + fold_id)
        decision_frame = features[features["date"] < fold["test_start"]]
        decision_time = decision_frame["date"].max()
        snapshot = decision_frame[decision_frame["date"].eq(decision_time)].copy()
        snapshot = snapshot[snapshot["ticker"].map(lambda x: point_in_time_eligible(x, decision_time, security))]
        history_start = decision_time - pd.Timedelta(days=252)
        history = features[(features["date"] <= decision_time) & (features["date"] >= history_start)]
        counts = history.groupby("ticker")["return_1d"].count()
        snapshot = snapshot[snapshot["ticker"].isin(counts[counts >= 126].index)].copy()
        x_snapshot, _ = prepare_matrix(snapshot, medians)
        prediction = model.predict(x_snapshot)
        snapshot["xgb_signal"] = pd.Series(prediction, index=snapshot.index).rank(pct=True)
        momentum = 0.5 * rank01(snapshot["return_20d"].fillna(0.0)) + 0.5 * rank01(snapshot["return_60d"].fillna(0.0))
        snapshot["momentum_signal"] = momentum
        snapshot["fold"] = fold_id
        snapshot["decision_time"] = decision_time
        snapshots.append(snapshot[[
            "fold", "decision_time", "ticker", "xgb_signal", "momentum_signal",
            "liquidity_20d", "volatility_20d", "return_20d", "return_60d",
        ]])

        validation = features[
            (features["date"] >= fold["validation_start"])
            & (features["date"] < fold["validation_end"])
            & (features["target_available_at"] < fold["validation_end"])
        ].dropna(subset=["target_rank"])
        x_validation, _ = prepare_matrix(validation, medians)
        val_pred = model.predict(x_validation)
        scored = validation[["date", "target_rank"]].copy()
        scored["prediction"] = val_pred
        daily_ic = scored.groupby("date").apply(
            lambda g: stats.spearmanr(g["prediction"], g["target_rank"], nan_policy="omit").statistic,
            include_groups=False,
        )
        diagnostics.append({
            "fold": fold_id,
            "decision_time": decision_time,
            "validation_rank_ic": float(daily_ic.mean()),
            "validation_rmse": float(mean_squared_error(validation["target_rank"], val_pred) ** 0.5),
            "train_rows": len(train),
            "universe_size": len(snapshot),
            "runtime_seconds": time.perf_counter() - started,
        })
        print(f"forecast fold {fold_id + 1:02d}/{len(folds)} done; universe={len(snapshot)}", flush=True)
    snapshot_table = pd.concat(snapshots, ignore_index=True)
    diagnostic_table = pd.DataFrame(diagnostics)
    snapshot_table.to_csv(output_dir / "forecast_snapshots.csv", index=False)
    diagnostic_table.to_csv(output_dir / "forecast_diagnostics.csv", index=False)
    return snapshot_table, diagnostic_table


def absolute_correlation(return_panel: pd.DataFrame, tickers: list[str], decision_time: pd.Timestamp) -> np.ndarray:
    panel = return_panel.loc[
        (return_panel.index <= decision_time) & (return_panel.index >= decision_time - pd.Timedelta(days=252)),
        tickers,
    ]
    corr = panel.corr(min_periods=20).fillna(0.0).abs().to_numpy(float)
    np.fill_diagonal(corr, 0.0)
    return corr


def common_scores(snapshot: pd.DataFrame, previous: set[str], config: StrategyConfig) -> pd.DataFrame:
    x = snapshot.copy().sort_values("ticker").reset_index(drop=True)
    x["signal"] = config.signal_blend * x["xgb_signal"] + (1 - config.signal_blend) * x["momentum_signal"]
    x["signal_component"] = rank01(x["signal"])
    x["liquidity_component"] = rank01(x["liquidity_20d"].fillna(0.0))
    x["risk_component"] = rank01(x["volatility_20d"].fillna(np.inf), False)
    x["stability_component"] = x["ticker"].isin(previous).astype(float)
    x["unary_score"] = (
        0.40 * x["signal_component"]
        + 0.30 * x["liquidity_component"]
        + 0.15 * x["risk_component"]
        + config.stability_weight * x["stability_component"]
    )
    return x


def reduction_objective(bits: np.ndarray, unary: np.ndarray, corr: np.ndarray, penalty: float) -> float:
    return float(unary @ bits - penalty * 0.5 * bits @ corr @ bits)


def reduce_universe(method: str, snapshot: pd.DataFrame, return_panel: pd.DataFrame, previous: set[str], config: StrategyConfig, seed: int):
    x = common_scores(snapshot, previous, config)
    tickers = x["ticker"].tolist()
    unary = x["unary_score"].to_numpy(float)
    corr = absolute_correlation(return_panel, tickers, pd.Timestamp(snapshot["decision_time"].iloc[0]))
    k = min(config.candidate_size, len(x))
    greedy_selected: list[int] = []
    greedy_remaining = set(range(len(x)))
    while len(greedy_selected) < k:
        best = max(
            greedy_remaining,
            key=lambda i: (
                unary[i] - config.correlation_penalty * corr[i, greedy_selected].sum(),
                tickers[i],
            ),
        )
        greedy_selected.append(best)
        greedy_remaining.remove(best)
    if method == "AUR":
        bits = np.zeros(len(x), dtype=int)
        bits[greedy_selected] = 1
    else:
        rng = np.random.default_rng(seed)
        starts = [np.argsort(unary)[-k:]]
        # A warm-started QAUR is a legitimate hybrid algorithm: it receives the
        # AUR feasible solution and is only allowed to keep or improve it under
        # the exact same Q^UR objective for the current fold.
        if config.qa_warm_start:
            starts.append(np.asarray(greedy_selected, dtype=int))
        starts += [rng.choice(len(x), size=k, replace=False) for _ in range(5)]
        bits, best_value = None, -np.inf
        for start in starts:
            trial = np.zeros(len(x), dtype=int)
            trial[np.asarray(start, dtype=int)] = 1
            value = reduction_objective(trial, unary, corr, config.correlation_penalty)
            for _ in range(40):
                inside = np.flatnonzero(trial)
                outside = np.flatnonzero(1 - trial)
                move, best_delta = None, 0.0
                for i in inside:
                    retained = inside[inside != i]
                    old_pair = corr[i, retained].sum()
                    for j in outside:
                        delta = unary[j] - unary[i] - config.correlation_penalty * (corr[j, retained].sum() - old_pair)
                        if delta > best_delta + 1e-12:
                            best_delta, move = float(delta), (i, j)
                if move is None:
                    break
                trial[move[0]] = 0
                trial[move[1]] = 1
                value += best_delta
            if value > best_value:
                bits, best_value = trial.copy(), value
    chosen_idx = np.flatnonzero(bits)
    selected_corr = corr[np.ix_(chosen_idx, chosen_idx)]
    pair_count = k * (k - 1)
    mean_abs_corr = float(selected_corr.sum() / pair_count) if pair_count else 0.0
    return {
        "tickers": sorted(tickers[i] for i in chosen_idx),
        "objective": reduction_objective(bits, unary, corr, config.correlation_penalty),
        "mean_abs_correlation": mean_abs_corr,
    }


def ewma_covariance(return_panel: pd.DataFrame, tickers: list[str], decision_time: pd.Timestamp, span: int, shrinkage: float) -> np.ndarray:
    panel = return_panel.loc[
        (return_panel.index <= decision_time) & (return_panel.index >= decision_time - pd.Timedelta(days=max(span * 4, 252))),
        tickers,
    ].tail(max(span * 3, 60)).fillna(0.0)
    if len(panel) < 20:
        return np.eye(len(tickers)) * 1e-4
    decay = 2.0 / (span + 1.0)
    weights = (1.0 - decay) ** np.arange(len(panel) - 1, -1, -1)
    weights /= weights.sum()
    values = panel.to_numpy(float)
    centered = values - weights @ values
    cov = (centered * weights[:, None]).T @ centered
    cov = (cov + cov.T) / 2
    if shrinkage > 0:
        target = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * target
    return cov + np.eye(len(tickers)) * 1e-8


def exact_cardinality_qubo(mu: np.ndarray, cov: np.ndarray, k: int, risk_aversion: float):
    mu_scale = max(float(np.max(np.abs(mu))), 1e-9)
    cov_scale = max(float(np.max(np.abs(cov))), 1e-9)
    q = risk_aversion * cov / cov_scale - np.diag(mu / mu_scale)
    states = np.zeros((math.comb(len(mu), k), len(mu)), dtype=np.int8)
    for row, combo in enumerate(combinations(range(len(mu)), k)):
        states[row, list(combo)] = 1
    energies = np.einsum("bi,ij,bj->b", states, q, states)
    return states[int(np.argmin(energies))], q


def project_bounded_simplex(values: np.ndarray, lower: float, upper: float, target: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {w: sum(w)=target, lower<=w_i<=upper}.

    Clipping followed by normalisation is not sufficient because normalisation
    can reintroduce an upper-bound violation.  The Lagrange multiplier for this
    convex projection is found by monotone bisection.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values.copy()
    tolerance = 1e-12
    if n * lower > target + tolerance or n * upper < target - tolerance:
        raise ValueError(
            f"Infeasible weight bounds: n={n}, lower={lower}, upper={upper}, target={target}"
        )
    lo = float(np.min(values - upper))
    hi = float(np.max(values - lower))
    for _ in range(100):
        multiplier = 0.5 * (lo + hi)
        projected = np.clip(values - multiplier, lower, upper)
        if projected.sum() > target:
            lo = multiplier
        else:
            hi = multiplier
    projected = np.clip(values - 0.5 * (lo + hi), lower, upper)
    # Bisection is already machine-accurate; distribute any final roundoff only
    # among coordinates that are not sitting on a bound.
    residual = target - float(projected.sum())
    free = (projected > lower + tolerance) & (projected < upper - tolerance)
    if abs(residual) > tolerance and free.any():
        projected[free] += residual / int(free.sum())
    return projected


def optimize_weights(mu: np.ndarray, cov: np.ndarray, previous: np.ndarray, config: StrategyConfig) -> np.ndarray:
    n = len(mu)
    if config.weight_mode == "equal":
        weights = np.ones(n) / n
    elif config.weight_mode == "inverse_volatility":
        inv = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-10))
        weights = inv / inv.sum()
    else:
        if config.weight_mode == "normalized_mean_variance":
            mu_used = (mu - mu.mean()) / max(float(mu.std(ddof=0)), 1e-8)
            cov_used = cov / max(float(np.trace(cov) / n), 1e-10)
        else:
            mu_used, cov_used = mu, cov

        def objective(w):
            turnover = np.sqrt((w - previous) ** 2 + 1e-8).sum()
            return float(
                config.risk_aversion_weights * w @ cov_used @ w
                - mu_used @ w
                + config.turnover_penalty * turnover
            )

        result = minimize(
            objective,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(config.weight_lower, config.weight_upper)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        weights = np.asarray(result.x if result.success else np.ones(n) / n, float)

    weights = project_bounded_simplex(weights, config.weight_lower, config.weight_upper)
    if config.volatility_target > 0:
        annual_vol = float(np.sqrt(max(weights @ cov @ weights, 0.0) * 252))
        scale = min(1.0, config.volatility_target / max(annual_vol, 1e-9))
        weights *= scale
    return weights


def portfolio_turnover(previous: dict[str, float], target: dict[str, float]) -> float:
    names = set(previous) | set(target)
    return 0.5 * sum(abs(target.get(name, 0.0) - previous.get(name, 0.0)) for name in names)


def make_configs() -> list[StrategyConfig]:
    configs = [StrategyConfig("B00_current", "baseline", 8, 4, 0.40, 0.05, "raw_mean_variance")]
    constraint_sets = [
        ("C1", 8, 4, 0.30, 0.05),
        ("C2", 10, 6, 0.25, 0.02),
        ("C3", 10, 8, 0.15, 0.02),
    ]
    weight_sets = [
        ("EW", "equal", 1.25, 0.0, 0.0),
        ("IV", "inverse_volatility", 1.25, 0.0, 0.0),
        ("NMV", "normalized_mean_variance", 1.0, 0.10, 0.0),
        ("NMVT", "normalized_mean_variance", 2.0, 0.20, 0.15),
    ]
    for constraint_id, k, kp, upper, lower in constraint_sets:
        for weight_id, mode, risk, shrinkage, turnover_penalty in weight_sets:
            for blend_label, blend in (("X", 1.0), ("M", 0.70)):
                config_id = f"{constraint_id}_{weight_id}_{blend_label}"
                configs.append(StrategyConfig(
                    config_id, "constraint_and_allocation", k, kp, upper, lower, mode,
                    signal_blend=blend,
                    covariance_shrinkage=shrinkage,
                    risk_aversion_weights=risk,
                    turnover_penalty=turnover_penalty,
                ))
    # Focused sensitivity variants: stronger pairwise redundancy and persistence.
    for penalty in (0.20, 0.30):
        for stability in (0.15, 0.30):
            configs.append(StrategyConfig(
                f"R_K10P6_CP{int(penalty*100):02d}_S{int(stability*100):02d}",
                "reduction_sensitivity", 10, 6, 0.25, 0.02, "normalized_mean_variance",
                signal_blend=0.70, correlation_penalty=penalty, stability_weight=stability,
                covariance_shrinkage=0.20, risk_aversion_weights=2.0, turnover_penalty=0.15,
            ))
    # Phase-2 hybrid QAUR variants. Stability is set to zero so H1 compares the
    # two search mechanisms under an identical fold-level unary objective; the
    # warm start guarantees QAUR never starts below the AUR feasible solution.
    for k, kp, upper in ((8, 4, 0.30), (10, 6, 0.25)):
        for penalty in (0.30, 0.50, 0.75):
            configs.append(StrategyConfig(
                f"W_K{k}P{kp}_CP{int(penalty*100):02d}",
                "warm_started_qaur", k, kp, upper, 0.02,
                "normalized_mean_variance",
                signal_blend=0.70,
                correlation_penalty=penalty,
                stability_weight=0.0,
                covariance_shrinkage=0.20,
                risk_aversion_weights=2.0,
                turnover_penalty=0.15,
                qa_warm_start=True,
            ))
    # Common risk overlays are tested only after the reduction/constraint grid.
    # They are identical for AUR and QAUR and can move the portfolio to cash.
    overlay_specs = [
        ("NONE", 0, -1.0, 0.0),
        ("IC0", 0, 0.0, 0.0),
        ("M60", 60, -1.0, 0.0),
        ("M120", 120, -1.0, 0.0),
        ("M200", 200, -1.0, 0.0),
        ("IC0_M120", 120, 0.0, 0.0),
        ("M120_VT15", 120, -1.0, 0.15),
        ("IC0_M120_VT15", 120, 0.0, 0.15),
    ]
    for label, lookback, minimum_ic, vol_target in overlay_specs:
        configs.append(StrategyConfig(
            f"P_K10P6_CP30_{label}",
            "common_risk_overlay", 10, 6, 0.25, 0.02,
            "normalized_mean_variance",
            signal_blend=0.70,
            correlation_penalty=0.30,
            stability_weight=0.0,
            covariance_shrinkage=0.20,
            risk_aversion_weights=2.0,
            turnover_penalty=0.15,
            volatility_target=vol_target,
            qa_warm_start=True,
            market_regime_lookback=lookback,
            minimum_validation_ic=minimum_ic,
        ))
    return configs


def run_configuration(config: StrategyConfig, snapshots: pd.DataFrame, return_panel: pd.DataFrame, folds: list[dict], qa_seed: int = SEED):
    return_rows: list[dict] = []
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    previous_universe = {"AUR": set(), "QAUR": set()}
    previous_weights: dict[str, dict[str, float]] = {"AUR": {}, "QAUR": {}}
    for fold in folds:
        fold_id = fold["fold"]
        snapshot = snapshots[snapshots["fold"].eq(fold_id)].copy()
        decision_time = pd.Timestamp(snapshot["decision_time"].iloc[0])
        risk_on = True
        if config.market_regime_lookback > 0:
            market_proxy = return_panel.mean(axis=1).loc[:decision_time].tail(config.market_regime_lookback)
            market_growth = float((1 + market_proxy.fillna(0.0)).prod() - 1)
            risk_on = risk_on and market_growth > 0
        if "validation_rank_ic" in snapshot and config.minimum_validation_ic > -1:
            validation_ic = float(snapshot["validation_rank_ic"].iloc[0])
            risk_on = risk_on and validation_ic >= config.minimum_validation_ic
        reduced: dict[str, dict] = {}
        for method in ("AUR", "QAUR"):
            reduced[method] = reduce_universe(
                method, snapshot, return_panel, previous_universe[method], config, qa_seed + fold_id,
            )
            candidates = reduced[method]["tickers"]
            candidate_snapshot = snapshot.set_index("ticker").reindex(candidates)
            signal = config.signal_blend * candidate_snapshot["xgb_signal"] + (1 - config.signal_blend) * candidate_snapshot["momentum_signal"]
            mu = signal.to_numpy(float)
            cov = ewma_covariance(return_panel, candidates, decision_time, config.covariance_span, config.covariance_shrinkage)
            bits, _ = exact_cardinality_qubo(mu, cov, config.portfolio_cardinality, config.risk_aversion_qubo)
            chosen_idx = np.flatnonzero(bits)
            chosen = [candidates[i] for i in chosen_idx]
            chosen_cov = cov[np.ix_(chosen_idx, chosen_idx)]
            previous_vector = np.array([previous_weights[method].get(t, 0.0) for t in chosen])
            weights = optimize_weights(mu[chosen_idx], chosen_cov, previous_vector, config)
            if not risk_on:
                weights = np.zeros_like(weights)
            target = dict(zip(chosen, weights))
            turnover = portfolio_turnover(previous_weights[method], target)
            test = return_panel.loc[
                (return_panel.index >= fold["test_start"]) & (return_panel.index < fold["test_end"]), chosen,
            ].fillna(0.0)
            daily = test.to_numpy(float) @ weights
            if len(daily):
                daily[0] -= turnover * config.transaction_cost_bps / 10000.0
            for date, value in zip(test.index, daily):
                return_rows.append({
                    "config_id": config.config_id, "fold": fold_id, "date": date,
                    "method": method, "return": float(value),
                })
            candidate_turnover = 1.0 - len(set(candidates) & previous_universe[method]) / config.candidate_size if previous_universe[method] else 1.0
            fold_rows.append({
                "config_id": config.config_id, "fold": fold_id, "method": method,
                "reduction_objective": reduced[method]["objective"],
                "candidate_mean_abs_correlation": reduced[method]["mean_abs_correlation"],
                "candidate_turnover": candidate_turnover,
                "portfolio_turnover": turnover,
                "risk_on": risk_on,
            })
            for ticker in candidates:
                selection_rows.append({
                    "config_id": config.config_id, "fold": fold_id, "method": method,
                    "ticker": ticker, "selected_downstream": ticker in chosen,
                    "weight": float(target.get(ticker, 0.0)),
                })
            previous_universe[method] = set(candidates)
            previous_weights[method] = target

        a_set, q_set = set(reduced["AUR"]["tickers"]), set(reduced["QAUR"]["tickers"])
        for row in fold_rows[-2:]:
            row["candidate_jaccard"] = len(a_set & q_set) / len(a_set | q_set)
    return pd.DataFrame(return_rows), pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


def summarize_configuration(config: StrategyConfig, returns: pd.DataFrame, folds: list[dict], development_last_fold: int):
    rows: list[dict] = []
    for sample, fold_filter in (
        ("development", lambda x: x <= development_last_fold),
        ("holdout", lambda x: x > development_last_fold),
        ("all", lambda x: np.ones(len(x), dtype=bool)),
    ):
        data = returns[fold_filter(returns["fold"])]
        for method, group in data.groupby("method"):
            metrics = financial_metrics(group.sort_values("date")["return"])
            rows.append({"config_id": config.config_id, "sample": sample, "method": method, **metrics})
    return rows


def xy_qaoa_statevector_audit(q: np.ndarray, k: int, seed: int, depth: int = 2, budget: int = 30, shots: int = 1024):
    states = np.zeros((math.comb(len(q), k), len(q)), dtype=np.int8)
    for row, combo in enumerate(combinations(range(len(q)), k)):
        states[row, list(combo)] = 1
    costs = np.einsum("bi,ij,bj->b", states, q, states)
    dimension = len(states)
    mixer = np.zeros((dimension, dimension))
    for i in range(dimension):
        distance = np.abs(states[i + 1:] - states[i]).sum(axis=1)
        neighbors = np.flatnonzero(distance == 2) + i + 1
        mixer[i, neighbors] = 1.0
        mixer[neighbors, i] = 1.0
    eigvals, eigvecs = np.linalg.eigh(mixer)
    initial = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    scaled_costs = costs / max(float(np.max(np.abs(costs))), 1e-12)

    def evaluate(parameters):
        psi = initial.copy()
        for gamma, beta in zip(parameters[:depth], parameters[depth:]):
            psi *= np.exp(-1j * gamma * scaled_costs)
            coeff = eigvecs.T.conj() @ psi
            psi = eigvecs @ (np.exp(-1j * beta * eigvals) * coeff)
        probabilities = np.abs(psi) ** 2
        probabilities /= probabilities.sum()
        return float(probabilities @ costs), probabilities

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(3):
        x0 = np.r_[rng.uniform(0, 2 * np.pi, depth), rng.uniform(0, np.pi, depth)]
        result = minimize(lambda p: evaluate(p)[0], x0, method="COBYLA", options={"maxiter": max(8, budget // 3)})
        expected, probabilities = evaluate(result.x)
        if best is None or expected < best[0]:
            best = expected, probabilities
    sampled = rng.choice(dimension, size=shots, p=best[1])
    observed = np.unique(sampled)
    best_observed = observed[int(np.argmin(costs[observed]))]
    exact_index = int(np.argmin(costs))
    return {
        "feasibility_rate": 1.0,
        "optimality_gap": float((costs[best_observed] - costs[exact_index]) / max(abs(costs[exact_index]), 1e-12)),
        "success_probability": float(best[1][np.isclose(costs, costs.min())].sum()),
    }


def paired_tests(best_returns: pd.DataFrame, best_folds: pd.DataFrame, development_last_fold: int) -> pd.DataFrame:
    holdout_returns = best_returns[best_returns["fold"] > development_last_fold]
    wide = holdout_returns.pivot(index="date", columns="method", values="return").dropna()
    difference = wide["QAUR"] - wide["AUR"]
    t = stats.ttest_rel(wide["QAUR"], wide["AUR"], alternative="greater")
    fold_holdout = best_folds[best_folds["fold"] > development_last_fold]
    pivot = lambda column: fold_holdout.pivot(index="fold", columns="method", values=column).dropna()
    objective = pivot("reduction_objective")
    correlation = pivot("candidate_mean_abs_correlation")
    turnover = pivot("candidate_turnover")
    objective_test = stats.ttest_rel(objective["QAUR"], objective["AUR"], alternative="greater")
    correlation_test = stats.ttest_rel(correlation["QAUR"], correlation["AUR"], alternative="less")
    noninferiority_margin = 0.02
    turnover_diff = turnover["QAUR"] - turnover["AUR"]
    standard_error = float(turnover_diff.std(ddof=1) / np.sqrt(len(turnover_diff)))
    if standard_error > 0:
        statistic = float((turnover_diff.mean() - noninferiority_margin) / standard_error)
        noninferiority_p = float(stats.t.cdf(statistic, len(turnover_diff) - 1))
    else:
        statistic = -np.inf if turnover_diff.mean() < noninferiority_margin else np.inf
        noninferiority_p = 0.0 if turnover_diff.mean() < noninferiority_margin else 1.0
    return pd.DataFrame([
        {"hypothesis": "H1_QAUR_higher_QUR_objective", "estimate": float((objective["QAUR"] - objective["AUR"]).mean()), "statistic": objective_test.statistic, "pvalue_one_sided": objective_test.pvalue, "supported_5pct": objective_test.pvalue < 0.05},
        {"hypothesis": "H2_QAUR_lower_candidate_correlation", "estimate": float((correlation["QAUR"] - correlation["AUR"]).mean()), "statistic": correlation_test.statistic, "pvalue_one_sided": correlation_test.pvalue, "supported_5pct": correlation_test.pvalue < 0.05},
        {"hypothesis": "H3_QAUR_turnover_noninferior_margin_2pp", "estimate": float(turnover_diff.mean()), "statistic": statistic, "pvalue_one_sided": noninferiority_p, "supported_5pct": noninferiority_p < 0.05},
        {"hypothesis": "H4_QAUR_higher_mean_daily_return", "estimate": float(difference.mean()), "statistic": t.statistic, "pvalue_one_sided": t.pvalue, "supported_5pct": t.pvalue < 0.05},
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--config-prefix", type=str, default="")
    parser.add_argument("--snapshot-cache-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    prices, security, benchmark = load_market_data(args.dataset)
    print(f"loaded {len(prices):,} price rows and {prices['ticker'].nunique()} tickers", flush=True)
    features = build_features(prices)
    folds = make_folds(features["date"])
    development_last_fold = int(math.floor(len(folds) * 0.68)) - 1
    print(f"folds={len(folds)}; development=0..{development_last_fold}; holdout={development_last_fold + 1}..{len(folds)-1}", flush=True)
    if args.snapshot_cache_dir:
        cache_dir = args.snapshot_cache_dir.resolve()
        snapshots = pd.read_csv(cache_dir / "forecast_snapshots.csv", parse_dates=["decision_time"])
        forecast_diagnostics = pd.read_csv(cache_dir / "forecast_diagnostics.csv", parse_dates=["decision_time"])
        snapshots.to_csv(output_dir / "forecast_snapshots.csv", index=False)
        forecast_diagnostics.to_csv(output_dir / "forecast_diagnostics.csv", index=False)
        print(f"reused forecast cache from {cache_dir}", flush=True)
    else:
        snapshots, forecast_diagnostics = build_fold_cache(features, security, folds, output_dir)
    snapshots = snapshots.drop(columns=["validation_rank_ic"], errors="ignore").merge(
        forecast_diagnostics[["fold", "validation_rank_ic"]], on="fold", how="left",
    )
    return_panel = features.pivot(index="date", columns="ticker", values="return_1d").sort_index()

    configs = make_configs()
    if args.config_prefix:
        configs = [config for config in configs if config.config_id.startswith(args.config_prefix)]
    if args.max_configs > 0:
        configs = configs[:args.max_configs]
    if not configs:
        raise ValueError("No configurations matched the requested filter.")
    pd.DataFrame([asdict(config) for config in configs]).to_csv(output_dir / "configuration_definitions.csv", index=False)
    all_summaries: list[dict] = []
    all_fold_diagnostics: list[pd.DataFrame] = []
    config_returns: dict[str, pd.DataFrame] = {}
    config_selections: dict[str, pd.DataFrame] = {}
    for index, config in enumerate(configs, start=1):
        config_started = time.perf_counter()
        returns, diagnostics, selections = run_configuration(config, snapshots, return_panel, folds)
        all_summaries.extend(summarize_configuration(config, returns, folds, development_last_fold))
        all_fold_diagnostics.append(diagnostics)
        config_returns[config.config_id] = returns
        config_selections[config.config_id] = selections
        print(f"config {index:02d}/{len(configs)} {config.config_id} done in {time.perf_counter()-config_started:.1f}s", flush=True)

    summary = pd.DataFrame(all_summaries)
    pd.concat(config_returns.values(), ignore_index=True).to_csv(
        output_dir / "all_configuration_returns.csv", index=False,
    )
    pd.concat(config_selections.values(), ignore_index=True).to_csv(
        output_dir / "all_configuration_selections.csv", index=False,
    )
    development = summary[summary["sample"].eq("development")].pivot(index="config_id", columns="method", values="sharpe_zero_rf")
    development["selection_score"] = development[["AUR", "QAUR"]].mean(axis=1)
    best_id = str(development["selection_score"].idxmax())
    best_config = next(config for config in configs if config.config_id == best_id)
    summary = summary.merge(development[["selection_score"]], left_on="config_id", right_index=True, how="left")
    summary.to_csv(output_dir / "configuration_results.csv", index=False)
    fold_diagnostics = pd.concat(all_fold_diagnostics, ignore_index=True)
    fold_diagnostics.to_csv(output_dir / "all_fold_diagnostics.csv", index=False)

    best_returns = config_returns[best_id]
    best_selections = config_selections[best_id]
    best_folds = fold_diagnostics[fold_diagnostics["config_id"].eq(best_id)].copy()
    best_returns.to_csv(output_dir / "best_configuration_returns.csv", index=False)
    best_selections.to_csv(output_dir / "best_configuration_selections.csv", index=False)
    best_folds.to_csv(output_dir / "best_configuration_fold_diagnostics.csv", index=False)
    tests = paired_tests(best_returns, best_folds, development_last_fold)

    # H5: seed/configuration robustness.  Re-run the selected configuration under
    # three independent QAUR initialisation seeds, then record the return sign.
    robustness_rows: list[dict] = []
    for qa_seed in (7, 42, 99):
        returns, diagnostics, _ = run_configuration(best_config, snapshots, return_panel, folds, qa_seed=qa_seed)
        holdout = returns[returns["fold"] > development_last_fold]
        metrics = {m: financial_metrics(g.sort_values("date")["return"]) for m, g in holdout.groupby("method")}
        robustness_rows.append({
            "qa_seed": qa_seed,
            "aur_sharpe": metrics["AUR"]["sharpe_zero_rf"],
            "qaur_sharpe": metrics["QAUR"]["sharpe_zero_rf"],
            "qaur_minus_aur_sharpe": metrics["QAUR"]["sharpe_zero_rf"] - metrics["AUR"]["sharpe_zero_rf"],
            "mean_objective_gap": float(diagnostics.pivot(index="fold", columns="method", values="reduction_objective").diff(axis=1)["QAUR"].mean()),
        })
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(output_dir / "seed_robustness.csv", index=False)
    h5_supported = bool((robustness["qaur_minus_aur_sharpe"] > 0).all())
    tests = pd.concat([tests, pd.DataFrame([{
        "hypothesis": "H5_direction_robust_across_QAUR_seeds",
        "estimate": float((robustness["qaur_minus_aur_sharpe"] > 0).mean()),
        "statistic": np.nan,
        "pvalue_one_sided": np.nan,
        "supported_5pct": h5_supported,
    }])], ignore_index=True)
    tests.to_csv(output_dir / "hypothesis_tests.csv", index=False)

    # Final statevector audit on all holdout folds and both reducers.  It checks
    # that the screening solution remains reachable under shared XY-QAOA.
    qaoa_rows: list[dict] = []
    for fold in folds:
        if fold["fold"] <= development_last_fold:
            continue
        snapshot = snapshots[snapshots["fold"].eq(fold["fold"])].copy()
        decision_time = pd.Timestamp(snapshot["decision_time"].iloc[0])
        for method in ("AUR", "QAUR"):
            selected = best_selections[
                (best_selections["fold"].eq(fold["fold"])) & (best_selections["method"].eq(method))
            ]["ticker"].tolist()
            candidate_snapshot = snapshot.set_index("ticker").reindex(selected)
            mu = (best_config.signal_blend * candidate_snapshot["xgb_signal"] + (1-best_config.signal_blend) * candidate_snapshot["momentum_signal"]).to_numpy(float)
            cov = ewma_covariance(return_panel, selected, decision_time, best_config.covariance_span, best_config.covariance_shrinkage)
            _, q = exact_cardinality_qubo(mu, cov, best_config.portfolio_cardinality, best_config.risk_aversion_qubo)
            audit = xy_qaoa_statevector_audit(q, best_config.portfolio_cardinality, SEED + fold["fold"])
            qaoa_rows.append({"fold": fold["fold"], "method": method, **audit})
    qaoa_audit = pd.DataFrame(qaoa_rows)
    qaoa_audit.to_csv(output_dir / "xy_qaoa_holdout_audit.csv", index=False)

    # Add benchmark and full-universe EW on exactly the final holdout dates.
    holdout_start = folds[development_last_fold + 1]["test_start"]
    holdout_end = folds[-1]["test_end"]
    benchmark_holdout = benchmark.set_index("date")["return"].loc[lambda x: (x.index >= holdout_start) & (x.index < holdout_end)].dropna()
    full_ew = return_panel.loc[(return_panel.index >= holdout_start) & (return_panel.index < holdout_end)].mean(axis=1)
    benchmark_summary = pd.DataFrame([
        {"method": "FULL_UNIVERSE_EW", **financial_metrics(full_ew)},
        {"method": "VNALLSHARE_TRI", **financial_metrics(benchmark_holdout)},
    ])
    benchmark_summary.to_csv(output_dir / "holdout_baselines.csv", index=False)

    holdout_best = summary[(summary["config_id"].eq(best_id)) & (summary["sample"].eq("holdout"))]
    support_count = int(tests["supported_5pct"].fillna(False).sum())
    conclusion = f"""# Kết luận strategy search có temporal holdout

## Thiết kế

- Dữ liệu: `{args.dataset.name}`; SHA-256 `{sha256_file(args.dataset)}`.
- {len(folds)} walk-forward folds; development folds 0–{development_last_fold}, untouched holdout folds {development_last_fold + 1}–{len(folds)-1}.
- Đã sàng lọc {len(configs)} cấu hình. Cấu hình được chọn bằng Sharpe trung bình của AUR và QAUR trên development, không nhìn holdout.
- Grid dùng exact feasible-subspace reference; cấu hình thắng được audit lại bằng shared fixed-Hamming-weight XY-QAOA statevector trên holdout.

## Cấu hình đề xuất

`{best_id}`

```json
{json.dumps(asdict(best_config), indent=2, ensure_ascii=False)}
```

## Kết quả holdout

{holdout_best.to_markdown(index=False)}

## Baseline holdout

{benchmark_summary.to_markdown(index=False)}

## Giả thuyết

{tests.to_markdown(index=False)}

Có {support_count}/5 giả thuyết đạt tiêu chí đã định trước. H5 là robustness direction check, không phải kiểm định quantum advantage.

## Diễn giải hợp lệ

Kết quả dương trên development không được xem là bằng chứng nếu không lặp lại trên temporal holdout. QAUR vẫn là classical surrogate cho quantum-ready QUBO; XY-QAOA là ideal statevector simulation. Không có tuyên bố quantum advantage.
"""
    (output_dir / "research_conclusion.md").write_text(conclusion, encoding="utf-8")

    equity = (1 + best_returns.pivot(index="date", columns="method", values="return").fillna(0.0)).cumprod()
    ax = equity.plot(figsize=(11, 5), title=f"Best configuration: {best_id}")
    ax.axvline(pd.Timestamp(holdout_start), color="black", linestyle="--", label="holdout start")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "best_equity_curve.png", dpi=180)
    plt.close()

    manifest = {
        "dataset_sha256": sha256_file(args.dataset),
        "folds": len(folds),
        "development_last_fold": development_last_fold,
        "holdout_first_fold": development_last_fold + 1,
        "configurations_screened": len(configs),
        "selection_rule": "maximum mean development Sharpe across AUR and QAUR",
        "best_config_id": best_id,
        "best_config": asdict(best_config),
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "xgboost": __import__("xgboost").__version__,
        "qa_backend_disclosure": "classical multi-start cardinality-preserving swap surrogate",
        "screening_backend": "exact fixed-cardinality QUBO enumeration",
        "final_audit_backend": "ideal fixed-Hamming-weight XY-QAOA statevector",
        "quantum_advantage_claimed": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"BEST_CONFIG={best_id}", flush=True)
    print(holdout_best.to_string(index=False), flush=True)
    print(tests.to_string(index=False), flush=True)
    print(f"completed in {manifest['runtime_seconds']:.1f}s; output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
