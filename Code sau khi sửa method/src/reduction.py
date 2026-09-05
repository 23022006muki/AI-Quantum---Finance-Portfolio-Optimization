from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReductionResult:
    method: str
    tickers: tuple[str, ...]
    objective: float
    diagnostics: dict


def _rank01(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = values.rank(method="average", pct=True)
    return ranked if higher_is_better else 1.0 - ranked


def common_scores(snapshot: pd.DataFrame, previous: set[str], weights: dict) -> pd.DataFrame:
    """Create the identical unary information set consumed by both reducers."""
    x = snapshot.copy().sort_values("ticker").reset_index(drop=True)
    x["signal_component"] = _rank01(x["signal"].fillna(x["signal"].median()))
    x["liquidity_component"] = _rank01(x["liquidity_20d"].fillna(0.0))
    x["risk_component"] = _rank01(x["volatility_20d"].fillna(np.inf), False)
    x["stability_component"] = x["ticker"].isin(previous).astype(float)
    x["unary_score"] = (
        weights["signal"] * x["signal_component"]
        + weights["liquidity"] * x["liquidity_component"]
        + weights["risk"] * x["risk_component"]
        + weights["stability"] * x["stability_component"]
    )
    return x


def _corr_matrix(tickers: list[str], history_returns: pd.DataFrame) -> np.ndarray:
    corr = history_returns.reindex(columns=tickers).corr(min_periods=20).fillna(0.0).abs()
    out = corr.to_numpy(float)
    np.fill_diagonal(out, 0.0)
    return out


def _objective(bits: np.ndarray, unary: np.ndarray, corr: np.ndarray, penalty: float) -> float:
    # Maximise retained quality and penalise redundant pairs.
    return float(unary @ bits - penalty * 0.5 * bits @ corr @ bits)


def adaptive_reduce(snapshot: pd.DataFrame, history_returns: pd.DataFrame, previous: set[str], k: int, cfg: dict) -> ReductionResult:
    """Sequential adaptive screening across the complete eligible universe."""
    x = common_scores(snapshot, previous, cfg)
    tickers = x["ticker"].tolist()
    corr = _corr_matrix(tickers, history_returns)
    selected: list[int] = []
    remaining = set(range(len(x)))
    while len(selected) < min(k, len(x)):
        best = max(
            remaining,
            key=lambda i: (
                x.loc[i, "unary_score"]
                - cfg["correlation_penalty"] * sum(corr[i, j] for j in selected),
                tickers[i],
            ),
        )
        selected.append(best)
        remaining.remove(best)
    bits = np.zeros(len(x), dtype=int)
    bits[selected] = 1
    names = tuple(sorted(tickers[i] for i in selected))
    return ReductionResult("AUR", names, _objective(bits, x["unary_score"].to_numpy(), corr, cfg["correlation_penalty"]), {"backend": "adaptive_greedy", "universe_size": len(x)})


def quantum_assisted_reduce(snapshot: pd.DataFrame, history_returns: pd.DataFrame, previous: set[str], k: int, cfg: dict, seed: int, restarts: int, swap_steps: int) -> ReductionResult:
    """Solve a full-universe cardinality QUBO with a classical surrogate backend.

    The QUBO is quantum-ready; the current backend is explicitly a validation
    surrogate and must not be represented as execution on quantum hardware.
    """
    x = common_scores(snapshot, previous, cfg)
    tickers = x["ticker"].tolist()
    unary = x["unary_score"].to_numpy(float)
    corr = _corr_matrix(tickers, history_returns)
    rng = np.random.default_rng(seed)
    starts = [np.argsort(unary)[-k:]]
    starts.extend(rng.choice(len(x), size=k, replace=False) for _ in range(max(0, restarts - 1)))
    best_bits = None
    best_value = -np.inf
    for start in starts:
        bits = np.zeros(len(x), dtype=int)
        bits[np.asarray(start, dtype=int)] = 1
        value = _objective(bits, unary, corr, cfg["correlation_penalty"])
        for _ in range(swap_steps):
            inside = np.flatnonzero(bits)
            outside = np.flatnonzero(1 - bits)
            candidate_best = value
            move = None
            for i in inside:
                for j in outside:
                    trial = bits.copy(); trial[i] = 0; trial[j] = 1
                    trial_value = _objective(trial, unary, corr, cfg["correlation_penalty"])
                    if trial_value > candidate_best + 1e-12:
                        candidate_best, move = trial_value, (i, j)
            if move is None:
                break
            bits[move[0]] = 0; bits[move[1]] = 1; value = candidate_best
        if value > best_value:
            best_bits, best_value = bits.copy(), value
    names = tuple(sorted(x.loc[best_bits.astype(bool), "ticker"].tolist()))
    return ReductionResult("QAUR", names, float(best_value), {"backend": "cardinality_preserving_classical_surrogate", "quantum_ready_qubo": True, "universe_size": len(x), "restarts": restarts})

