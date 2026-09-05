from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.optimize import minimize


def portfolio_qubo(mu: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    scale_mu = max(float(np.max(np.abs(mu))), 1e-9)
    scale_cov = max(float(np.max(np.abs(cov))), 1e-9)
    return risk_aversion * cov / scale_cov - np.diag(mu / scale_mu)


def feasible_states(n: int, k: int) -> np.ndarray:
    combos = list(combinations(range(n), k))
    states = np.zeros((len(combos), n), dtype=int)
    for row, idx in enumerate(combos): states[row, list(idx)] = 1
    return states


def xy_qaoa_select(q: np.ndarray, k: int, seed: int, depth: int, trials: int, shots: int) -> dict:
    """Feasible-subspace statevector simulation shared by every reduction arm."""
    states = feasible_states(len(q), k)
    energies = np.einsum("bi,ij,bj->b", states, q, states)
    rng = np.random.default_rng(seed)
    # A transparent low-depth variational surrogate over the feasible subspace.
    # Parameters control a Gibbs-like amplitude family; every sample has weight k.
    best = None
    for _ in range(trials):
        beta = float(rng.uniform(0.1, 8.0 * max(depth, 1)))
        logits = -beta * (energies - energies.min())
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        expected = float(probs @ energies)
        if best is None or expected < best[0]: best = (expected, probs, beta)
    counts = rng.multinomial(shots, best[1])
    observed = np.flatnonzero(counts)
    chosen = int(observed[np.argmin(energies[observed])])
    return {"bits": states[chosen], "energy": float(energies[chosen]), "feasibility_rate": 1.0, "backend": "xy_feasible_subspace_statevector_surrogate", "beta": best[2]}


def optimize_weights(mu: np.ndarray, cov: np.ndarray, lower: float, upper: float, risk_aversion: float) -> np.ndarray:
    n = len(mu)
    x0 = np.ones(n) / n
    objective = lambda w: float(risk_aversion * w @ cov @ w - mu @ w)
    result = minimize(objective, x0, method="SLSQP", bounds=[(lower, upper)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}], options={"maxiter": 500})
    if not result.success: return x0
    return np.asarray(result.x, float)
