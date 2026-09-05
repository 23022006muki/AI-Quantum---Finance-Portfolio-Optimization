from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_constraint_strategy_search import StrategyConfig, optimize_weights, project_bounded_simplex


def test_projection_preserves_sum_and_bounds():
    projected = project_bounded_simplex(np.array([0.95, 0.03, 0.01, 0.01]), 0.05, 0.30)
    assert projected.sum() == pytest.approx(1.0, abs=1e-10)
    assert projected.min() >= 0.05 - 1e-10
    assert projected.max() <= 0.30 + 1e-10


def test_inverse_volatility_weights_respect_declared_constraints():
    config = StrategyConfig(
        "test", "test", 8, 4, 0.30, 0.05, "inverse_volatility"
    )
    covariance = np.diag([1e-5, 2e-4, 3e-4, 4e-4])
    weights = optimize_weights(np.ones(4), covariance, np.zeros(4), config)
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert weights.min() >= config.weight_lower - 1e-10
    assert weights.max() <= config.weight_upper + 1e-10


def test_projection_rejects_infeasible_bounds():
    with pytest.raises(ValueError, match="Infeasible weight bounds"):
        project_bounded_simplex(np.ones(4), 0.05, 0.20)
