"""Validation tests embedded in the standalone Google Colab notebook."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.import_colab_complete_csv import bool_series, parse_dates, require_columns
from src.data_pipeline import sha256_file
from src.research import (
    build_features,
    energy,
    ewma_mean_cov,
    exact_solver,
    feasible_states,
    financial_metrics,
    holm_adjust,
    make_folds,
    optimize_weights,
    paired_block_bootstrap_test,
    penalty_qaoa_statevector,
    purged_fold_frames,
    simulated_annealing,
    transaction_cost_breakdown,
    xy_qaoa_statevector,
)


def _prices(tickers: int = 4, periods: int = 190) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    rows: list[dict] = []
    for j in range(tickers):
        close = 20 + j + np.cumsum(0.02 + 0.1 * np.sin(np.arange(periods) / 11 + j))
        for i, date in enumerate(dates):
            rows.append({
                "ticker": f"T{j}", "date": date, "open": close[i] * 0.998,
                "high": close[i] * 1.01, "low": close[i] * 0.99,
                "close": close[i], "adjusted_close": close[i],
                "volume": 1_000_000 + 1000 * i,
                "trading_value": close[i] * (1_000_000 + 1000 * i),
                "available_at": date + pd.Timedelta(hours=8),
            })
    return pd.DataFrame(rows)


def test_01_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"standalone-colab")
    assert sha256_file(path) == hashlib.sha256(b"standalone-colab").hexdigest()


def test_02_schema_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(pd.DataFrame({"ticker": ["AAA"]}), ["ticker", "date"], "PRICE")


def test_03_parse_dates_is_explicit() -> None:
    result = parse_dates(pd.DataFrame({"date": ["2025-01-02"]}), ["date"])
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_04_boolean_parser_is_conservative() -> None:
    parsed = bool_series(pd.Series(["true", "1", "yes", "false", "0", ""])).tolist()
    assert parsed == [True, True, True, False, False, False]


def test_05_price_fixture_has_valid_ohlc() -> None:
    frame = _prices()
    assert (frame.high >= frame[["open", "close"]].max(axis=1)).all()
    assert (frame.low <= frame[["open", "close"]].min(axis=1)).all()
    assert (frame.volume >= 0).all()


def test_06_feature_builder_is_point_in_time() -> None:
    features = build_features(_prices(), target_horizon_days=20)
    assert (features["feature_available_at"] >= features["date"]).all()
    assert "target_return_20d" in features
    assert "label_end_time" in features


def test_07_walk_forward_ordering() -> None:
    folds = make_folds(pd.Series(pd.bdate_range("2020-01-01", "2025-12-31")), 24, 3, 1, 4)
    assert len(folds) == 4
    assert all(f["train_start"] < f["train_end"] <= f["validation_start"] for f in folds)
    assert all(f["validation_start"] < f["validation_end"] <= f["test_start"] for f in folds)


def test_08_purge_prevents_label_overlap() -> None:
    features = build_features(_prices(periods=500), target_horizon_days=20)
    fold = make_folds(features.date, 12, 2, 1, 1, embargo_days=5)[0]
    train, validation, _, _ = purged_fold_frames(features, fold)
    assert train["label_end_time"].max() < fold["validation_start"] - pd.Timedelta(days=5)
    assert validation["label_end_time"].max() < fold["test_start"] - pd.Timedelta(days=5)


def test_09_feasible_states_preserve_cardinality() -> None:
    states = feasible_states(8, 4)
    assert states.shape == (70, 8)
    assert np.all(states.sum(axis=1) == 4)


def test_10_qubo_energy_matches_definition() -> None:
    q = np.array([[1.0, -0.25], [-0.25, 0.5]])
    bits = np.array([1, 1])
    assert energy(bits, q) == pytest.approx(float(bits @ q @ bits))


def test_11_exact_solver_is_reference_optimum() -> None:
    q = np.diag([0.4, -0.5, 0.1, -0.2])
    result = exact_solver(q, 2)
    all_energies = [energy(s, q) for s in feasible_states(4, 2)]
    assert result["energy"] == pytest.approx(min(all_energies))
    assert result["bits"].sum() == 2


def test_12_simulated_annealing_keeps_cardinality() -> None:
    result = simulated_annealing(np.eye(6), 3, seed=42, steps=50)
    assert result["bits"].sum() == 3
    assert result["feasibility_rate"] == 1.0


def test_13_xy_qaoa_preserves_feasible_subspace() -> None:
    q = np.diag(np.linspace(-0.4, 0.3, 6))
    result = xy_qaoa_statevector(q, 3, p=1, trials=8, shots=128, seed=11)
    assert result["bits"].sum() == 3
    assert result["feasibility_rate"] == 1.0
    assert result["backend"] == "internal_ideal_statevector_fixed_weight"


def test_14_xy_qaoa_is_seed_reproducible() -> None:
    q = np.diag(np.linspace(-0.4, 0.3, 5))
    first = xy_qaoa_statevector(q, 2, p=1, trials=8, shots=64, seed=23)
    second = xy_qaoa_statevector(q, 2, p=1, trials=8, shots=64, seed=23)
    assert np.array_equal(first["bits"], second["bits"])
    assert first["energy"] == pytest.approx(second["energy"])


def test_15_penalty_qaoa_reports_measured_feasibility() -> None:
    q = np.diag(np.linspace(-0.2, 0.2, 5))
    result = penalty_qaoa_statevector(q, 2, p=1, trials=8, shots=128, seed=47)
    assert 0.0 <= result["feasibility_rate"] <= 1.0
    assert result["backend"] == "internal_ideal_statevector_full_hilbert"


def test_16_ewma_covariance_is_symmetric_psd() -> None:
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(120, 4)), columns=list("ABCD"))
    mean, cov = ewma_mean_cov(returns, span=30, horizon=20)
    assert len(mean) == 4
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-9


def test_17_weight_optimizer_enforces_long_only_budget() -> None:
    mu = np.array([0.02, 0.01, 0.015, 0.005])
    cov = np.eye(4) * 0.02
    weights = optimize_weights(mu, cov, 0.05, 0.5, 1.0, None, 0.01)
    assert weights.sum() == pytest.approx(1.0, abs=1e-7)
    assert np.all(weights >= 0.05 - 1e-8)
    assert np.all(weights <= 0.5 + 1e-8)


def test_18_transaction_cost_reduces_net_return() -> None:
    total, details = transaction_cost_breakdown(
        {"AAA": 0.6, "BBB": -0.4}, commission_bps=10.0,
        sell_tax_bps=10.0, slippage_bps=5.0, impact_coefficient=0.0005,
        adv_capacity_weights={"AAA": 5.0, "BBB": 4.0},
    )
    assert total > 0
    assert details["BBB"]["sell_tax_cost"] > 0


def test_19_holm_adjustment_controls_familywise_error() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.20])
    assert adjusted[0] == pytest.approx(0.03)
    assert all(0 <= value <= 1 for value in adjusted)


def test_20_block_bootstrap_returns_valid_statistics() -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    a = pd.Series(np.linspace(-0.01, 0.02, 80), index=index)
    b = pd.Series(np.linspace(-0.012, 0.015, 80), index=index)
    result = paired_block_bootstrap_test(a, b, seed=42, samples=50, block=5)
    assert result["ci_low"] <= result["ci_high"]
    assert 0 <= result["p_value"] <= 1


def test_21_financial_metrics_include_drawdown() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.005] * 20)
    metrics = financial_metrics(returns, 0.03)
    assert metrics["observations"] == 80
    assert metrics["max_drawdown"] <= 0
