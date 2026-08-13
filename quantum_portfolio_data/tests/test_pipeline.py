from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    Paths, build_complete_case_workspace, build_universe, generate_fixture, leakage_audit,
    validate_data,
)
from src.research import (
    aligned_previous_weights, attach_point_in_time_features, build_features, drift_weights,
    energy, ewma_mean_cov, exact_solver, feasible_states, optimize_weights,
    penalty_qaoa_statevector, portfolio_turnover, qubo_instance, run_experiment,
    xy_qaoa_statevector,
)


def test_fixture_quality_and_universe(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2022-01-01", "2023-12-31", ["AAA", "BBB", "CCC"], 7)
    report, coverage = validate_data(paths)
    assert report["status"] == "pass"
    assert report["data_class"] == ["fixture"]
    universe = build_universe(paths)
    assert not universe.empty
    assert (universe["decision_time"] >= pd.Timestamp("2022-01-01")).all()


def test_complete_case_workspace_is_isolated_and_excludes_short_histories(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2022-01-01", "2022-12-31", ["AAA", "BBB"], 7)
    source_prices = paths.normalized / "prices.parquet"
    prices = pd.read_parquet(source_prices)
    prices["data_class"] = "real"
    prices["source"] = "test_real_source"
    prices["source_url"] = "https://example.test/prices"
    prices["adjustment_policy"] = "unverified"
    prices = prices[~(prices["ticker"].eq("BBB") & (prices.groupby("ticker").cumcount() >= 5))]
    prices.to_parquet(source_prices, index=False)
    master_path = paths.normalized / "security_master.parquet"
    master = pd.read_parquet(master_path)
    master["data_class"] = "real"
    master["source"] = "test_real_source"
    master["source_url"] = "https://example.test/master"
    master.to_parquet(master_path, index=False)

    before = source_prices.read_bytes()
    workspace, manifest = build_complete_case_workspace(
        paths, "2022-01-01", "2022-12-31", minimum_total_observations=40,
    )
    retained = pd.read_parquet(Paths(workspace).normalized / "prices.parquet")
    restricted_master = pd.read_parquet(Paths(workspace).normalized / "security_master.parquet")
    exclusions = pd.read_csv(Paths(workspace).reports / "complete_case_exclusions.csv")

    assert set(retained["ticker"]) == {"AAA"}
    assert set(restricted_master["ticker"]) == {"AAA"}
    assert exclusions.set_index("ticker").loc["BBB", "reason"] == (
        "fewer_than_minimum_complete_observations"
    )
    assert manifest["tickers_retained"] == 1
    assert source_prices.read_bytes() == before


def test_dynamic_universe_uses_only_trailing_liquidity(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2022-01-01", "2022-12-31", ["AAA", "BBB"], 7)
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    prices["trading_value"] = 1.0
    prices.loc[prices.ticker.eq("AAA"), "trading_value"] = 1_000_000.0
    prices.loc[
        prices.ticker.eq("BBB") & (pd.to_datetime(prices.date) >= pd.Timestamp("2022-08-01")),
        "trading_value",
    ] = 1_000_000_000.0
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    universe = build_universe(
        paths, max_assets=1, liquidity_lookback_days=20, minimum_observations=5
    )
    july = universe[universe.decision_time.dt.month.eq(7)]
    september = universe[universe.decision_time.dt.month.eq(9)]
    assert set(july.ticker) == {"AAA"}
    assert set(september.ticker) == {"BBB"}
    audit = pd.read_parquet(paths.curated / "universe_eligibility_audit.parquet")
    assert {"trailing_observations", "trailing_liquidity", "reason"} <= set(audit.columns)


def test_leakage_audit_fixture_contracts_are_complete_but_waived(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2022-01-01", "2022-12-31", ["AAA", "BBB"], 7)
    audit = leakage_audit(paths)
    assert audit["blockers"] == []
    assert audit["status"] == "pass_for_fixture_demo"


def test_qubo_exact_matches_enumeration():
    mu = np.array([0.03, 0.02, 0.01])
    cov = np.diag([0.02, 0.01, 0.03])
    q = qubo_instance(mu, cov, 0.5)
    result = exact_solver(q, 2)
    states = feasible_states(3, 2)
    assert result["energy"] == min(energy(s, q) for s in states)
    assert result["bits"].sum() == 2


def test_xy_qaoa_dicke_preserves_cardinality():
    q = np.diag([0.1, -0.2, 0.05, -0.1])
    result = xy_qaoa_statevector(q, k=2, p=1, trials=5, shots=64, seed=3)
    assert result["bits"].sum() == 2
    assert result["feasibility_rate"] == 1.0
    assert all(bitstring.count("1") == 2 for bitstring in result["bitstring_counts"])


def test_penalty_qaoa_is_full_space_circuit_simulation():
    q = np.diag([0.1, -0.2, 0.05, -0.1])
    result = penalty_qaoa_statevector(q, k=2, p=1, trials=5, shots=128, seed=3)
    assert result["method"] == "penalty_qaoa_ideal_statevector"
    assert result["backend"] == "internal_ideal_statevector_full_hilbert"
    assert 0 <= result["feasibility_rate"] <= 1
    assert result["bits"].shape == (4,)


def test_weights_constraints():
    mu = np.array([0.02, 0.01, 0.015])
    cov = np.eye(3) * 0.01
    w = optimize_weights(mu, cov, 0.0, 0.6, 1.0, None, 0.001)
    assert np.isclose(w.sum(), 1)
    assert (w >= -1e-9).all()
    assert (w <= 0.6 + 1e-9).all()


def test_multivariate_ewma_mean_and_covariance():
    returns = pd.DataFrame({
        "AAA": [0.01, -0.01, 0.02, 0.03],
        "BBB": [0.00, 0.01, -0.02, 0.01],
    })
    mu, cov = ewma_mean_cov(returns, span=3, horizon=20)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-9
    assert mu[0] > returns["AAA"].mean() * 20


def test_turnover_includes_purchases_and_sales():
    previous = {"AAA": 0.6, "BBB": 0.4}
    target = {"AAA": 0.3, "CCC": 0.7}
    turnover, trades = portfolio_turnover(previous, target)
    assert np.isclose(turnover, 1.4)
    assert trades == {"AAA": -0.3, "BBB": -0.4, "CCC": 0.7}
    assert np.allclose(aligned_previous_weights(["AAA", "CCC"], previous), [0.6, 0.0])


def test_weights_drift_between_rebalances():
    target = {"AAA": 0.5, "BBB": 0.5}
    realized = pd.DataFrame({"AAA": [0.10], "BBB": [0.0]})
    drifted = drift_weights(target, realized)
    assert np.isclose(sum(drifted.values()), 1.0)
    assert drifted["AAA"] > 0.5


def test_financial_features_join_on_publication_availability(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2022-01-01", "2023-12-31", ["AAA", "BBB"], 7)
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    features = attach_point_in_time_features(build_features(prices), paths)
    statements = pd.read_parquet(paths.normalized / "financial_statements.parquet")
    first_available = pd.to_datetime(statements["available_at"]).min()
    assert features.loc[features["date"] < first_available, "roe_pit"].isna().all()
    assert features.loc[features["date"] >= first_available, "roe_pit"].notna().any()


def test_research_mode_refuses_fixture_data(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2020-01-01", "2023-12-31", ["AAA", "BBB", "CCC"], 7)
    config = tmp_path / "full.yaml"
    config.write_text(
        "mode: research\n"
        "data: {source: configured}\n"
        "reduction: {candidate_size: 3}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refuses fixture"):
        run_experiment(tmp_path, config)
