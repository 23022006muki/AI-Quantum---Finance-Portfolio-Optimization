from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    Paths, apply_price_adjustment_contract, build_universe, generate_fixture,
    leakage_audit, sha256_file, validate_data,
)
from src.research import (
    FEATURES,
    ResearchRunBlocked,
    adaptive_reduce,
    build_features,
    energy,
    fit_ranker,
    ising_energy,
    make_folds,
    paired_block_bootstrap_test,
    penalty_qaoa_statevector,
    purged_fold_frames,
    qubo_to_ising,
    record_rebalanced_strategy,
    prepare_realized_return_panel,
    run_experiment,
    simulate_buy_and_hold,
    transaction_cost_breakdown,
    xy_qaoa_statevector,
)
from src.sources import import_point_in_time_table


def test_validate_rejects_availability_before_observation(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2022-01-01", "2022-12-31", ["AAA", "BBB"])
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    prices.loc[0, "available_at"] = prices.loc[0, "date"] - pd.Timedelta(days=1)
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    report, _ = validate_data(paths)
    assert report["status"] == "fail"
    assert any(issue["check"] == "available_before_observation" for issue in report["issues"])


def test_real_proxy_history_is_a_research_blocker(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2022-01-01", "2022-12-31", ["AAA", "BBB"])
    master = pd.read_parquet(paths.normalized / "security_master.parquet")
    master["data_class"] = "real"
    master["source"] = "market_prices"
    master["history_method"] = "first_price_observation_proxy"
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    prices["data_class"] = "real"
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    build_universe(paths)
    audit = leakage_audit(paths)
    assert audit["status"] == "blocked"
    assert "historical_universe_source_trusted" in audit["blockers"]


def test_universe_respects_record_availability(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2022-01-01", "2022-12-31", ["AAA", "BBB"])
    master = pd.read_parquet(paths.normalized / "security_master.parquet")
    master.loc[master.ticker == "BBB", "available_at"] = pd.Timestamp("2022-07-01")
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)
    universe = build_universe(paths)
    early = universe[universe.decision_time < pd.Timestamp("2022-07-01")]
    assert "BBB" not in set(early.ticker)


def test_feature_target_stores_label_end_time():
    dates = pd.bdate_range("2022-01-01", periods=40)
    prices = pd.DataFrame({
        "date": dates, "ticker": "AAA", "open": 10.0, "high": 11.0, "low": 9.0,
        "close": np.arange(10, 50), "adjusted_close": np.arange(10, 50),
        "volume": 1000, "trading_value": 10000, "available_at": dates,
    })
    features = build_features(prices, target_horizon_days=5)
    assert features.loc[0, "label_end_time"] == dates[5]
    assert features.loc[0, "target_horizon_days"] == 5


def test_purged_fold_removes_overlapping_labels():
    dates = pd.bdate_range("2022-01-01", periods=160)
    boundary = dates[100]
    frame = pd.DataFrame({
        "date": dates, "label_end_time": pd.Series(dates).shift(-10),
        "feature_available_at": dates,
    })
    fold = {
        "fold": 0, "train_start": dates[0], "train_end": boundary,
        "validation_start": boundary, "validation_end": dates[130],
        "test_start": dates[130], "test_end": dates[-1], "embargo_days": 2,
    }
    train, validation, _, audit = purged_fold_frames(frame, fold)
    assert (train.label_end_time < boundary - pd.Timedelta(days=2)).all()
    assert (validation.label_end_time < dates[130] - pd.Timedelta(days=2)).all()
    assert audit["train_rows_purged"] > 0


def test_fold_boundaries_are_chronological():
    folds = make_folds(pd.Series(pd.bdate_range("2020-01-01", "2024-12-31")), 24, 3, 1, 4, 20)
    assert len(folds) == 4
    assert all(f["train_end"] == f["validation_start"] for f in folds)
    assert all(f["validation_end"] == f["test_start"] for f in folds)


def test_limited_folds_are_evenly_spread_across_oos_period():
    dates = pd.Series(pd.bdate_range("2020-01-01", "2025-12-31"))
    folds = make_folds(dates, 24, 3, 1, 4, 20, "evenly_spaced")
    assert len(folds) == 4
    assert folds[0]["test_start"].year < folds[-1]["test_start"].year
    assert folds[0]["test_start"] < pd.Timestamp("2024-01-01")


def test_fold_feature_coverage_drops_fully_missing_columns():
    rng = np.random.default_rng(3)
    train = pd.DataFrame({feature: np.nan for feature in FEATURES}, index=range(40))
    validation = pd.DataFrame({feature: np.nan for feature in FEATURES}, index=range(20))
    train["return_5d"] = rng.normal(size=40)
    validation["return_5d"] = rng.normal(size=20)
    train["target_rank"] = rng.uniform(size=40)
    validation["target_rank"] = rng.uniform(size=20)
    bundle = fit_ranker(train, validation, {
        "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
        "min_feature_coverage": 0.1, "seed": 3,
    })
    assert bundle["active_features"] == ["return_5d"]
    assert len(bundle["tuning"]) >= 2


def _aur_inputs():
    tickers = list("ABCDEFGH")
    snapshot = pd.DataFrame({
        "ticker": tickers, "signal": np.linspace(0.0, 1.0, 8),
        "liquidity_20d": np.linspace(10, 20, 8),
        "volatility_20d": np.linspace(0.1, 0.2, 8),
    })
    dates = pd.bdate_range("2022-01-01", periods=80)
    history = pd.DataFrame([
        {"date": date, "ticker": ticker, "ret1": (i + 1) * 0.0001 + j * 0.00001}
        for j, date in enumerate(dates) for i, ticker in enumerate(tickers)
    ])
    return snapshot, history


def test_aur_respects_qubit_budget_and_emits_diagnostics():
    snapshot, history = _aur_inputs()
    reduced = adaptive_reduce(snapshot, history, {
        "candidate_size": 7, "min_candidate_size": 4, "max_candidate_size": 7,
        "qubit_budget": 6, "cardinality": 3, "signal_weight": 0.55,
        "liquidity_weight": 0.2, "risk_weight": 0.15, "correlation_penalty": 0.1,
        "high_correlation_threshold": 0.1,
    })
    assert 3 <= reduced.selected_candidate.sum() <= 6
    assert {"selected_m", "signal_dispersion", "average_abs_correlation",
            "candidate_size_reason"} <= set(reduced.columns)


def test_qubo_ising_energy_equivalence():
    q = np.array([[0.3, -0.2, 0.1], [-0.2, 0.4, 0.05], [0.1, 0.05, -0.1]])
    offset, h, j = qubo_to_ising(q)
    for bits in itertools.product([0, 1], repeat=3):
        bits = np.asarray(bits)
        spins = 1 - 2 * bits
        assert np.isclose(energy(bits, q), ising_energy(spins, offset, h, j))


def test_xy_primary_is_most_probable_and_trace_is_persisted():
    q = np.diag([0.3, -0.2, 0.1, -0.1])
    result = xy_qaoa_statevector(q, 2, 1, 16, 128, 11)
    primary = "".join(map(str, result["bits"]))
    assert np.isclose(result["primary_probability"], max(result["bitstring_probabilities"].values()))
    assert primary in result["bitstring_probabilities"]
    assert result["parameter_trace"]
    assert result["optimizer"] == "COBYLA_multi_start"


def test_xy_noise_is_explicitly_a_proxy():
    result = xy_qaoa_statevector(np.eye(4), 2, 1, 8, 32, 2, 0.1)
    assert result["uniform_probability_noise_proxy"] == 0.1
    assert "noise_proxy" not in result


def test_xy_readout_channel_can_break_measured_cardinality_but_postselects_solution():
    result = xy_qaoa_statevector(
        np.eye(5), 2, 1, 8, 64, 2,
        depolarizing_probability=0.0, readout_error_probability=1.0,
    )
    assert result["noise_model"] == "phenomenological_depolarizing_plus_readout_sampling"
    assert result["feasibility_rate"] == 0.0
    assert result["bits"].sum() == 2


def test_penalty_primary_solution_is_feasible():
    result = penalty_qaoa_statevector(np.diag([0.1, -0.2, 0.05, -0.1]), 2, 1, 12, 64, 3)
    assert result["bits"].sum() == 2
    assert result["parameter_trace"]


def test_buy_and_hold_does_not_rebalance_daily():
    returns = pd.DataFrame({"AAA": [0.10, 0.10], "BBB": [0.0, 0.0]})
    result = simulate_buy_and_hold({"AAA": 0.5, "BBB": 0.5}, returns)
    assert np.isclose(result["gross_wealth"].iloc[-1], 0.5 * 1.1 * 1.1 + 0.5)
    assert result["ending_weights"]["AAA"] > 0.5


def test_transaction_cost_is_debited_exactly_at_rebalance():
    returns = pd.DataFrame({"AAA": [0.0], "BBB": [0.0]})
    result = simulate_buy_and_hold({"AAA": 0.5, "BBB": 0.5}, returns, 0.01)
    assert np.isclose(result["net_returns"].iloc[0], -0.01)
    assert np.isclose(result["gross_returns"].iloc[0], 0.0)


def test_simulator_rejects_unresolved_missing_returns():
    with pytest.raises(ValueError, match="unresolved"):
        simulate_buy_and_hold({"AAA": 1.0}, pd.DataFrame({"AAA": [np.nan]}))


def test_realized_panel_requires_verified_delisting_for_long_suffix():
    dates = pd.bdate_range("2024-01-01", periods=10)
    test = pd.DataFrame({
        "date": dates[:3], "ticker": "AAA", "ret1": [0.01, 0.0, -0.01],
    })
    # Include the market calendar through other securities.
    test = pd.concat([test, pd.DataFrame({"date": dates, "ticker": "BBB", "ret1": 0.0})])
    master = pd.DataFrame({"ticker": ["AAA", "BBB"], "delisting_date": [pd.NaT, pd.NaT]})
    with pytest.raises(ValueError, match="without a verified delisting"):
        prepare_realized_return_panel(test, ["AAA"], master, research_mode=True,
                                      maximum_unexplained_gap_days=2)
    master.loc[master.ticker == "AAA", "delisting_date"] = dates[3]
    panel, events = prepare_realized_return_panel(
        test, ["AAA"], master, research_mode=True, maximum_unexplained_gap_days=2,
    )
    assert panel.loc[dates[3], "AAA"] == -1.0
    assert any(row["event"] == "verified_delisting_liquidation" for row in events)


def test_transaction_cost_breakdown_separates_sell_tax_and_impact():
    total, details = transaction_cost_breakdown(
        {"AAA": 0.2, "BBB": -0.3}, commission_bps=10, sell_tax_bps=10,
        slippage_bps=5, impact_coefficient=0.001,
        adv_capacity_weights={"AAA": 0.5, "BBB": 0.5},
    )
    assert total > 0
    assert details["AAA"]["sell_tax_cost"] == 0
    assert details["BBB"]["sell_tax_cost"] > 0
    assert details["BBB"]["market_impact_cost"] > 0


def test_common_strategy_recorder_creates_cost_ledger_inputs():
    previous, weights, trades, returns = {}, [], [], []
    fold = {"fold": 0, "test_start": pd.Timestamp("2023-01-31")}
    test_returns = pd.DataFrame({"AAA": [0.01], "BBB": [0.0]}, index=[pd.Timestamp("2023-02-01")])
    result = record_rebalanced_strategy(
        "benchmark", fold, {"AAA": 0.5, "BBB": 0.5}, test_returns,
        previous, 0.001, weights, trades, returns,
    )
    assert result["transaction_cost"] > 0
    assert trades and all(row["strategy"] == "benchmark" for row in trades)
    assert {"gross_return", "net_return"} <= set(returns[0])


def test_centered_block_bootstrap_reports_method():
    a = pd.Series(np.linspace(-0.01, 0.02, 60))
    b = pd.Series(np.linspace(-0.01, 0.01, 60))
    result = paired_block_bootstrap_test(a, b, 3, samples=100, block=5)
    assert result["bootstrap_centered_under_null"] is True
    assert 0 <= result["p_value"] <= 1


def test_research_block_writes_no_metrics(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2020-01-01", "2023-12-31", ["AAA", "BBB", "CCC"])
    config = tmp_path / "research.yaml"
    config.write_text("mode: research\nlabel: blocked\ndata: {source: fixture}\n", encoding="utf-8")
    with pytest.raises(ResearchRunBlocked) as caught:
        run_experiment(tmp_path, config)
    out = caught.value.output_dir
    assert json.loads((out / "manifest.json").read_text())["status"] == "blocked"
    assert (out / "RESEARCH_BLOCKED.md").exists()
    assert not (out / "metrics_long.csv").exists()


def test_blocked_run_passes_honesty_audit_with_explicit_flag(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(paths, "2020-01-01", "2023-12-31", ["AAA", "BBB", "CCC"])
    config = tmp_path / "research.yaml"
    config.write_text("mode: research\nlabel: blocked\ndata: {source: fixture}\n", encoding="utf-8")
    with pytest.raises(ResearchRunBlocked) as caught:
        run_experiment(tmp_path, config)
    script = Path(__file__).parents[1] / "scripts" / "audit_research_run.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(caught.value.output_dir), "--allow-blocked"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert "blocked_valid" in completed.stdout


def test_point_in_time_import_requires_source_url(tmp_path: Path):
    source = tmp_path / "membership.csv"
    pd.DataFrame({
        "ticker": ["AAA"], "effective_from": ["2022-01-01"],
        "effective_to": [None], "available_at": ["2021-12-20"], "source": ["official"],
    }).to_csv(source, index=False)
    with pytest.raises(ValueError, match="source_url"):
        import_point_in_time_table(
            source, tmp_path / "out.parquet",
            {"ticker", "effective_from", "effective_to", "available_at", "source"},
            "index_membership",
        )


def test_benchmark_import_rejects_price_index_masquerading_as_total_return(tmp_path: Path):
    source = tmp_path / "benchmark.csv"
    pd.DataFrame([{
        "date": "2022-01-03", "benchmark": "VNINDEX", "total_return_index": 1000,
        "index_type": "price", "methodology_url": "https://example.test/method",
        "available_at": "2022-01-04", "source": "provider",
        "source_url": "https://example.test/data",
    }]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="total_return"):
        import_point_in_time_table(
            source, tmp_path / "outputs" / "normalized" / "benchmark.parquet",
            {"date", "benchmark", "total_return_index", "index_type", "methodology_url",
             "available_at", "source", "source_url"}, "benchmark",
        )


def test_security_master_import_requires_stable_security_id(tmp_path: Path):
    source = tmp_path / "master.csv"
    pd.DataFrame([{
        "security_id": "", "ticker": "AAA", "exchange": "HOSE",
        "listing_date": "2022-01-01", "delisting_date": None,
        "effective_from": "2022-01-01", "effective_to": None,
        "available_at": "2021-12-20", "history_method": "exchange_listing_history",
        "source": "official", "source_url": "https://example.test/listing",
    }]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="security_id"):
        import_point_in_time_table(
            source, tmp_path / "outputs" / "normalized" / "security_master.parquet",
            {"security_id", "ticker", "exchange", "listing_date", "delisting_date",
             "effective_from", "effective_to", "available_at", "history_method",
             "source", "source_url"}, "security_master",
        )


def test_successful_research_mode_run_is_auditable_and_tamper_evident(tmp_path: Path):
    paths = Paths(tmp_path)
    generate_fixture(
        paths, "2020-01-01", "2025-12-31",
        ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"], 9,
    )
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    prices["data_class"] = "real"
    prices["source"] = "authorized_test_market"
    prices["source_url"] = "https://example.test/market"
    prices["adjustment_policy"] = "verified_corporate_action_adjusted"
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    contract = tmp_path / "adjustment_contract.json"
    contract.write_text(json.dumps({
        "price_dataset_sha256": sha256_file(paths.normalized / "prices.parquet"),
        "adjustment_policy": "verified_corporate_action_adjusted",
        "source": "authorized_test_market",
        "source_url": "https://example.test/adjustment-methodology",
        "methodology": "integration-test adjusted total-return contract",
        "certified_by": "integration-test",
        "certified_at": "2026-08-06T00:00:00Z",
    }), encoding="utf-8")
    apply_price_adjustment_contract(paths, contract)
    master = pd.read_parquet(paths.normalized / "security_master.parquet")
    master["data_class"] = "real"
    master["source"] = "official_exchange_test_history"
    master["source_url"] = "https://example.test/exchange-history"
    master["history_method"] = "exchange_listing_history"
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)
    for optional in ["financial_statements", "macro", "foreign_flow", "index_membership"]:
        (paths.normalized / f"{optional}.parquet").unlink(missing_ok=True)
    config = Path(__file__).parents[1] / "configs" / "quick.yaml"
    research_config = tmp_path / "research.yaml"
    text = config.read_text(encoding="utf-8").replace("mode: demo_fixture", "mode: research")
    text = text.replace('label: "NOT RESEARCH RESULT"', 'label: "AUTHORIZED INTEGRATION TEST"')
    text = text.replace("  source: fixture", "  source: configured")
    research_config.write_text(text, encoding="utf-8")
    out = run_experiment(tmp_path, research_config)
    script = Path(__file__).parents[1] / "scripts" / "audit_research_run.py"
    completed = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    metrics = out / "metrics_long.csv"
    metrics.write_text(metrics.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True)
    assert tampered.returncode == 1
    assert "SHA-256 mismatch" in tampered.stdout


def test_source_files_are_valid_utf8_without_replacement_character():
    root = Path(__file__).parents[1]
    files = [*root.glob("*.md"), *root.glob("src/*.py"), *root.glob("configs/*.yaml"),
             *root.glob("docs/*.md")]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text, path
