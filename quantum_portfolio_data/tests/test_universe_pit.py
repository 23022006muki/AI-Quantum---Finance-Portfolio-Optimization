from __future__ import annotations

import pandas as pd

from src.data_pipeline import Paths
from src.universe_pit import build_historical_universe_pit


def test_universe_uses_only_prior_data_and_keeps_delisted_names(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    master = pd.DataFrame([
        {
            "security_id": "HOSE:OLD", "ticker": "OLD", "company_name": "Old",
            "exchange": "HOSE", "listing_date": "2019-01-01", "delisting_date": "2020-03-15",
            "effective_from": "2019-01-01", "effective_to": "2020-03-15",
            "available_at": "2018-12-01", "source": "official", "source_url": "https://example.test",
            "raw_checksum": "a", "history_method": "exchange_listing_history",
        },
        {
            "security_id": "HOSE:NEW", "ticker": "NEW", "company_name": "New",
            "exchange": "HOSE", "listing_date": "2020-02-15", "delisting_date": pd.NaT,
            "effective_from": "2020-02-15", "effective_to": pd.NaT,
            "available_at": "2020-02-01", "source": "official", "source_url": "https://example.test",
            "raw_checksum": "b", "history_method": "exchange_listing_history",
        },
    ])
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)
    dates = pd.bdate_range("2020-01-01", "2020-04-30")
    prices = pd.concat([
        pd.DataFrame({
            "date": dates, "ticker": ticker, "trading_value": 1_000_000.0,
            "close": 10.0, "adjusted_close": 10.0,
        }) for ticker in ["OLD", "NEW"]
    ], ignore_index=True)
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)

    result = build_historical_universe_pit(
        paths, "2020-01-01", "2020-04-30", lookback_days=45, minimum_observations=5
    )
    universe = pd.read_parquet(
        tmp_path / "outputs" / "research_v2" / "curated" / "universe_monthly_pit.parquet"
    )
    january = universe[universe["decision_date"].dt.month.eq(1)]
    march = universe[universe["decision_date"].dt.month.eq(3)]
    assert not january.loc[january["ticker"].eq("NEW"), "eligible"].item()
    assert not march.loc[march["ticker"].eq("OLD"), "eligible"].item()
    assert result["delisted_securities_retained_in_master"] == 1
    assert result["whole_sample_completeness_filter_used"] is False
