from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_pipeline import Paths, sha256_file


def _hash_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def build_historical_universe_pit(
    paths: Paths,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    lookback_days: int = 90,
    minimum_observations: int = 40,
) -> dict:
    """Build a decision-date universe without whole-sample completeness filters.

    Official listing and delisting dates define legal listing eligibility. Data-history
    and liquidity conditions are recomputed from observations strictly before each
    decision date. Suspension and sector fields stay unknown when no PIT source exists.
    """
    workspace = paths.root / "outputs" / "research_v2"
    normalized = workspace / "normalized"
    curated = workspace / "curated"
    reports = workspace / "reports"
    for directory in (normalized, curated, reports):
        directory.mkdir(parents=True, exist_ok=True)
    master_path = paths.normalized / "security_master.parquet"
    price_path = paths.normalized / "prices.parquet"
    if not master_path.exists() or not price_path.exists():
        raise FileNotFoundError("Official security master and price panel are required.")
    master = pd.read_parquet(master_path).copy()
    prices = pd.read_parquet(price_path).copy()
    for column in ["listing_date", "delisting_date", "effective_from", "effective_to", "available_at"]:
        master[column] = pd.to_datetime(master[column], errors="coerce")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])

    pit = master.copy()
    pit["sector"] = pd.NA
    pit["suspension_start"] = pd.NaT
    pit["suspension_end"] = pd.NaT
    pit["status"] = np.where(pit["delisting_date"].notna(), "delisted", "listed")
    pit["delisting_known_at"] = pit["delisting_date"]
    pit["suspension_data_status"] = "unavailable_fail_closed"
    pit["sector_data_status"] = "unavailable_not_used"
    pit["checksum"] = pit.get("raw_checksum", pd.Series(index=pit.index, dtype=str))
    required_order = [
        "security_id", "ticker", "company_name", "exchange", "sector",
        "listing_date", "delisting_date", "suspension_start", "suspension_end",
        "status", "effective_from", "effective_to", "available_at", "source",
        "source_url", "checksum", "delisting_known_at", "suspension_data_status",
        "sector_data_status", "history_method",
    ]
    pit = pit[required_order].sort_values(["ticker", "effective_from"])
    pit_path = normalized / "security_master_pit.parquet"
    pit.to_parquet(pit_path, index=False)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    dates = prices.loc[prices["date"].between(start_ts, end_ts), "date"].drop_duplicates()
    decision_dates = (
        pd.DataFrame({"date": dates})
        .assign(month=lambda x: x["date"].dt.to_period("M"))
        .groupby("month", as_index=False)["date"].max()["date"]
        .sort_values()
        .tolist()
    )
    rows: list[dict] = []
    for decision in decision_dates:
        cutoff = pd.Timestamp(decision)
        lookback_start = cutoff - pd.Timedelta(days=lookback_days)
        history = prices[(prices["date"] < cutoff) & (prices["date"] >= lookback_start)]
        stats = history.groupby("ticker").agg(
            observations=("date", "nunique"),
            last_observation=("date", "max"),
            median_trading_value=("trading_value", "median"),
        )
        for item in pit.itertuples(index=False):
            listed = pd.notna(item.listing_date) and item.listing_date <= cutoff
            not_delisted = pd.isna(item.delisting_date) or item.delisting_date > cutoff
            known = pd.notna(item.available_at) and item.available_at <= cutoff
            ticker_stats = stats.loc[item.ticker] if item.ticker in stats.index else None
            observations = int(ticker_stats["observations"]) if ticker_stats is not None else 0
            trading_value = (
                float(ticker_stats["median_trading_value"])
                if ticker_stats is not None and pd.notna(ticker_stats["median_trading_value"])
                else np.nan
            )
            sufficient = observations >= minimum_observations
            liquid = pd.notna(trading_value) and trading_value > 0
            eligible = bool(listed and not_delisted and known and sufficient and liquid)
            reasons = []
            if not listed: reasons.append("not_yet_listed")
            if not not_delisted: reasons.append("delisted")
            if not known: reasons.append("identity_not_available")
            if not sufficient: reasons.append("insufficient_prior_history")
            if not liquid: reasons.append("no_positive_prior_trading_value")
            rows.append({
                "decision_date": cutoff, "security_id": item.security_id,
                "ticker": item.ticker, "eligible": eligible,
                "eligibility_reason": "eligible" if eligible else "|".join(reasons),
                "prior_observations": observations,
                "median_trading_value_prior_window": trading_value,
                "listing_date": item.listing_date, "delisting_date": item.delisting_date,
                "security_available_at": item.available_at,
                "suspension_check": "unknown_not_filtered",
                "source": item.source, "source_url": item.source_url,
            })
    universe = pd.DataFrame(rows)
    universe_path = curated / "universe_monthly_pit.parquet"
    universe.to_parquet(universe_path, index=False)
    universe.to_csv(reports / "universe_eligibility_audit.csv", index=False)

    future_completeness_filter = False
    survivorship = {
        "status": "partial_blocked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "security_master_rows": len(pit),
        "decision_dates": len(decision_dates),
        "eligible_ticker_decision_rows": int(universe["eligible"].sum()),
        "unique_eligible_tickers": int(universe.loc[universe["eligible"], "ticker"].nunique()),
        "delisted_securities_retained_in_master": int(pit["delisting_date"].notna().sum()),
        "whole_sample_completeness_filter_used": future_completeness_filter,
        "suspension_history_available": False,
        "sector_pit_available": False,
        "blockers": [
            "historical suspension intervals are unavailable",
            "official historical master may not cover every renamed or merged legal entity",
        ],
        "security_master_sha256": sha256_file(pit_path),
        "universe_sha256": sha256_file(universe_path),
        "input_security_master_sha256": sha256_file(master_path),
        "input_price_sha256": sha256_file(price_path),
        "eligibility_logic_sha256": _hash_frame(universe[[
            "decision_date", "security_id", "eligible", "eligibility_reason"
        ]]),
    }
    (reports / "survivorship_bias_audit.json").write_text(
        json.dumps(survivorship, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return survivorship
