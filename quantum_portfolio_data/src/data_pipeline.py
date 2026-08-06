from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLUMNS = [
    "date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume",
    "trading_value", "source", "source_url", "fetched_at", "available_at",
    "raw_checksum", "parser_version", "data_class",
]

PROVENANCE_COLUMNS = {"source", "source_url", "fetched_at", "raw_checksum"}
HISTORICAL_UNIVERSE_METHODS = {
    "official_event_history",
    "exchange_listing_history",
    "verified_membership_history",
    "fixture",
}
VERIFIED_ADJUSTMENT_POLICIES = {
    "verified_corporate_action_adjusted",
    "unadjusted_with_verified_actions_join",
    "fixture",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "outputs" / "raw"

    @property
    def normalized(self) -> Path:
        return self.root / "outputs" / "normalized"

    @property
    def curated(self) -> Path:
        return self.root / "outputs" / "curated"

    @property
    def reports(self) -> Path:
        return self.root / "outputs" / "reports"

    def ensure(self) -> None:
        for p in (self.raw, self.normalized, self.curated, self.reports):
            p.mkdir(parents=True, exist_ok=True)


def generate_fixture(
    paths: Paths,
    start: str,
    end: str,
    tickers: list[str],
    seed: int = 42,
) -> dict:
    """Create deterministic synthetic market data, explicitly marked as fixture."""
    paths.ensure()
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    market = rng.normal(0.00025, 0.009, len(dates))
    frames = []
    master = []
    for i, ticker in enumerate(tickers):
        beta = 0.7 + 0.12 * i
        alpha = (i - len(tickers) / 2) * 0.000015
        ret = alpha + beta * market + rng.normal(0, 0.006 + i * 0.00035, len(dates))
        close = (30 + 4 * i) * np.exp(np.cumsum(ret))
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        spread = np.abs(rng.normal(0.005, 0.002, len(dates)))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        volume = rng.integers(50_000, 2_000_000, len(dates))
        frame = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adjusted_close": close,
            "volume": volume,
            "trading_value": volume * close,
        })
        frames.append(frame)
        master.append({
            "ticker": ticker, "company_name": f"Fixture Company {ticker}",
            "exchange": "HOSE_FIXTURE", "industry": f"Industry {i % 4}",
            "sector": f"Sector {i % 3}", "listing_date": dates[0],
            "delisting_date": pd.NaT, "effective_from": dates[0],
            "effective_to": pd.NaT, "available_at": dates[0],
            "source": "deterministic_fixture", "data_class": "fixture",
            "source_url": "local://tests/fixture", "fetched_at": pd.Timestamp.utcnow(),
            "raw_checksum": "pending", "history_method": "fixture",
        })
    prices = pd.concat(frames, ignore_index=True)
    raw_csv = paths.raw / "fixture_prices.csv"
    prices.to_csv(raw_csv, index=False)
    checksum = sha256_file(raw_csv)
    now = datetime.now(timezone.utc).isoformat()
    prices["source"] = "deterministic_fixture"
    prices["source_url"] = "local://tests/fixture"
    prices["fetched_at"] = now
    prices["available_at"] = prices["date"]
    prices["raw_checksum"] = checksum
    prices["parser_version"] = "fixture-v1"
    prices["data_class"] = "fixture"
    prices["adjustment_policy"] = "fixture"
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    master_df = pd.DataFrame(master)
    master_df["raw_checksum"] = checksum
    master_df.to_parquet(paths.normalized / "security_master.parquet", index=False)
    pd.DataFrame(columns=[
        "ticker", "event_type", "event_date", "announcement_date", "effective_date",
        "available_at", "cash_dividend", "split_ratio", "adjustment_factor", "source",
    ]).to_parquet(paths.normalized / "corporate_actions.parquet", index=False)
    pd.DataFrame([
        {
            "ticker": ticker, "index_code": "VN30_FIXTURE",
            "effective_from": dates[0], "effective_to": dates[-1],
            "announcement_date": dates[0], "available_at": dates[0],
            "source": "deterministic_fixture", "data_class": "fixture",
            "source_url": "local://tests/fixture", "fetched_at": now,
            "raw_checksum": checksum, "history_method": "fixture",
        }
        for ticker in tickers[: min(4, len(tickers))]
    ]).to_parquet(paths.normalized / "index_membership.parquet", index=False)
    quarter_ends = pd.date_range(dates[0], dates[-1], freq="QE")
    statements = []
    for i, ticker in enumerate(tickers):
        for quarter in quarter_ends:
            publication = quarter + pd.Timedelta(days=35)
            statements.append({
                "ticker": ticker, "fiscal_period_end": quarter,
                "publication_date": publication, "available_at": publication,
                "revenue": float(1000 + i * 50), "net_income": float(80 + i * 5),
                "total_assets": float(2000 + i * 100), "equity": float(1000 + i * 60),
                "source": "deterministic_fixture", "data_class": "fixture",
            })
    pd.DataFrame(statements).to_parquet(
        paths.normalized / "financial_statements.parquet", index=False
    )
    month_ends = pd.date_range(dates[0], dates[-1], freq="ME")
    pd.DataFrame({
        "series_id": "FIXTURE_POLICY_RATE", "observation_date": month_ends,
        "release_date": month_ends + pd.Timedelta(days=5),
        "available_at": month_ends + pd.Timedelta(days=5), "value": 0.04,
        "source": "deterministic_fixture", "data_class": "fixture",
    }).to_parquet(paths.normalized / "macro.parquet", index=False)
    pd.DataFrame({
        "date": dates, "ticker": "MARKET", "available_at": dates,
        "foreign_net_value": rng.normal(0, 1e9, len(dates)),
        "source": "deterministic_fixture", "data_class": "fixture",
    }).to_parquet(paths.normalized / "foreign_flow.parquet", index=False)
    manifest = {
        "status": "success", "data_class": "fixture", "label": "NOT RESEARCH RESULT",
        "records": len(prices), "tickers": len(tickers), "start": str(dates.min().date()),
        "end": str(dates.max().date()), "source": "deterministic_fixture",
        "raw_checksum": checksum, "generated_at": now,
    }
    (paths.raw / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def import_csv(paths: Paths, input_path: Path, source: str, source_url: str) -> dict:
    """Import a user-authorized CSV source without guessing or scraping endpoints."""
    paths.ensure()
    df = pd.read_csv(input_path)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    checksum = sha256_file(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df["adjusted_close"] = df.get("adjusted_close", df["close"])
    df["trading_value"] = df.get("trading_value", df["volume"] * df["close"])
    df["source"] = source
    df["source_url"] = source_url
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    df["available_at"] = pd.to_datetime(df.get("available_at", df["date"]))
    df["raw_checksum"] = checksum
    df["parser_version"] = "csv-v1"
    df["data_class"] = "real"
    df[PRICE_COLUMNS].to_parquet(paths.normalized / "prices.parquet", index=False)
    return {"records": len(df), "data_class": "real", "raw_checksum": checksum}


def validate_data(paths: Paths) -> tuple[dict, pd.DataFrame]:
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    issues: list[dict] = []
    duplicated = prices.duplicated(["date", "ticker"], keep=False)
    if duplicated.any():
        issues.append({"severity": "error", "check": "unique_date_ticker", "count": int(duplicated.sum())})
    bad_ohlc = (
        (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
        | (prices[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    if bad_ohlc.any():
        issues.append({"severity": "error", "check": "ohlc_logic", "count": int(bad_ohlc.sum())})
    negative = (prices[["volume", "trading_value"]] < 0).any(axis=1)
    if negative.any():
        issues.append({"severity": "error", "check": "nonnegative_volume_value", "count": int(negative.sum())})
    future = pd.to_datetime(prices["available_at"]) < pd.to_datetime(prices["date"])
    if future.any():
        issues.append({"severity": "error", "check": "available_before_observation", "count": int(future.sum())})
    returns = prices.sort_values(["ticker", "date"]).groupby("ticker")["adjusted_close"].pct_change()
    outliers = returns.abs() > 0.30
    if outliers.any():
        issues.append({"severity": "warning", "check": "return_outlier_flag", "count": int(outliers.sum())})
    coverage = (
        prices.assign(year=pd.to_datetime(prices["date"]).dt.year)
        .groupby(["ticker", "year", "source", "data_class"], dropna=False)
        .agg(records=("date", "size"), start=("date", "min"), end=("date", "max"),
             missing_close=("close", lambda s: int(s.isna().sum())))
        .reset_index()
    )
    report = {
        "status": "pass" if not any(i["severity"] == "error" for i in issues) else "fail",
        "data_class": sorted(prices["data_class"].astype(str).unique().tolist()),
        "records": len(prices), "tickers": prices["ticker"].nunique(),
        "start": str(pd.to_datetime(prices["date"]).min().date()),
        "end": str(pd.to_datetime(prices["date"]).max().date()),
        "issues": issues,
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    coverage.to_csv(paths.reports / "coverage.csv", index=False)
    return report, coverage


def build_universe(paths: Paths, rebalance: str = "monthly") -> pd.DataFrame:
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    master = pd.read_parquet(paths.normalized / "security_master.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    master["listing_date"] = pd.to_datetime(master["listing_date"])
    master["delisting_date"] = pd.to_datetime(master["delisting_date"])
    master["available_at"] = pd.to_datetime(master["available_at"])
    master["effective_from"] = pd.to_datetime(master.get("effective_from", master["listing_date"]))
    master["effective_to"] = pd.to_datetime(master.get("effective_to", master["delisting_date"]))
    anchors = (
        prices.set_index("date").groupby("ticker")["close"].resample("ME").last().dropna()
        .reset_index()["date"].drop_duplicates().sort_values()
    )
    rows = []
    for date in anchors:
        for row in master.itertuples():
            eligible = (
                row.listing_date <= date
                and row.effective_from <= date
                and row.available_at <= date
                and (pd.isna(row.delisting_date) or row.delisting_date >= date)
                and (pd.isna(row.effective_to) or row.effective_to >= date)
            )
            if eligible:
                rows.append({"decision_time": date, "ticker": row.ticker, "eligible": True,
                             "reason": "listed_and_available_point_in_time",
                             "data_class": row.data_class,
                             "effective_from": row.effective_from,
                             "effective_to": row.effective_to,
                             "record_available_at": row.available_at,
                             "source": row.source,
                             "source_url": getattr(row, "source_url", None),
                             "fetched_at": getattr(row, "fetched_at", None),
                             "raw_checksum": getattr(row, "raw_checksum", None),
                             "history_method": getattr(row, "history_method", None)})
    universe = pd.DataFrame(rows)
    universe.to_parquet(paths.curated / "universe_monthly.parquet", index=False)
    return universe


def leakage_audit(paths: Paths) -> dict:
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    master = pd.read_parquet(paths.normalized / "security_master.parquet")
    actions_path = paths.normalized / "corporate_actions.parquet"
    actions = pd.read_parquet(actions_path) if actions_path.exists() else pd.DataFrame()
    membership_path = paths.normalized / "index_membership.parquet"
    membership = pd.read_parquet(membership_path) if membership_path.exists() else pd.DataFrame()
    auxiliary_names = [
        "index_membership", "corporate_actions", "financial_statements",
        "macro", "foreign_flow",
    ]
    auxiliary = {}
    fixture_auxiliary = []
    for name in auxiliary_names:
        path = paths.normalized / f"{name}.parquet"
        if not path.exists():
            auxiliary[name] = "missing"
            continue
        table = pd.read_parquet(path)
        is_fixture = bool(
            ("data_class" in table and table["data_class"].astype(str).eq("fixture").any())
            or ("source" in table and table["source"].astype(str).str.contains("fixture", case=False).any())
        )
        auxiliary[name] = "fixture" if is_fixture else "real_or_empty"
        if is_fixture:
            fixture_auxiliary.append(name)
    master_has_contract = bool(
        {"listing_date", "delisting_date", "effective_from", "effective_to", "available_at",
         "history_method"} | PROVENANCE_COLUMNS <= set(master.columns)
    )
    history_methods = set(master.get("history_method", pd.Series(dtype=str)).dropna().astype(str))
    trusted_history = bool(history_methods) and history_methods <= HISTORICAL_UNIVERSE_METHODS
    master_times_valid = False
    if {"listing_date", "effective_from", "available_at"} <= set(master.columns):
        listing = pd.to_datetime(master["listing_date"], errors="coerce")
        effective = pd.to_datetime(master["effective_from"], errors="coerce")
        available = pd.to_datetime(master["available_at"], errors="coerce")
        master_times_valid = bool(
            listing.notna().all() and effective.notna().all() and available.notna().all()
        )
    universe_path = paths.curated / "universe_monthly.parquet"
    universe = pd.read_parquet(universe_path) if universe_path.exists() else pd.DataFrame()
    universe_contract = bool(
        not universe.empty
        and {"decision_time", "ticker", "eligible", "effective_from", "record_available_at",
             "source", "source_url", "fetched_at", "raw_checksum", "history_method"}
        <= set(universe.columns)
    )
    universe_times_valid = False
    universe_methods: set[str] = set()
    universe_fixture_free = False
    if universe_contract:
        decision = pd.to_datetime(universe["decision_time"], errors="coerce")
        effective = pd.to_datetime(universe["effective_from"], errors="coerce")
        available = pd.to_datetime(universe["record_available_at"], errors="coerce")
        universe_times_valid = bool(
            decision.notna().all() and effective.notna().all() and available.notna().all()
            and (effective <= decision).all() and (available <= decision).all()
        )
        universe_methods = set(universe["history_method"].dropna().astype(str))
        universe_fixture_free = not universe.get(
            "data_class", pd.Series(dtype=str)
        ).astype(str).eq("fixture").any()
    membership_contract = bool(
        not membership.empty
        and {"ticker", "effective_from", "effective_to", "available_at",
             "history_method"} | PROVENANCE_COLUMNS <= set(membership.columns)
        and not membership.get("data_class", pd.Series(dtype=str)).astype(str).eq("fixture").any()
        and set(membership["history_method"].dropna().astype(str)) <= HISTORICAL_UNIVERSE_METHODS
    )
    actions_contract = bool(
        not actions.empty
        and {"ticker", "announcement_date", "effective_date", "available_at"}
        | PROVENANCE_COLUMNS <= set(actions.columns)
    )
    price_times_valid = bool(
        (pd.to_datetime(prices["available_at"], errors="coerce")
         >= pd.to_datetime(prices["date"], errors="coerce")).all()
    )
    adjustment_policies = set(
        prices.get("adjustment_policy", pd.Series(dtype=str)).dropna().astype(str)
    )
    checks = {
        "historical_universe_contract": master_has_contract,
        "historical_universe_source_trusted": trusted_history,
        "historical_universe_fixture_free": not master.get(
            "data_class", pd.Series(dtype=str)
        ).astype(str).eq("fixture").any(),
        "historical_universe_times_valid": master_times_valid,
        "universe_snapshots_built_with_provenance": universe_contract,
        "universe_snapshot_times_valid": universe_times_valid,
        "universe_snapshot_source_trusted": bool(universe_methods)
        and universe_methods <= HISTORICAL_UNIVERSE_METHODS,
        "universe_snapshot_fixture_free": universe_fixture_free,
        "historical_membership_events_available": membership_contract,
        "real_prices_only": not prices["data_class"].astype(str).eq("fixture").any(),
        "auxiliary_tables_fixture_free": not fixture_auxiliary,
        "corporate_actions_point_in_time_available": actions_contract,
        "price_adjustment_policy_verified": bool(adjustment_policies)
        and adjustment_policies <= VERIFIED_ADJUSTMENT_POLICIES,
        "availability_not_before_observation": price_times_valid,
    }
    fixture_only = prices["data_class"].eq("fixture").all()
    blockers = [] if fixture_only else [k for k, v in checks.items() if not v]
    limitations = [
        f"{name}_not_configured"
        for name, status in auxiliary.items()
        if status == "missing"
    ]
    report = {
        "status": (
            "pass_for_fixture_demo" if fixture_only
            else ("blocked" if blockers else ("pass_with_limitations" if limitations else "pass"))
        ),
        "checks": checks, "blockers": blockers,
        "limitations": limitations,
        "auxiliary_tables": auxiliary,
        "note": (
            "Historical universe/membership, corporate actions and adjustment policy are core "
            "research contracts. Optional PIT features are excluded when unavailable. Fixture "
            "auxiliary tables block a real-data run."
        ),
    }
    (paths.reports / "leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
