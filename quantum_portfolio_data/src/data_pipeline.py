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
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    pd.DataFrame(master).to_parquet(paths.normalized / "security_master.parquet", index=False)
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
    anchors = (
        prices.set_index("date").groupby("ticker")["close"].resample("ME").last().dropna()
        .reset_index()["date"].drop_duplicates().sort_values()
    )
    rows = []
    for date in anchors:
        for row in master.itertuples():
            eligible = (
                row.listing_date <= date
                and row.available_at <= date
                and (pd.isna(row.delisting_date) or row.delisting_date >= date)
            )
            if eligible:
                rows.append({"decision_time": date, "ticker": row.ticker, "eligible": True,
                             "reason": "listed_and_available_point_in_time",
                             "data_class": row.data_class})
    universe = pd.DataFrame(rows)
    universe.to_parquet(paths.curated / "universe_monthly.parquet", index=False)
    return universe


def leakage_audit(paths: Paths) -> dict:
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    master = pd.read_parquet(paths.normalized / "security_master.parquet")
    actions_path = paths.normalized / "corporate_actions.parquet"
    actions = pd.read_parquet(actions_path) if actions_path.exists() else pd.DataFrame()
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
    checks = {
        "current_universe_not_backfilled": bool({"listing_date", "delisting_date", "available_at"} <= set(master.columns)),
        "real_prices_only": not prices["data_class"].astype(str).eq("fixture").any(),
        "auxiliary_tables_fixture_free": not fixture_auxiliary,
        "corporate_action_timestamps": bool(actions.empty or {"announcement_date", "effective_date", "available_at"} <= set(actions.columns)),
        "available_at_not_after_decision_observation": bool(
            (pd.to_datetime(prices["available_at"]) <= pd.to_datetime(prices["date"])).all()
        ),
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
            "Missing optional PIT tables are excluded from features and reported as limitations; "
            "fixture auxiliary tables block a real-data run."
        ),
    }
    (paths.reports / "leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
