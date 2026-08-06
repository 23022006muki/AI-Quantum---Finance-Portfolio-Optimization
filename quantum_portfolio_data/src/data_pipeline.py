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
    "raw_checksum", "parser_version", "data_class", "adjustment_policy",
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
UNIVERSE_DEFINITIONS = {"hose_all_listed", "index_membership"}


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


def quarantine_fixture_auxiliary(paths: Paths) -> list[str]:
    """Move stale fixture-only auxiliary tables out of a real-data workspace."""
    names = ["index_membership", "corporate_actions", "financial_statements", "macro", "foreign_flow"]
    fixture_paths: list[Path] = []
    for name in names:
        path = paths.normalized / f"{name}.parquet"
        if not path.exists():
            continue
        table = pd.read_parquet(path)
        if table.empty:
            continue
        fixture = bool(
            ("data_class" in table and table["data_class"].astype(str).eq("fixture").all())
            or ("source" in table and table["source"].astype(str).str.contains(
                "fixture", case=False, na=False
            ).all())
        )
        if fixture:
            fixture_paths.append(path)
    if not fixture_paths:
        return []
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    destination = paths.root / "outputs" / "quarantine" / "fixture_auxiliary" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in fixture_paths:
        target = destination / path.name
        path.replace(target)
        moved.append(str(target))
    return moved


def apply_price_adjustment_contract(paths: Paths, contract_path: Path) -> dict:
    """Apply a documented adjustment policy only to the exact certified price panel."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "price_dataset_sha256", "adjustment_policy", "source", "source_url",
        "methodology", "certified_by", "certified_at",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"Price adjustment contract missing fields: {missing}")
    if contract["adjustment_policy"] not in VERIFIED_ADJUSTMENT_POLICIES - {"fixture"}:
        raise ValueError("The adjustment policy is not accepted for a real research run.")
    prices_path = paths.normalized / "prices.parquet"
    before_hash = sha256_file(prices_path)
    if contract["price_dataset_sha256"] != before_hash:
        raise ValueError(
            "Price dataset hash does not match the contract; refusing to certify another panel."
        )
    prices = pd.read_parquet(prices_path)
    if prices["data_class"].astype(str).eq("fixture").any():
        raise ValueError("A real adjustment contract cannot certify fixture prices.")
    prices["adjustment_policy"] = contract["adjustment_policy"]
    prices.to_parquet(prices_path, index=False)
    stored = {
        **contract,
        "contract_file_sha256": sha256_file(contract_path),
        "input_price_dataset_sha256": before_hash,
        "output_price_dataset_sha256": sha256_file(prices_path),
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }
    output = paths.normalized / "price_adjustment_contract.json"
    output.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return stored


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
    (paths.normalized / "price_adjustment_contract.json").write_text(json.dumps({
        "adjustment_policy": "fixture",
        "source": "deterministic_fixture", "source_url": "local://tests/fixture",
        "methodology": "fixture prices require no corporate-action adjustment",
        "certified_by": "fixture_generator", "certified_at": now,
        "output_price_dataset_sha256": sha256_file(paths.normalized / "prices.parquet"),
    }, indent=2), encoding="utf-8")
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
    # Research mode accepts only an explicitly documented adjustment contract.
    # A missing declaration remains usable for inspection but fails the research audit.
    df["adjustment_policy"] = df.get("adjustment_policy", "unverified")
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
    ordered = prices.sort_values(["ticker", "date"]).copy()
    ordered["return_1d"] = ordered.groupby("ticker")["adjusted_close"].pct_change()
    outlier_rows = ordered[ordered["return_1d"].abs() > 0.30].copy()
    if not outlier_rows.empty:
        actions_path = paths.normalized / "corporate_actions.parquet"
        actions = pd.read_parquet(actions_path) if actions_path.exists() else pd.DataFrame()
        if not actions.empty and {"ticker", "effective_date"} <= set(actions.columns):
            actions["effective_date"] = pd.to_datetime(actions["effective_date"], errors="coerce")
            action_dates = {
                ticker: list(group["effective_date"].dropna())
                for ticker, group in actions.groupby("ticker")
            }
            outlier_rows["corporate_action_match"] = [
                any(abs((pd.Timestamp(date) - action_date).days) <= 3
                    for action_date in action_dates.get(ticker, []))
                for ticker, date in zip(outlier_rows["ticker"], outlier_rows["date"])
            ]
        else:
            outlier_rows["corporate_action_match"] = False
        contract_path = paths.normalized / "price_adjustment_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
        contract_matches_dataset = bool(
            contract.get("output_price_dataset_sha256") == sha256_file(
                paths.normalized / "prices.parquet"
            )
        )
        outlier_rows["adjustment_contract_verified"] = contract_matches_dataset & outlier_rows.get(
            "adjustment_policy", pd.Series("unverified", index=outlier_rows.index)
        ).astype(str).isin(VERIFIED_ADJUSTMENT_POLICIES)
        outlier_rows["resolution"] = np.select(
            [outlier_rows["corporate_action_match"], outlier_rows["adjustment_contract_verified"]],
            ["matched_verified_corporate_action", "covered_by_verified_adjustment_contract"],
            default="unresolved",
        )
        unresolved = int(outlier_rows["resolution"].eq("unresolved").sum())
        issues.append({
            "severity": "error" if unresolved else "warning",
            "check": "unresolved_return_outlier" if unresolved else "return_outlier_reviewed",
            "count": unresolved if unresolved else len(outlier_rows),
        })
    outlier_columns = [
        "date", "ticker", "close", "adjusted_close", "return_1d",
        "adjustment_policy", "corporate_action_match",
        "adjustment_contract_verified", "resolution", "source", "source_url",
    ]
    paths.reports.mkdir(parents=True, exist_ok=True)
    outlier_rows.reindex(columns=outlier_columns).to_csv(
        paths.reports / "return_outlier_review.csv", index=False
    )
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
    (paths.reports / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    coverage.to_csv(paths.reports / "coverage.csv", index=False)
    return report, coverage


def build_universe(
    paths: Paths,
    rebalance: str = "monthly",
    definition: str = "hose_all_listed",
    index_code: str | None = None,
) -> pd.DataFrame:
    """Build auditable point-in-time snapshots for one declared universe definition.

    ``hose_all_listed`` uses exchange listing/delisting event history from the security
    master. ``index_membership`` uses effective membership intervals and therefore must
    never be substituted for an all-HOSE study (or vice versa).
    """
    if definition not in UNIVERSE_DEFINITIONS:
        raise ValueError(f"Unsupported universe definition: {definition}")
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
    if definition == "index_membership":
        membership_path = paths.normalized / "index_membership.parquet"
        if not membership_path.exists():
            raise ValueError("index_membership universe requires index_membership.parquet")
        records = pd.read_parquet(membership_path).copy()
        if index_code:
            records = records[records["index_code"].astype(str) == str(index_code)]
        if records.empty:
            raise ValueError("No membership records match the requested index universe.")
        for column in ["effective_from", "effective_to", "available_at"]:
            records[column] = pd.to_datetime(records[column], errors="coerce")
        reason = "index_member_and_available_point_in_time"
    else:
        records = master.copy()
        reason = "listed_on_hose_and_available_point_in_time"
    rows = []
    for date in anchors:
        for row in records.itertuples():
            start = row.effective_from
            end = row.effective_to
            available = row.available_at
            if definition == "hose_all_listed":
                start = max(row.listing_date, start)
                ends = [value for value in (row.delisting_date, end) if pd.notna(value)]
                end = min(ends) if ends else pd.NaT
            eligible = (
                pd.notna(start) and pd.notna(available)
                and start <= date and available <= date
                and (pd.isna(end) or end >= date)
            )
            if eligible:
                rows.append({
                    "decision_time": date, "ticker": row.ticker, "eligible": True,
                    "reason": reason, "universe_definition": definition,
                    "index_code": getattr(row, "index_code", None),
                    "data_class": getattr(row, "data_class", "real"),
                    "effective_from": start, "effective_to": end,
                    "record_available_at": available, "source": row.source,
                    "source_url": getattr(row, "source_url", None),
                    "fetched_at": getattr(row, "fetched_at", None),
                    "raw_checksum": getattr(row, "raw_checksum", None),
                    "history_method": getattr(row, "history_method", None),
                })
    universe = pd.DataFrame(rows)
    if universe.empty:
        raise ValueError("The declared point-in-time universe produced no eligible snapshots.")
    paths.curated.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(paths.curated / "universe_monthly.parquet", index=False)
    (paths.curated / "universe_contract.json").write_text(json.dumps({
        "definition": definition, "index_code": index_code, "rebalance": rebalance,
        "snapshot_rows": len(universe), "tickers": int(universe["ticker"].nunique()),
        "start": str(pd.to_datetime(universe["decision_time"]).min().date()),
        "end": str(pd.to_datetime(universe["decision_time"]).max().date()),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    return universe


def leakage_audit(paths: Paths) -> dict:
    prices = pd.read_parquet(paths.normalized / "prices.parquet")
    master = pd.read_parquet(paths.normalized / "security_master.parquet")
    actions_path = paths.normalized / "corporate_actions.parquet"
    actions = pd.read_parquet(actions_path) if actions_path.exists() else pd.DataFrame()
    membership_path = paths.normalized / "index_membership.parquet"
    membership = pd.read_parquet(membership_path) if membership_path.exists() else pd.DataFrame()
    universe_metadata_path = paths.curated / "universe_contract.json"
    universe_metadata = (
        json.loads(universe_metadata_path.read_text(encoding="utf-8"))
        if universe_metadata_path.exists() else {}
    )
    universe_definition = universe_metadata.get("definition")
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
    adjustment_contract_path = paths.normalized / "price_adjustment_contract.json"
    adjustment_contract = (
        json.loads(adjustment_contract_path.read_text(encoding="utf-8"))
        if adjustment_contract_path.exists() else {}
    )
    adjustment_contract_required = {
        "adjustment_policy", "source", "source_url", "methodology", "certified_by",
        "certified_at", "output_price_dataset_sha256",
    }
    adjustment_contract_valid = bool(
        adjustment_contract
        and adjustment_contract_required <= set(adjustment_contract)
        and adjustment_contract.get("adjustment_policy") in adjustment_policies
        and adjustment_contract.get("output_price_dataset_sha256")
        == sha256_file(paths.normalized / "prices.parquet")
    )
    adjustment_verified = bool(
        adjustment_policies
        and adjustment_policies <= VERIFIED_ADJUSTMENT_POLICIES
        and adjustment_contract_valid
    )
    actions_required = "unadjusted_with_verified_actions_join" in adjustment_policies
    declared_universe_valid = universe_definition in UNIVERSE_DEFINITIONS
    membership_required = universe_definition == "index_membership"
    checks = {
        "universe_definition_declared": declared_universe_valid,
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
        "historical_membership_events_available_when_required": (
            membership_contract if membership_required else True
        ),
        "real_prices_only": not prices["data_class"].astype(str).eq("fixture").any(),
        "auxiliary_tables_fixture_free": not fixture_auxiliary,
        "corporate_actions_point_in_time_available_when_required": (
            actions_contract if actions_required else True
        ),
        "price_adjustment_policy_verified": adjustment_verified,
        "price_adjustment_contract_matches_dataset": adjustment_contract_valid,
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
        "universe_definition": universe_definition,
        "index_code": universe_metadata.get("index_code"),
        "adjustment_policies": sorted(adjustment_policies),
        "price_adjustment_contract": adjustment_contract,
        "corporate_actions_required": actions_required,
        "note": (
            "Historical universe/membership, corporate actions and adjustment policy are core "
            "research contracts. Optional PIT features are excluded when unavailable. Fixture "
            "auxiliary tables block a real-data run."
        ),
    }
    (paths.reports / "leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
