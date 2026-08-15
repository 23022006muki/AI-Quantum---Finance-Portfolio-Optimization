from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLUMNS = [
    "date", "ticker", "security_id", "open", "high", "low", "close", "adjusted_close", "volume",
    "trading_value", "source", "source_url", "fetched_at", "available_at",
    "raw_checksum", "parser_version", "data_class", "adjustment_policy",
]

PROVENANCE_COLUMNS = {"source", "source_url", "fetched_at", "raw_checksum"}
HISTORICAL_UNIVERSE_METHODS = {
    "official_event_history",
    "exchange_listing_history",
    "verified_membership_history",
    "verified_provider_history",
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

    @property
    def staging(self) -> Path:
        return self.root / "outputs" / "staging"

    def ensure(self) -> None:
        for p in (self.raw, self.normalized, self.curated, self.reports, self.staging):
            p.mkdir(parents=True, exist_ok=True)


def create_staging_run(paths: Paths, source: str) -> Path:
    """Create a versioned collection directory without touching normalized data."""
    paths.ensure()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = paths.staging / f"{stamp}-{source}-{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def build_complete_case_workspace(
    source_paths: Paths,
    start: str,
    end: str,
    minimum_total_observations: int = 40,
    maximum_calendar_gap_days: int = 5,
    forced_excluded_tickers: list[str] | None = None,
) -> tuple[Path, dict]:
    """Create an isolated, analysis-ready real-data workspace.

    The function never changes the canonical normalized panel. It retains only rows
    with complete OHLCV/provenance fields and securities with enough observations in
    the requested interval. The resulting security master is restricted to those
    observed securities so that downstream universe construction cannot reintroduce
    symbols with no usable price history.

    This is intentionally an *exploratory complete-case* dataset. Selecting securities
    using full-period data availability can introduce coverage/survivorship selection
    bias, and an unverified corporate-action adjustment policy remains unverified.
    """
    if minimum_total_observations < 1:
        raise ValueError("minimum_total_observations must be positive")
    if maximum_calendar_gap_days < 0:
        raise ValueError("maximum_calendar_gap_days must not be negative")
    prices_path = source_paths.normalized / "prices.parquet"
    master_path = source_paths.normalized / "security_master.parquet"
    if not prices_path.exists() or not master_path.exists():
        raise FileNotFoundError(
            "Complete-case construction requires normalized prices.parquet and "
            "security_master.parquet."
        )

    prices = pd.read_parquet(prices_path).copy()
    master = pd.read_parquet(master_path).copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["available_at"] = pd.to_datetime(prices["available_at"], errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    forced_excluded = {
        str(ticker).upper().strip() for ticker in (forced_excluded_tickers or [])
    }
    if start_ts > end_ts:
        raise ValueError("start must not be after end")
    prices = prices[prices["date"].between(start_ts, end_ts)].copy()
    if prices.get("data_class", pd.Series(dtype=str)).astype(str).eq("fixture").any():
        raise ValueError("Complete-case real-data construction refuses fixture observations.")

    required_non_null = [
        "date", "ticker", "security_id", "open", "high", "low", "close",
        "adjusted_close", "volume", "trading_value", "source", "source_url",
        "fetched_at", "available_at", "raw_checksum", "parser_version", "data_class",
    ]
    missing_columns = sorted(set(required_non_null) - set(prices.columns))
    if missing_columns:
        raise ValueError(f"Price panel is missing complete-case fields: {missing_columns}")
    numeric = ["open", "high", "low", "close", "adjusted_close", "volume", "trading_value"]
    complete = prices[required_non_null].notna().all(axis=1)
    complete &= np.isfinite(prices[numeric].astype(float)).all(axis=1)
    complete &= prices[["open", "high", "low", "close", "adjusted_close"]].gt(0).all(axis=1)
    complete &= prices[["volume", "trading_value"]].ge(0).all(axis=1)
    complete &= prices["high"].ge(prices[["open", "close", "low"]].max(axis=1))
    complete &= prices["low"].le(prices[["open", "close", "high"]].min(axis=1))
    complete &= prices["available_at"].ge(prices["date"])
    complete &= ~prices.duplicated(["ticker", "date"], keep=False)
    valid_prices = prices.loc[complete].sort_values(["ticker", "date"]).copy()

    master["ticker"] = master["ticker"].astype(str)
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], errors="coerce")
    relevant = master[
        master["listing_date"].le(end_ts)
        & (master["delisting_date"].isna() | master["delisting_date"].ge(start_ts))
    ].copy()
    calendar = pd.DatetimeIndex(sorted(valid_prices["date"].dropna().unique()))
    calendar_position = {date: position for position, date in enumerate(calendar)}
    counts = valid_prices.groupby("ticker")["date"].nunique()
    coverage_diagnostics: dict[str, dict] = {}
    eligible_tickers: set[str] = set()
    for row in relevant.itertuples():
        ticker = str(row.ticker)
        ticker_dates = pd.DatetimeIndex(sorted(
            valid_prices.loc[valid_prices["ticker"].astype(str).eq(ticker), "date"].unique()
        ))
        listing_start = max(start_ts, row.listing_date)
        listing_end = (
            min(end_ts, row.delisting_date) if pd.notna(row.delisting_date) else end_ts
        )
        expected = calendar[(calendar >= listing_start) & (calendar <= listing_end)]
        if len(ticker_dates):
            leading_gap = int((expected < ticker_dates.min()).sum())
            trailing_gap = int((expected > ticker_dates.max()).sum())
            positions = np.asarray([
                calendar_position[date] for date in ticker_dates if date in calendar_position
            ])
            maximum_internal_gap = int(np.diff(positions).max() - 1) if len(positions) > 1 else 0
        else:
            leading_gap = trailing_gap = maximum_internal_gap = int(len(expected))
        usable_rows = int(counts.get(ticker, 0))
        coverage_diagnostics[ticker] = {
            "complete_rows": usable_rows,
            "leading_calendar_gap_days": leading_gap,
            "trailing_calendar_gap_days": trailing_gap,
            "maximum_internal_calendar_gap_days": maximum_internal_gap,
        }
        if (
            ticker not in forced_excluded
            and usable_rows >= minimum_total_observations
            and leading_gap <= maximum_calendar_gap_days
            and trailing_gap <= maximum_calendar_gap_days
            and maximum_internal_gap <= maximum_calendar_gap_days
        ):
            eligible_tickers.add(ticker)

    valid_prices = valid_prices[valid_prices["ticker"].astype(str).isin(eligible_tickers)].copy()
    if valid_prices.empty:
        raise ValueError("No securities satisfy the complete-case eligibility criteria.")
    restricted_master = relevant[relevant["ticker"].isin(eligible_tickers)].copy()
    price_only = eligible_tickers - set(restricted_master["ticker"])
    if price_only:
        raise ValueError(
            "Usable prices have no verified security-master identity: "
            + ", ".join(sorted(price_only))
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(
        (sha256_file(prices_path) + str(start_ts.date()) + str(end_ts.date())
         + str(minimum_total_observations) + str(maximum_calendar_gap_days)
         + "|".join(sorted(forced_excluded))).encode("utf-8")
    ).hexdigest()[:10]
    workspace = source_paths.root / "outputs" / "complete_case_workspaces" / f"{stamp}-{digest}"
    target = Paths(workspace)
    target.ensure()
    valid_prices.to_parquet(target.normalized / "prices.parquet", index=False)
    restricted_master.to_parquet(target.normalized / "security_master.parquet", index=False)

    actions_path = source_paths.normalized / "corporate_actions.parquet"
    if actions_path.exists():
        shutil.copy2(actions_path, target.normalized / "corporate_actions.parquet")

    excluded_rows = []
    observed_counts = prices.groupby("ticker")["date"].nunique()
    for row in relevant.itertuples():
        ticker = str(row.ticker)
        if ticker in eligible_tickers:
            continue
        observed = int(observed_counts.get(ticker, 0))
        diagnostic = coverage_diagnostics.get(ticker, {})
        usable = int(diagnostic.get("complete_rows", 0))
        if ticker in forced_excluded:
            reason = "forced_quality_exclusion"
        elif observed == 0:
            reason = "no_price_observations_in_requested_period"
        elif usable < minimum_total_observations:
            reason = "fewer_than_minimum_complete_observations"
        elif diagnostic.get("leading_calendar_gap_days", 0) > maximum_calendar_gap_days:
            reason = "unexplained_leading_calendar_gap"
        elif diagnostic.get("trailing_calendar_gap_days", 0) > maximum_calendar_gap_days:
            reason = "unexplained_trailing_calendar_gap"
        elif diagnostic.get("maximum_internal_calendar_gap_days", 0) > maximum_calendar_gap_days:
            reason = "unexplained_internal_calendar_gap"
        else:
            reason = "failed_complete_case_contract"
        excluded_rows.append({
            "ticker": ticker,
            "observed_rows": observed,
            "complete_rows": usable,
            "minimum_required": minimum_total_observations,
            "leading_calendar_gap_days": diagnostic.get("leading_calendar_gap_days"),
            "trailing_calendar_gap_days": diagnostic.get("trailing_calendar_gap_days"),
            "maximum_internal_calendar_gap_days": diagnostic.get(
                "maximum_internal_calendar_gap_days"
            ),
            "maximum_allowed_calendar_gap_days": maximum_calendar_gap_days,
            "reason": reason,
        })
    exclusions = pd.DataFrame(excluded_rows, columns=[
        "ticker", "observed_rows", "complete_rows", "minimum_required",
        "leading_calendar_gap_days", "trailing_calendar_gap_days",
        "maximum_internal_calendar_gap_days", "maximum_allowed_calendar_gap_days", "reason",
    ]).sort_values("ticker")
    exclusions.to_csv(target.reports / "complete_case_exclusions.csv", index=False)

    manifest = {
        "dataset_kind": "exploratory_complete_case_real_prices",
        "label": "EXPLORATORY ONLY - RESTRICTED OBSERVED HOSE PANEL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_start": str(start_ts.date()),
        "requested_end": str(end_ts.date()),
        "minimum_total_observations": int(minimum_total_observations),
        "maximum_calendar_gap_days": int(maximum_calendar_gap_days),
        "forced_quality_exclusions": sorted(forced_excluded),
        "source_price_dataset": str(prices_path),
        "source_price_dataset_sha256": sha256_file(prices_path),
        "source_security_master": str(master_path),
        "source_security_master_sha256": sha256_file(master_path),
        "records_before_row_filter": int(len(prices)),
        "records_failing_complete_row_contract": int((~complete).sum()),
        "records_retained": int(len(valid_prices)),
        "tickers_observed": int(prices["ticker"].nunique()),
        "tickers_retained": int(valid_prices["ticker"].nunique()),
        "relevant_master_tickers": int(relevant["ticker"].nunique()),
        "relevant_tickers_excluded": int(len(exclusions)),
        "excluded_tickers": exclusions["ticker"].tolist(),
        "price_dataset_sha256": sha256_file(target.normalized / "prices.parquet"),
        "security_master_sha256": sha256_file(target.normalized / "security_master.parquet"),
        "selection_rule": (
            "complete OHLCV/provenance row contract and at least "
            f"{minimum_total_observations} observations with no unexplained leading, "
            f"trailing or internal calendar gap above {maximum_calendar_gap_days} sessions"
        ),
        "limitations": [
            "full_period_availability_filter_can_create_coverage_or_survivorship_selection_bias",
            "corporate_action_adjustment_policy_is_not_verified",
            "no_verified_total_return_benchmark",
            "optional_fundamental_macro_and_foreign_flow_features_are_not_used",
            "results_must_not_be_labeled_confirmatory_or_full_hose_research",
        ],
    }
    (target.raw / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (target.reports / "COMPLETE_CASE_DATASET.md").write_text(
        "# Exploratory complete-case HOSE dataset\n\n"
        f"- Period: `{manifest['requested_start']}` to `{manifest['requested_end']}`\n"
        f"- Retained: **{manifest['tickers_retained']} tickers / "
        f"{manifest['records_retained']:,} rows**\n"
        f"- Excluded relevant tickers: **{manifest['relevant_tickers_excluded']}**\n"
        f"- Minimum observations: **{minimum_total_observations}**\n\n"
        f"- Maximum unexplained calendar gap: **{maximum_calendar_gap_days} sessions**\n\n"
        "This dataset is suitable for an exploratory end-to-end run. It is not a "
        "replacement for a verified point-in-time, corporate-action-adjusted, "
        "full-HOSE research panel. See `complete_case_exclusions.csv` and the "
        "limitations in `outputs/raw/manifest.json`.\n",
        encoding="utf-8",
    )
    return workspace, manifest


def promote_staged_file(paths: Paths, staged_file: Path, target_name: str) -> dict:
    """Atomically promote one validated file and retain a recoverable previous copy."""
    if not staged_file.is_file():
        raise FileNotFoundError(staged_file)
    paths.normalized.mkdir(parents=True, exist_ok=True)
    target = paths.normalized / target_name
    archive = paths.root / "outputs" / "archive" / datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = None
    if target.exists():
        archive.mkdir(parents=True, exist_ok=True)
        backup = archive / target.name
        shutil.copy2(target, backup)
    temporary = target.with_name(f".{target.name}.promoting")
    shutil.copy2(staged_file, temporary)
    temporary.replace(target)
    return {
        "target": str(target), "sha256": sha256_file(target),
        "backup": str(backup) if backup else None,
    }


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
            "security_id": f"FIXTURE:{ticker}",
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
            "security_id": f"FIXTURE:{ticker}", "ticker": ticker,
            "company_name": f"Fixture Company {ticker}",
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
    df["security_id"] = df.get("security_id", df["ticker"].map(lambda x: f"TICKER:{x}"))
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
    required_price_columns = set(PRICE_COLUMNS)
    missing_price_columns = sorted(required_price_columns - set(prices.columns))
    if missing_price_columns:
        issues.append({
            "severity": "error", "check": "required_price_schema",
            "count": len(missing_price_columns), "columns": missing_price_columns,
        })
    if "security_id" not in prices:
        # Continue the diagnostic pass, but do not silently manufacture research identity.
        prices["security_id"] = pd.NA
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
            ["verified_corporate_action", "verified_vendor_adjustment"],
            default="unresolved",
        )
        ledger_path = paths.reports / "outlier_resolution_ledger.csv"
        if ledger_path.exists():
            ledger = pd.read_csv(ledger_path)
            ledger["date"] = pd.to_datetime(ledger.get("date"), errors="coerce")
            allowed = {
                "verified_corporate_action", "verified_cross_source_correction",
                "verified_vendor_adjustment", "unresolved", "genuine_market_move",
            }
            required_ledger = {"ticker", "date", "resolution", "reviewer_status", "source_url"}
            if required_ledger <= set(ledger.columns):
                ledger = ledger[
                    ledger["resolution"].isin(allowed)
                    & ledger["reviewer_status"].astype(str).isin({"verified", "rejected", "pending"})
                ].drop_duplicates(["ticker", "date"], keep="last")
                outlier_rows = outlier_rows.merge(
                    ledger[["ticker", "date", "resolution", "reviewer_status", "source_url"]]
                    .rename(columns={
                        "resolution": "ledger_resolution", "source_url": "ledger_source_url",
                    }), on=["ticker", "date"], how="left",
                )
                verified_ledger = (
                    outlier_rows["reviewer_status"].eq("verified")
                    & outlier_rows["ledger_resolution"].isin(allowed - {"unresolved"})
                    & outlier_rows["ledger_source_url"].astype(str).str.startswith(("http://", "https://"))
                )
                outlier_rows.loc[verified_ledger, "resolution"] = outlier_rows.loc[
                    verified_ledger, "ledger_resolution"
                ]
        unresolved = int(outlier_rows["resolution"].eq("unresolved").sum())
        issues.append({
            "severity": "error" if unresolved else "warning",
            "check": "unresolved_return_outlier" if unresolved else "return_outlier_reviewed",
            "count": unresolved if unresolved else len(outlier_rows),
        })
    outlier_columns = [
        "date", "ticker", "close", "adjusted_close", "return_1d",
        "adjustment_policy", "corporate_action_match",
        "adjustment_contract_verified", "resolution", "reviewer_status",
        "ledger_source_url", "source", "source_url",
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
    max_assets: int | None = None,
    liquidity_lookback_days: int = 60,
    minimum_observations: int = 40,
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
    prices["available_at"] = pd.to_datetime(prices["available_at"], errors="coerce")
    if "security_id" not in prices:
        prices["security_id"] = prices["ticker"]
    if "security_id" not in master:
        master["security_id"] = master["ticker"]
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
    audit_rows = []
    for date in anchors:
        eligible_records = []
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
                eligible_records.append({
                    "decision_time": date, "ticker": row.ticker, "eligible": True,
                    "security_id": getattr(row, "security_id", row.ticker),
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
            else:
                audit_rows.append({
                    "decision_time": date, "ticker": row.ticker,
                    "security_id": getattr(row, "security_id", row.ticker),
                    "included": False, "reason": "outside_verified_listing_interval_or_not_yet_available",
                    "trailing_observations": 0, "trailing_liquidity": np.nan,
                })
        eligible_frame = pd.DataFrame(eligible_records)
        if eligible_frame.empty:
            continue
        start_lookback = date - pd.offsets.BDay(max(1, liquidity_lookback_days))
        history = prices[
            (prices["date"] <= date)
            & (prices["date"] >= start_lookback)
            & (prices["available_at"] <= date)
            & prices["ticker"].isin(eligible_frame["ticker"])
        ]
        liquidity = history.groupby("ticker").agg(
            trailing_observations=("date", "nunique"),
            trailing_liquidity=("trading_value", "mean"),
        )
        eligible_frame = eligible_frame.merge(liquidity, on="ticker", how="left")
        eligible_frame["trailing_observations"] = eligible_frame["trailing_observations"].fillna(0).astype(int)
        enough_history = eligible_frame["trailing_observations"] >= minimum_observations
        ranked = eligible_frame[enough_history].sort_values(
            ["trailing_liquidity", "ticker"], ascending=[False, True], na_position="last"
        )
        if max_assets is not None:
            ranked = ranked.head(int(max_assets))
        selected = set(ranked["ticker"])
        for item in eligible_frame.itertuples():
            included = item.ticker in selected
            if included:
                exclusion_reason = "selected_by_trailing_liquidity_point_in_time"
            elif item.trailing_observations < minimum_observations:
                exclusion_reason = "insufficient_trailing_observations"
            else:
                exclusion_reason = "outside_dynamic_top_n_liquidity"
            audit_rows.append({
                "decision_time": date, "ticker": item.ticker,
                "security_id": item.security_id, "included": included,
                "reason": exclusion_reason,
                "trailing_observations": item.trailing_observations,
                "trailing_liquidity": item.trailing_liquidity,
            })
        rows.extend(ranked.to_dict("records"))
    universe = pd.DataFrame(rows)
    if universe.empty:
        raise ValueError("The declared point-in-time universe produced no eligible snapshots.")
    paths.curated.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(paths.curated / "universe_monthly.parquet", index=False)
    eligibility_audit = pd.DataFrame(audit_rows)
    eligibility_audit.to_parquet(paths.curated / "universe_eligibility_audit.parquet", index=False)
    eligibility_audit.to_csv(paths.curated / "universe_eligibility_audit.csv", index=False)
    (paths.curated / "universe_contract.json").write_text(json.dumps({
        "definition": definition, "index_code": index_code, "rebalance": rebalance,
        "max_assets": max_assets, "liquidity_lookback_days": liquidity_lookback_days,
        "minimum_observations": minimum_observations,
        "selection_information_cutoff": "price.available_at <= decision_time",
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
        {"security_id", "listing_date", "delisting_date", "effective_from", "effective_to", "available_at",
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
        and {"ticker", "security_id", "announcement_date", "effective_date", "available_at"}
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
    price_start = pd.to_datetime(prices["date"], errors="coerce").min()
    price_end = pd.to_datetime(prices["date"], errors="coerce").max()
    master_listing = pd.to_datetime(master.get("listing_date"), errors="coerce")
    master_delisting = pd.to_datetime(master.get("delisting_date"), errors="coerce")
    relevant_master = master[
        master_listing.le(price_end)
        & (master_delisting.isna() | master_delisting.ge(price_start))
    ]
    observed_tickers = set(prices["ticker"].dropna().astype(str))
    required_tickers = set(relevant_master["ticker"].dropna().astype(str))
    missing_universe_prices = sorted(required_tickers - observed_tickers)
    master_security_ids = set(master.get("security_id", pd.Series(dtype=str)).dropna().astype(str))
    price_security_ids = set(prices.get("security_id", pd.Series(dtype=str)).dropna().astype(str))
    unmatched_price_security_ids = sorted(price_security_ids - master_security_ids)
    checks = {
        "universe_definition_declared": declared_universe_valid,
        "historical_universe_contract": master_has_contract,
        "historical_universe_source_trusted": trusted_history,
        "historical_universe_fixture_free": not master.get(
            "data_class", pd.Series(dtype=str)
        ).astype(str).eq("fixture").any(),
        "historical_universe_times_valid": master_times_valid,
        "historical_universe_price_coverage_complete": not missing_universe_prices,
        "price_security_identity_matches_master": not unmatched_price_security_ids,
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
        "historical_universe_price_coverage": {
            "required_tickers": len(required_tickers),
            "observed_tickers": len(required_tickers & observed_tickers),
            "missing_count": len(missing_universe_prices),
            "missing_tickers": missing_universe_prices,
        },
        "unmatched_price_security_ids": unmatched_price_security_ids,
        "note": (
            "Historical universe/membership, corporate actions and adjustment policy are core "
            "research contracts. Optional PIT features are excluded when unavailable. Fixture "
            "auxiliary tables block a real-data run."
        ),
    }
    (paths.reports / "leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
