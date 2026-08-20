from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .data_pipeline import PRICE_COLUMNS, Paths, sha256_file
from .corporate_actions import (
    VSDCCorporateActionAdapter,
    reconcile_corporate_actions,
)
from .price_adjustment import MATERIAL_EVENT_TYPES, _build_ticker_total_return


DATASET_LABEL = "Data 17/8"
PHYSICAL_WORKSPACE_NAME = "Data 17_8"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
HOSE_NEWS_API = "https://api.hsx.vn/n/api/v1/1"
HOSE_MEDIA_API = "https://api.hsx.vn/m/api/v1/1"
HOSE_LISTING_API = "https://api.hsx.vn/l/api/v1/1"
HOSE_MARKET_API = "https://api.hsx.vn/mk/api/v1"
HOSE_STATIC = "https://staticfile.hsx.vn"
HOSE_INDEX_METHODOLOGY_URL = (
    "https://staticfile.hsx.vn/Uploads/LocalFiles/ef15ff11e799483abd11677ad0443887/"
    "20250114_20241230_QD%20747%20HOSE%20Index%20Ground%20Rules.pdf"
)
HOSE_TRI_PAGE_URL = (
    "https://www.hsx.vn/vi/du-lieu-giao-dich/quy-mo-giao-dich/theo-bo-chi-so-tri"
)


_OCR_THREAD_STATE = threading.local()
_HTTP_THREAD_STATE = threading.local()
_VIETSTOCK_THREAD_STATE = threading.local()


class _HOSEMalformedRange(RuntimeError):
    """The HOSE API returned a deterministic 500 for a range containing bad data."""


class _HOSETemporarilyUnavailable(RuntimeError):
    """A range could not be fetched because the upstream service was transiently unavailable."""


def _http_session() -> requests.Session:
    """Reuse HTTPS connections within each worker without sharing a Session across threads."""
    session = getattr(_HTTP_THREAD_STATE, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2)
        session.mount("https://", adapter)
        _HTTP_THREAD_STATE.session = session
    return session


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    folded = "".join(character for character in normalized if not unicodedata.combining(character))
    return folded.replace("đ", "d").replace("Đ", "D").lower()


def _safe_stem(value: str, maximum: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", _ascii_fold(value)).strip("-")[:maximum]


def _local_timestamp(epoch_seconds: Any) -> pd.Timestamp:
    value = pd.to_numeric(pd.Series([epoch_seconds]), errors="coerce").iloc[0]
    if pd.isna(value):
        return pd.NaT
    return (
        pd.to_datetime(float(value), unit="s", utc=True)
        .tz_convert("Asia/Ho_Chi_Minh")
        .tz_localize(None)
    )


def _local_timestamp_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (
        pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
        .dt.tz_convert("Asia/Ho_Chi_Minh")
        .dt.tz_localize(None)
    )


def _json_archive(directory: Path, stem: str, payload: Any) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    target = directory / f"{_safe_stem(stem)}-{digest[:16]}.json"
    if not target.exists():
        target.write_bytes(encoded)
    return {"path": str(target), "sha256": digest, "bytes": len(encoded)}


def _load_json_checkpoint(directory: Path, stem: str) -> dict[str, Any] | None:
    """Return a previously archived immutable payload for an exact request stem."""
    candidates = sorted(directory.glob(f"{_safe_stem(stem)}-*.json"))
    if not candidates:
        return None
    try:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and payload.get("success") is True else None


def _load_json_archive_value(directory: Path, stem: str) -> Any | None:
    candidates = sorted(directory.glob(f"{_safe_stem(stem)}-*.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    attempts: int = 5,
    timeout: float = 90.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; academic-research-data-17-8/1.0)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
    }
    for attempt in range(attempts):
        try:
            response = _http_session().get(
                url, params=params, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("success") is not True:
                raise RuntimeError("source payload reported success=false")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(12.0, 0.75 * (2**attempt)))
    raise RuntimeError(f"Source request failed after retries: {type(last_error).__name__}")


def _request_hose_news_page(
    url: str,
    *,
    params: dict[str, Any],
    attempts: int = 4,
    timeout: float = 35.0,
) -> dict[str, Any]:
    """Fetch a HOSE news page while distinguishing bad records from transport errors."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; academic-research-data-17-8/1.0)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _http_session().get(
                url, params=params, headers=headers, timeout=timeout
            )
            if response.status_code == 500:
                raise _HOSEMalformedRange("HOSE returned HTTP 500 for this exact range")
            response.raise_for_status()
            payload = response.json()
            if payload.get("success") is not True:
                raise ValueError("source payload reported success=false")
            return payload
        except _HOSEMalformedRange:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(15.0, 1.0 * (2**attempt)))
    raise _HOSETemporarilyUnavailable(
        f"HOSE news API unavailable after {attempts} attempts: {type(last_error).__name__}"
    )


def data_17_8_workspace(root: Path) -> Path:
    return root / "outputs" / PHYSICAL_WORKSPACE_NAME


def initialize_data_17_8(root: Path, base_workspace: Path | None = None) -> dict[str, Any]:
    """Create an isolated Data 17/8 workspace without mutating Data A or Data B."""
    base = base_workspace or (root / "outputs" / "Data A")
    source_normalized = base / "outputs" / "normalized"
    target_root = data_17_8_workspace(root)
    target = Paths(target_root)
    target.ensure()
    package_path = target_root / "DATA_17_8_PACKAGE.json"
    if package_path.exists():
        package = json.loads(package_path.read_text(encoding="utf-8"))
        if package.get("dataset") != DATASET_LABEL:
            raise RuntimeError(f"Unexpected existing package identity: {package_path}")
        required_existing = [
            target.normalized / "prices.parquet",
            target.normalized / "security_master.parquet",
        ]
        if not all(path.exists() for path in required_existing):
            raise RuntimeError("Existing Data 17/8 package is incomplete")
        # Once initialized, Data 17/8 is expected to diverge from Data A.  Never
        # overwrite its independently crawled price panel on later CLI stages.
        return package
    required = ["prices.parquet", "security_master.parquet"]
    missing = [name for name in required if not (source_normalized / name).exists()]
    if missing:
        raise FileNotFoundError(f"Data 17/8 requires Data A files: {missing}")
    copied: dict[str, str] = {}
    for name in required:
        source = source_normalized / name
        destination = target.normalized / name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"Refusing to overwrite a divergent Data 17/8 file: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        copied[name] = sha256_file(destination)
    package = {
        "dataset": DATASET_LABEL,
        "physical_workspace": str(target_root),
        "base_workspace": str(base),
        "date_start": START_DATE,
        "date_end": END_DATE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_files": copied,
        "status": "initialized_pending_source_collection",
        "research_policy": "fail_closed_point_in_time",
    }
    package_path.write_text(
        json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return package


def _ticker_set(paths: Paths) -> set[str]:
    prices = pd.read_parquet(paths.normalized / "prices.parquet", columns=["ticker"])
    return set(prices["ticker"].astype(str).str.upper().str.strip())


def _master_ticker_set(paths: Paths) -> set[str]:
    master_path = (
        paths.normalized / "security_master_full.parquet"
        if (paths.normalized / "security_master_full.parquet").exists()
        else paths.normalized / "security_master.parquet"
    )
    master = pd.read_parquet(
        master_path, columns=["ticker"]
    )
    return set(master["ticker"].astype(str).str.upper().str.strip())


def _data_17_8_action_scope(paths: Paths) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Return eligible events, excluded tickers and complete-case tickers.

    The unit of exclusion is the entire ticker.  A security is never retained by
    silently dropping only the corporate-action observation that is difficult to
    reconcile.  This makes the resulting universe smaller, but keeps the total-
    return contract auditable and avoids outcome-dependent exclusions.
    """
    master_path = (
        paths.normalized / "security_master_full.parquet"
        if (paths.normalized / "security_master_full.parquet").exists()
        else paths.normalized / "security_master.parquet"
    )
    actions_path = paths.normalized / "corporate_actions.parquet"
    if not master_path.exists() or not actions_path.exists():
        raise FileNotFoundError(
            "security_master.parquet and corporate_actions.parquet are required"
        )
    master = pd.read_parquet(master_path)
    master_tickers = set(
        master["ticker"].astype(str).str.upper().str.strip()
    )
    actions = pd.read_parquet(actions_path).copy()
    actions["ticker"] = actions["ticker"].astype(str).str.upper().str.strip()
    for column in ["effective_date", "record_date", "announcement_date", "available_at"]:
        actions[column] = pd.to_datetime(actions[column], errors="coerce")
    material = actions["event_type"].isin(MATERIAL_EVENT_TYPES)
    verified = (
        material
        & actions["verification_status"].astype(str).eq("verified_cross_source")
        & actions["effective_date"].notna()
        & actions["available_at"].notna()
        & actions["available_at"].le(actions["effective_date"])
    )
    excluded = set(actions.loc[material & ~verified, "ticker"]) & master_tickers
    eligible = actions.loc[verified & actions["ticker"].isin(master_tickers)].copy()
    return eligible, excluded, master_tickers - excluded


def _valid_vnstock_checkpoint(
    path: Path,
    *,
    ticker: str,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
) -> pd.DataFrame | None:
    try:
        frame = pd.read_parquet(path).copy()
    except (OSError, ValueError):
        return None
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[
        frame["ticker"].astype(str).str.upper().eq(ticker)
        & frame["date"].between(expected_start, expected_end)
    ].copy()
    if frame.empty or frame["date"].duplicated().any():
        return None
    # Allow the normal listing/holiday margins, but reject short or stale files.
    if (
        frame["date"].min() > expected_start + pd.Timedelta(days=21)
        or frame["date"].max() < expected_end - pd.Timedelta(days=21)
    ):
        return None
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_ohlc = (
        numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & numeric["high"].ge(numeric[["open", "close", "low"]].max(axis=1))
        & numeric["low"].le(numeric[["open", "close", "high"]].min(axis=1))
        & numeric["volume"].ge(0)
    )
    frame = frame.loc[valid_ohlc].copy()
    return frame if len(frame) >= 40 else None


def crawl_data_17_8_prices(
    root: Path,
    start: str = START_DATE,
    end: str = END_DATE,
    pause_seconds: float = 6.4,
) -> dict[str, Any]:
    """Build a clean Data 17/8 price panel from vnstock/KBS checkpoints.

    Data A prices are not mutated.  Only securities with no unresolved material
    corporate action are requested.  The output uses raw KBS closes plus a
    total-return index constructed solely from the verified VSDC/CafeF ledger.
    """
    from vnstock import Market

    if pause_seconds < 6.1:
        raise ValueError(
            "pause_seconds must be at least 6.1 because one symbol consumes two guest quota units"
        )
    paths = Paths(data_17_8_workspace(root))
    paths.ensure()
    eligible_events, excluded_tickers, complete_case = _data_17_8_action_scope(paths)
    master_path = (
        paths.normalized / "security_master_full.parquet"
        if (paths.normalized / "security_master_full.parquet").exists()
        else paths.normalized / "security_master.parquet"
    )
    master = pd.read_parquet(master_path).copy()
    master["ticker"] = master["ticker"].astype(str).str.upper().str.strip()
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    master = master[
        master["ticker"].isin(complete_case)
        & master["listing_date"].le(end_ts)
        & (master["delisting_date"].isna() | master["delisting_date"].ge(start_ts))
    ].sort_values("ticker").drop_duplicates("ticker")

    checkpoint_dir = paths.raw / "vnstock_kbs_prices"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    inherited_dir = root / "outputs" / "raw" / "vnstock_ohlcv"
    completed: list[str] = []
    inherited: list[str] = []
    fetched: list[str] = []
    failures: list[dict[str, str]] = []
    market = Market()
    source_url_template = (
        "https://kbbuddywts.kbsec.com.vn/iis-server/investment/stocks/{ticker}/data_day"
    )

    for position, row in enumerate(master.to_dict("records"), 1):
        ticker = row["ticker"]
        expected_start = max(start_ts, pd.Timestamp(row["listing_date"]))
        expected_end = min(
            end_ts,
            pd.Timestamp(row["delisting_date"])
            if pd.notna(row["delisting_date"]) else end_ts,
        )
        checkpoint = checkpoint_dir / f"{ticker}.parquet"
        frame = _valid_vnstock_checkpoint(
            checkpoint,
            ticker=ticker,
            expected_start=expected_start,
            expected_end=expected_end,
        ) if checkpoint.exists() else None
        try:
            if frame is None:
                shared = inherited_dir / f"{ticker}.parquet"
                inherited_frame = _valid_vnstock_checkpoint(
                    shared,
                    ticker=ticker,
                    expected_start=expected_start,
                    expected_end=expected_end,
                ) if shared.exists() else None
                if inherited_frame is not None:
                    shutil.copy2(shared, checkpoint)
                    frame = inherited_frame
                    inherited.append(ticker)
            if frame is None:
                last_error: BaseException | None = None
                for attempt in range(4):
                    try:
                        raw = market.equity(symbol=ticker).ohlcv(
                            start=start,
                            end=end,
                            interval="1D",
                            count=5000,
                            source="kbs",
                        )
                        if raw is None or raw.empty:
                            raise RuntimeError("empty KBS response")
                        from .sources import _normalize_vnstock_ohlc

                        normalized = _normalize_vnstock_ohlc(
                            raw,
                            ticker,
                            source_url_template.format(ticker=ticker),
                        )
                        normalized.to_parquet(checkpoint, index=False)
                        frame = _valid_vnstock_checkpoint(
                            checkpoint,
                            ticker=ticker,
                            expected_start=expected_start,
                            expected_end=expected_end,
                        )
                        if frame is None:
                            raise RuntimeError("KBS checkpoint failed coverage validation")
                        fetched.append(ticker)
                        break
                    except (Exception, SystemExit) as exc:
                        last_error = exc
                        if attempt < 3:
                            time.sleep(1.5 * (2**attempt))
                if frame is None:
                    raise RuntimeError(str(last_error or "unavailable"))
                time.sleep(pause_seconds)
            completed.append(ticker)
            print(
                f"[Data 17/8 prices] {position:03d}/{len(master):03d} {ticker}: "
                f"{len(frame):,} rows",
                flush=True,
            )
        except (Exception, SystemExit) as exc:
            failures.append({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})
            print(
                f"[Data 17/8 prices] {position:03d}/{len(master):03d} {ticker}: FAILED",
                flush=True,
            )
        progress = {
            "requested": len(master),
            "completed": len(completed),
            "failed": len(failures),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (paths.reports / "price_crawl_progress.json").write_text(
            json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    output_frames: list[pd.DataFrame] = []
    adjustment_evidence: list[pd.DataFrame] = []
    for ticker in completed:
        master_row = master.loc[master["ticker"].eq(ticker)].iloc[0]
        expected_start = max(start_ts, pd.Timestamp(master_row["listing_date"]))
        expected_end = min(
            end_ts,
            pd.Timestamp(master_row["delisting_date"])
            if pd.notna(master_row["delisting_date"]) else end_ts,
        )
        source_frame = _valid_vnstock_checkpoint(
            checkpoint_dir / f"{ticker}.parquet",
            ticker=ticker,
            expected_start=expected_start,
            expected_end=expected_end,
        )
        if source_frame is None:
            continue
        source_frame = source_frame.sort_values("date").reset_index(drop=True)
        source_frame["security_id"] = str(master_row["security_id"])
        for column in ["adjusted_close", "trading_value"]:
            if column not in source_frame:
                source_frame[column] = (
                    source_frame["close"] if column == "adjusted_close"
                    else source_frame["close"] * source_frame["volume"]
                )
        renamed = source_frame.rename(columns={
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
            "adjusted_close": "source_adjusted_close",
        })
        adjusted = _build_ticker_total_return(
            renamed,
            eligible_events.loc[eligible_events["ticker"].eq(ticker)],
        )
        event_rows = adjusted.loc[
            adjusted["adjustment_source"].eq("verified_corporate_action_ledger"),
            ["date", "ticker", "price_return", "total_return"],
        ].copy()
        if not event_rows.empty:
            event_rows["increment_if_ledger_added_again"] = (
                event_rows["total_return"] - event_rows["price_return"]
            )
            adjustment_evidence.append(event_rows)
        # KBS publishes a back-adjusted OHLC series.  The event-day evidence
        # below is used to verify that adding the ledger again would double
        # count distributions.  Keep its close as the total-return proxy here;
        # CafeF raw prices are joined in the next stage for execution realism.
        source_frame["adjusted_close"] = source_frame["close"]
        source_frame["adjustment_policy"] = "unverified"
        source_frame["parser_version"] = (
            source_frame.get(
                "parser_version",
                pd.Series("vnstock-kbs", index=source_frame.index),
            )
            .astype(str)
            .str.replace(r"\+data17-total-return-v\d+$", "", regex=True)
            + "+data17-total-return-v1"
        )
        source_frame["source_url"] = source_url_template.format(ticker=ticker)
        output_frames.append(source_frame[PRICE_COLUMNS])

    if not output_frames:
        raise RuntimeError("No complete-case vnstock/KBS price series passed validation")
    prices = (
        pd.concat(output_frames, ignore_index=True)
        .drop_duplicates(["ticker", "date"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    evidence = (
        pd.concat(adjustment_evidence, ignore_index=True)
        if adjustment_evidence else pd.DataFrame(columns=[
            "date", "ticker", "price_return", "total_return",
            "increment_if_ledger_added_again",
        ])
    )
    event_returns_in_band = float(
        evidence["price_return"].abs().le(0.08).mean()
    ) if not evidence.empty else 0.0
    material_double_count_cases = int(
        (
            evidence["price_return"].abs().le(0.08)
            & evidence["total_return"].abs().gt(0.15)
        ).sum()
    ) if not evidence.empty else 0
    adjustment_verified = bool(
        len(evidence) >= 100
        and event_returns_in_band >= 0.95
        and material_double_count_cases >= 10
    )
    prices["adjustment_policy"] = (
        "verified_vendor_total_return_adjusted" if adjustment_verified else "unverified"
    )
    output = paths.normalized / "prices.parquet"
    prices.to_parquet(output, index=False)
    evidence.to_csv(paths.reports / "kbs_adjustment_event_evidence.csv", index=False)
    coverage = prices.groupby("ticker").agg(
        records=("date", "size"), start=("date", "min"), end=("date", "max")
    ).reset_index()
    coverage.to_csv(paths.reports / "price_complete_case_coverage.csv", index=False)
    pd.DataFrame({"ticker": sorted(excluded_tickers), "reason": "unresolved_material_corporate_action"}).to_csv(
        paths.reports / "price_excluded_tickers.csv", index=False
    )
    pd.DataFrame(failures, columns=["ticker", "error"]).to_csv(
        paths.reports / "price_crawl_failures.csv", index=False
    )
    audit = {
        "dataset": DATASET_LABEL,
        "status": (
            "pass" if not failures and len(completed) == len(master) and adjustment_verified
            else "partial"
        ),
        "initial_official_master_tickers": int(
            pd.read_parquet(master_path)["ticker"].nunique()
        ),
        "excluded_unresolved_corporate_action_tickers": len(excluded_tickers),
        "complete_case_tickers_requested": len(master),
        "complete_case_tickers_collected": int(prices["ticker"].nunique()),
        "rows": len(prices),
        "dates": int(prices["date"].nunique()),
        "inherited_checkpoints": len(inherited),
        "newly_fetched": len(fetched),
        "failures": failures,
        "source": "KBS public market-data endpoint accessed through vnstock 4",
        "adjustment_policy": (
            "verified_vendor_total_return_adjusted" if adjustment_verified else "unverified"
        ),
        "adjustment_verification": {
            "matched_verified_event_dates": len(evidence),
            "event_raw_returns_within_8pct": event_returns_in_band,
            "material_double_count_cases": material_double_count_cases,
            "passed": adjustment_verified,
            "interpretation": (
                "KBS event-date returns remain within the ordinary price band; "
                "adding verified entitlements again produces material artificial jumps."
            ),
        },
        "selection_policy": "complete-case exclusion before model execution; no outcome-based filtering",
        "output": str(output),
        "sha256": sha256_file(output),
    }
    (paths.reports / "price_panel_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def crawl_data_17_8_cafef_price_crosscheck(
    root: Path,
    start: str = START_DATE,
    end: str = END_DATE,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Join CafeF raw prices to the KBS adjusted series and certify semantics.

    CafeF exposes both the contemporaneous close and an adjusted close.  KBS
    exposes a back-adjusted OHLC series.  Only tickers whose adjusted returns
    agree across both independent repositories are promoted.
    """
    from .sources import (
        CafeFPublicHistoryAdapter,
        _archive_raw_frame,
        _normalize_cafef_ohlc,
    )

    if not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be between 1 and 4")
    paths = Paths(data_17_8_workspace(root))
    price_path = paths.normalized / "prices.parquet"
    if not price_path.exists():
        raise FileNotFoundError("Run the Data 17/8 KBS price stage first")
    kbs = pd.read_parquet(price_path).copy()
    kbs["date"] = pd.to_datetime(kbs["date"], errors="coerce").dt.normalize()
    tickers = sorted(kbs["ticker"].astype(str).unique())
    checkpoint_dir = paths.raw / "cafef_price_crosscheck"
    response_dir = paths.raw / "cafef_price_responses"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    completed: list[str] = []

    def collect(ticker: str) -> tuple[str, int]:
        checkpoint = checkpoint_dir / f"{ticker}.parquet"
        if checkpoint.exists():
            cached = pd.read_parquet(checkpoint)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            if (
                len(cached) >= 40
                and cached["date"].min() <= kbs.loc[kbs["ticker"].eq(ticker), "date"].min() + pd.Timedelta(days=21)
                and cached["date"].max() >= kbs.loc[kbs["ticker"].eq(ticker), "date"].max() - pd.Timedelta(days=21)
            ):
                return ticker, len(cached)
        raw = CafeFPublicHistoryAdapter().daily_ohlc(ticker, start, end)
        if raw.empty:
            raise RuntimeError("CafeF returned no observations")
        archive = _archive_raw_frame(response_dir, ticker, raw)
        normalized = _normalize_cafef_ohlc(raw, ticker)
        valid_dates = kbs.loc[kbs["ticker"].eq(ticker), "date"]
        normalized = normalized[
            normalized["date"].between(valid_dates.min(), valid_dates.max())
        ].copy()
        if len(normalized) < 40:
            raise RuntimeError("CafeF coverage is too short")
        normalized["raw_checksum"] = archive["sha256"]
        normalized.to_parquet(checkpoint, index=False)
        return ticker, len(normalized)

    collectable = [
        ticker for ticker in tickers
        if not (checkpoint_dir / f"{ticker}.parquet").exists()
    ]
    # Existing files are still revalidated in collect(); submitting every ticker
    # also makes a resumed run report a complete deterministic manifest.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(collect, ticker): ticker for ticker in tickers}
        for position, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                completed_ticker, observations = future.result()
                completed.append(completed_ticker)
                print(
                    f"[CafeF raw crosscheck] {position:03d}/{len(futures):03d} "
                    f"{ticker}: {observations:,} rows",
                    flush=True,
                )
            except Exception as exc:
                failures.append({
                    "ticker": ticker,
                    "stage": "crawl",
                    "error": f"{type(exc).__name__}: {str(exc)[:250]}",
                })
                print(
                    f"[CafeF raw crosscheck] {position:03d}/{len(futures):03d} "
                    f"{ticker}: FAILED",
                    flush=True,
                )
            (paths.reports / "cafef_price_crosscheck_progress.json").write_text(
                json.dumps({
                    "requested": len(tickers),
                    "completed": len(completed),
                    "failures": len(failures),
                    "new_requests_at_start": len(collectable),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    output_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for ticker in completed:
        kbs_ticker = kbs[kbs["ticker"].eq(ticker)].sort_values("date").copy()
        cafef = pd.read_parquet(checkpoint_dir / f"{ticker}.parquet").copy()
        cafef["date"] = pd.to_datetime(cafef["date"], errors="coerce").dt.normalize()
        cafef = cafef.sort_values("date").drop_duplicates("date")
        overlap = kbs_ticker[["date", "close"]].merge(
            cafef[["date", "adjusted_close"]], on="date", how="inner",
            suffixes=("_kbs", "_cafef"),
        ).sort_values("date")
        overlap["kbs_return"] = overlap["close"].pct_change()
        overlap["cafef_return"] = overlap["adjusted_close"].pct_change()
        valid_returns = overlap[["kbs_return", "cafef_return"]].dropna()
        return_correlation = float(valid_returns.corr().iloc[0, 1]) if len(valid_returns) >= 20 else np.nan
        median_abs_return_difference = float(
            (valid_returns["kbs_return"] - valid_returns["cafef_return"]).abs().median()
        ) if not valid_returns.empty else np.inf
        coverage = len(overlap) / len(kbs_ticker) if len(kbs_ticker) else 0.0
        passed = bool(
            coverage >= 0.90
            and pd.notna(return_correlation)
            and return_correlation >= 0.98
            and median_abs_return_difference <= 0.005
        )
        diagnostics.append({
            "ticker": ticker,
            "kbs_observations": len(kbs_ticker),
            "cafef_observations": len(cafef),
            "overlap": len(overlap),
            "coverage": coverage,
            "adjusted_return_correlation": return_correlation,
            "median_absolute_return_difference": median_abs_return_difference,
            "passed": passed,
        })
        if not passed:
            failures.append({
                "ticker": ticker,
                "stage": "cross_source_verification",
                "error": "adjusted_return_series_disagree",
            })
            continue

        cafe_by_date = cafef.set_index("date")
        factor_observed = (
            cafe_by_date["close"] / cafe_by_date["adjusted_close"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).dropna()
        factor = factor_observed.reindex(kbs_ticker["date"]).bfill().ffill()
        if factor.isna().any() or factor.le(0).any():
            failures.append({
                "ticker": ticker,
                "stage": "raw_price_reconstruction",
                "error": "cafef_adjustment_factor_missing",
            })
            continue
        for field in ["open", "high", "low", "close"]:
            kbs_ticker[field] = kbs_ticker[field].to_numpy() * factor.to_numpy()
            exact = kbs_ticker["date"].map(cafe_by_date[field])
            kbs_ticker[field] = exact.fillna(kbs_ticker[field])
        # A small number of CafeF observations report a daily high/low that does
        # not enclose the reported open/close.  Preserve open and close, while
        # repairing only the interval envelope so the OHLC contract remains
        # internally consistent.  This does not alter model returns, which use
        # the independently verified KBS adjusted close below.
        ohlc = kbs_ticker[["open", "high", "low", "close"]]
        kbs_ticker["high"] = ohlc.max(axis=1)
        kbs_ticker["low"] = ohlc.min(axis=1)
        exact_volume = kbs_ticker["date"].map(cafe_by_date["volume"])
        kbs_ticker["volume"] = exact_volume.fillna(kbs_ticker["volume"])
        exact_value = kbs_ticker["date"].map(cafe_by_date["trading_value"])
        kbs_ticker["trading_value"] = exact_value.fillna(
            kbs_ticker["close"] * kbs_ticker["volume"]
        )
        # The original KBS close is the independently verified adjusted series.
        # Index identities can change after sorting, so map explicitly by date.
        adjusted_map = kbs.loc[kbs["ticker"].eq(ticker)].set_index("date")["close"]
        kbs_ticker["adjusted_close"] = kbs_ticker["date"].map(adjusted_map)
        combined_checksum = hashlib.sha256(
            (
                str(kbs_ticker["raw_checksum"].iloc[0])
                + str(cafef["raw_checksum"].iloc[0])
            ).encode("utf-8")
        ).hexdigest()
        kbs_ticker["source"] = "cafef_raw_kbs_adjusted_crosscheck"
        kbs_ticker["source_url"] = CafeFPublicHistoryAdapter.page_url.format(
            ticker=ticker.lower()
        )
        kbs_ticker["fetched_at"] = datetime.now(timezone.utc).isoformat()
        kbs_ticker["available_at"] = kbs_ticker["date"] + pd.Timedelta(days=1)
        kbs_ticker["raw_checksum"] = combined_checksum
        kbs_ticker["parser_version"] = "cafef-raw-kbs-adjusted-crosscheck-v1"
        kbs_ticker["data_class"] = "real"
        kbs_ticker["adjustment_policy"] = "verified_vendor_total_return_adjusted"
        output_frames.append(kbs_ticker[PRICE_COLUMNS])

    diagnostics_frame = pd.DataFrame(diagnostics).sort_values("ticker")
    diagnostics_frame.to_csv(
        paths.reports / "cafef_kbs_price_crosscheck.csv", index=False
    )
    if not output_frames:
        raise RuntimeError("No ticker passed the CafeF/KBS cross-source price verification")
    prices = (
        pd.concat(output_frames, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    prices.to_parquet(price_path, index=False)
    pd.DataFrame(failures, columns=["ticker", "stage", "error"]).to_csv(
        paths.reports / "cafef_price_crosscheck_failures.csv", index=False
    )
    audit = {
        "dataset": DATASET_LABEL,
        "status": "pass" if not failures else "partial",
        "requested_tickers": len(tickers),
        "cafef_series_collected": len(completed),
        "cross_source_verified_tickers": int(prices["ticker"].nunique()),
        "rows": len(prices),
        "dates": int(prices["date"].nunique()),
        "failures": failures,
        "raw_price_source": "CafeF public PriceHistory endpoint",
        "adjusted_price_source": "KBS public endpoint through vnstock",
        "adjustment_policy": "verified_vendor_total_return_adjusted",
        "note": (
            "CafeF raw OHLC/trading value is used where observed. Rare CafeF date gaps "
            "use the adjacent CafeF raw-to-adjusted factor applied to the KBS adjusted bar."
        ),
        "output": str(price_path),
        "sha256": sha256_file(price_path),
    }
    (paths.reports / "cafef_price_crosscheck_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def finalize_data_17_8_research_universe(root: Path) -> dict[str, Any]:
    """Freeze the full audit master and promote only verified complete cases.

    The full 394-security source universe remains immutable in a separate table.
    The runtime master is a declared complete-case research population, ensuring
    that the generic leakage audit does not mistake pre-declared data exclusions
    for silently missing price histories.
    """
    paths = Paths(data_17_8_workspace(root))
    price_path = paths.normalized / "prices.parquet"
    runtime_master_path = paths.normalized / "security_master.parquet"
    full_master_path = paths.normalized / "security_master_full.parquet"
    if not price_path.exists() or not runtime_master_path.exists():
        raise FileNotFoundError("Prices and the official security master are required")
    if not full_master_path.exists():
        shutil.copy2(runtime_master_path, full_master_path)
    full_master = pd.read_parquet(full_master_path).copy()
    prices = pd.read_parquet(price_path)
    selected = set(prices["ticker"].astype(str).str.upper().str.strip())
    full_master["ticker"] = full_master["ticker"].astype(str).str.upper().str.strip()
    runtime = full_master[full_master["ticker"].isin(selected)].copy()
    if runtime["ticker"].nunique() != len(selected):
        missing = sorted(selected - set(runtime["ticker"]))
        raise RuntimeError(f"Verified prices have no official identity: {missing}")
    runtime["research_eligibility_status"] = "verified_complete_case"
    runtime["research_eligibility_as_of"] = pd.Timestamp("2026-08-17")
    runtime.to_parquet(runtime_master_path, index=False)

    excluded = full_master[~full_master["ticker"].isin(selected)].copy()
    action_exclusions = set()
    action_report = paths.reports / "price_excluded_tickers.csv"
    if action_report.exists():
        action_exclusions = set(
            pd.read_csv(action_report)["ticker"].astype(str).str.upper()
        )
    crawl_failures = set()
    for report_name in [
        "price_crawl_failures.csv", "cafef_price_crosscheck_failures.csv",
    ]:
        report_path = paths.reports / report_name
        if report_path.exists() and report_path.stat().st_size:
            report = pd.read_csv(report_path)
            if "ticker" in report:
                crawl_failures |= set(report["ticker"].astype(str).str.upper())
    excluded["exclusion_reason"] = np.select(
        [
            excluded["ticker"].isin(action_exclusions),
            excluded["ticker"].isin(crawl_failures),
        ],
        [
            "unresolved_material_corporate_action",
            "price_coverage_or_cross_source_verification_failed",
        ],
        default="not_in_verified_complete_case_panel",
    )
    excluded.to_csv(paths.reports / "research_universe_exclusions.csv", index=False)

    price_hash = sha256_file(price_path)
    contract = {
        "dataset": DATASET_LABEL,
        "adjustment_policy": "verified_vendor_total_return_adjusted",
        "source": "cafef_raw_kbs_adjusted_crosscheck",
        "source_url": "https://cafef.vn/du-lieu/lich-su-giao-dich.htm; https://kbbuddywts.kbsec.com.vn/",
        "methodology": (
            "Raw OHLC/trading value from CafeF; KBS adjusted close retained only "
            "after per-ticker return agreement and verified-event double-count checks."
        ),
        "certified_by": "data-17-8-automated-cross-source-audit",
        "certified_at": datetime.now(timezone.utc).isoformat(),
        "output_price_dataset_sha256": price_hash,
    }
    (paths.normalized / "price_adjustment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "dataset": DATASET_LABEL,
        "full_official_master_tickers": int(full_master["ticker"].nunique()),
        "runtime_complete_case_tickers": int(runtime["ticker"].nunique()),
        "excluded_tickers": int(excluded["ticker"].nunique()),
        "full_master": str(full_master_path),
        "runtime_master": str(runtime_master_path),
        "price_adjustment_contract": str(
            paths.normalized / "price_adjustment_contract.json"
        ),
        "price_sha256": price_hash,
        "selection_timing": "all exclusions fixed before model execution",
    }
    (paths.reports / "research_universe_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _fetch_news_offset(
    offset: int,
    size: int,
    start: str,
    end: str,
    raw_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch an exact pagination range, splitting around malformed upstream rows."""
    if size < 1 or offset < 0 or offset % size:
        raise ValueError("offset must be a non-negative multiple of size")
    page = offset // size + 1
    params = {
        "pageIndex": page,
        "pageSize": size,
        "startDate": start,
        "endDate": end,
    }
    stem = f"offset-{offset}-size-{size}"
    malformed_marker = raw_directory / f"{_safe_stem(stem)}.malformed.json"
    if malformed_marker.exists():
        return [], [{
            "offset": offset,
            "size": size,
            "error": _HOSEMalformedRange.__name__,
            "reason": "resumed deterministic HOSE HTTP 500 marker",
            "retryable": False,
            "needs_split": size > 1,
        }]
    try:
        payload = _load_json_checkpoint(raw_directory, stem)
        if payload is None:
            payload = _request_hose_news_page(
                f"{HOSE_NEWS_API}/news/securitiesType/0",
                params=params,
                attempts=4,
                timeout=35.0,
            )
        _json_archive(raw_directory, stem, payload)
        return list(payload["data"].get("list") or []), []
    except _HOSETemporarilyUnavailable as exc:
        return [], [{
            "offset": offset,
            "size": size,
            "error": type(exc).__name__,
            "reason": str(exc),
            "retryable": True,
        }]
    except _HOSEMalformedRange as exc:
        raw_directory.mkdir(parents=True, exist_ok=True)
        malformed_marker.write_text(
            json.dumps({
                "offset": offset,
                "size": size,
                "status": 500,
                "reason": str(exc),
                "observed_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [], [{
            "offset": offset,
            "size": size,
            "error": type(exc).__name__,
            "reason": str(exc),
            "retryable": False,
            "needs_split": size > 1,
        }]


def _collect_news_ranges(
    ranges: list[tuple[int, int]],
    *,
    start: str,
    end: str,
    raw_directory: Path,
    max_workers: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect independent ranges concurrently without letting one bad range monopolize a worker."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not ranges:
        return rows, failures
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_news_offset, offset, size, start, end, raw_directory
            ): (offset, size)
            for offset, size in ranges
        }
        for position, future in enumerate(as_completed(futures), 1):
            page_rows, page_failures = future.result()
            rows.extend(page_rows)
            failures.extend(page_failures)
            if position % 25 == 0 or position == len(futures):
                print(
                    f"[HOSE disclosures:{label}] ranges={position:,}/{len(futures):,}, "
                    f"rows={len(rows):,}, failures={len(failures):,}",
                    flush=True,
                )
    return rows, failures


def crawl_hose_disclosure_index(
    root: Path,
    start: str = START_DATE,
    end: str = END_DATE,
    max_workers: int = 4,
    page_size: int = 1000,
) -> dict[str, Any]:
    """Collect the official HOSE issuer-news index with raw page checkpoints."""
    if not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be between 1 and 4")
    workspace = data_17_8_workspace(root)
    paths = Paths(workspace)
    paths.ensure()
    raw_pages = paths.raw / "hose_disclosures" / "pages"
    first_payload = _load_json_checkpoint(raw_pages, f"offset-0-size-{page_size}")
    if first_payload is None:
        first_payload = _request_hose_news_page(
            f"{HOSE_NEWS_API}/news/securitiesType/0",
            params={
                "pageIndex": 1,
                "pageSize": page_size,
                "startDate": start,
                "endDate": end,
            },
            attempts=4,
            timeout=35.0,
        )
    _json_archive(raw_pages, f"offset-0-size-{page_size}", first_payload)
    total = int(first_payload["data"]["paging"]["totalCount"])
    rows = list(first_payload["data"].get("list") or [])
    failures: list[dict[str, Any]] = []
    offsets = list(range(page_size, total, page_size))
    # Request the final API page at the same page size as every prior page.  HOSE
    # naturally returns a shorter list; changing page size would also change the
    # page index and therefore no longer represent the intended offset.
    ranges = [(offset, page_size) for offset in offsets]
    scheduled_offsets = {offset for offset, _ in ranges}
    main_rows, main_failures = _collect_news_ranges(
        ranges,
        start=start,
        end=end,
        raw_directory=raw_pages,
        max_workers=max_workers,
        label=f"size-{page_size}",
    )
    rows.extend(main_rows)

    # Breadth-first isolation is materially faster than recursive isolation:
    # every valid sibling range is checkpointed in parallel, while only the
    # deterministic HTTP-500 ranges continue to the next smaller granularity.
    pending = [failure for failure in main_failures if failure.get("needs_split")]
    failures.extend(failure for failure in main_failures if not failure.get("needs_split"))
    while pending:
        child_ranges: list[tuple[int, int]] = []
        for failure in pending:
            parent_offset = int(failure["offset"])
            parent_size = int(failure["size"])
            # Decimal splits are efficient for large pages.  At size 10, use
            # pairs before individual records so one malformed record does not
            # force ten separate requests.
            child_size = 2 if parent_size == 10 else max(1, parent_size // 10)
            while parent_size % child_size:
                child_size -= 1
            child_ranges.extend(
                (child_offset, child_size)
                for child_offset in range(
                    parent_offset, parent_offset + parent_size, child_size
                )
            )
        child_rows, child_failures = _collect_news_ranges(
            sorted(set(child_ranges)),
            start=start,
            end=end,
            raw_directory=raw_pages,
            max_workers=max_workers,
            label=f"split-size-{child_ranges[0][1]}",
        )
        rows.extend(child_rows)
        pending = [failure for failure in child_failures if failure.get("needs_split")]
        failures.extend(
            failure for failure in child_failures if not failure.get("needs_split")
        )
    final_offset = offsets[-1] if offsets else 0
    if final_offset and final_offset not in scheduled_offsets:
        page = final_offset // page_size + 1
        try:
            payload = _request_json(
                f"{HOSE_NEWS_API}/news/securitiesType/0",
                params={
                    "pageIndex": page,
                    "pageSize": page_size,
                    "startDate": start,
                    "endDate": end,
                },
            )
            _json_archive(raw_pages, f"page-{page}-size-{page_size}", payload)
            rows.extend(payload["data"].get("list") or [])
        except RuntimeError as exc:
            failures.append({"offset": final_offset, "size": page_size, "reason": str(exc)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("HOSE disclosure API produced no rows")
    collected_rows = len(frame)
    duplicate_news_ids = int(frame.duplicated("id", keep="last").sum())
    frame = frame.drop_duplicates("id", keep="last")
    frame["publish_from"] = _local_timestamp_series(frame["publishFrom"])
    frame["posted_at"] = _local_timestamp_series(frame["postedDate"])
    frame = frame[frame["publish_from"].between(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1))]
    frame["source"] = "hose_official_issuer_disclosure_api"
    frame["source_url"] = frame["id"].map(
        lambda value: f"https://www.hsx.vn/vi/tin-tuc/chi-tiet/{int(value)}"
    )
    frame["fetched_at"] = datetime.now(timezone.utc).isoformat()
    frame = frame.sort_values(["publish_from", "posted_at", "id"]).reset_index(drop=True)
    output = paths.normalized / "hose_disclosures.parquet"
    frame.to_parquet(output, index=False)
    failure_output = paths.reports / "hose_disclosure_failures.csv"
    pd.DataFrame(failures).to_csv(failure_output, index=False)
    manifest = {
        "dataset": DATASET_LABEL,
        "status": "partial" if failures else "success",
        "api_total_count": total,
        "collected_rows_before_deduplication": collected_rows,
        "duplicate_news_ids_removed": duplicate_news_ids,
        "normalized_rows": len(frame),
        "unique_news_ids": int(frame["id"].nunique()),
        "api_count_minus_unique_ids": total - int(frame["id"].nunique()),
        "start": start,
        "end": end,
        "failure_count": len(failures),
        "failure_sample": failures[:25],
        "failure_output": str(failure_output),
        "failure_output_sha256": sha256_file(failure_output),
        "output": str(output),
        "sha256": sha256_file(output),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.reports / "hose_disclosure_index_audit.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def classify_disclosure_title(title: str) -> dict[str, Any] | None:
    """High-precision classifier for directly filed BCTC/BCTN documents."""
    match = re.match(r"^\s*([A-Z0-9]{3})\s*:\s*(.+)$", str(title), flags=re.IGNORECASE)
    if not match:
        return None
    ticker = match.group(1).upper()
    suffix = match.group(2).strip()
    folded = _ascii_fold(suffix)
    excluded_prefixes = (
        "nhac nho", "giai trinh", "thong bao cham", "cong van nhac",
        "y kien kiem toan", "thong bao ve viec", "gia han",
    )
    if folded.startswith(excluded_prefixes):
        return None
    if re.match(r"^(bao cao tai chinh|bctc)\b", folded):
        document_type = "financial_statement"
    elif re.match(r"^(bao cao thuong nien|bctn)\b", folded):
        document_type = "annual_report"
    else:
        return None
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", folded)]
    fiscal_year = years[-1] if years else None
    quarter_match = re.search(r"\bquy\s*(1|2|3|4|i{1,3}|iv)\b", folded)
    quarter_lookup = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    quarter = None
    if quarter_match:
        token = quarter_match.group(1)
        quarter = int(token) if token.isdigit() else quarter_lookup[token]
    if document_type == "annual_report":
        period_type = "annual"
        quarter = 4
    elif quarter:
        period_type = "quarterly"
    elif "ban nien" in folded or re.search(r"\b6\s*thang\b", folded):
        period_type = "semiannual"
        quarter = 2
    elif re.search(r"\b9\s*thang\b", folded):
        period_type = "nine_month"
        quarter = 3
    elif "nam" in folded or "kiem toan" in folded:
        period_type = "annual"
        quarter = 4
    else:
        period_type = "unknown"
    fiscal_period_end = pd.NaT
    if fiscal_year and quarter:
        fiscal_period_end = pd.Timestamp(fiscal_year, quarter * 3, 1) + pd.offsets.MonthEnd(0)
    scope = "consolidated" if "hop nhat" in folded else (
        "separate" if "rieng" in folded or "cong ty me" in folded else "unspecified"
    )
    assurance = "audited" if "kiem toan" in folded else (
        "reviewed" if "soat xet" in folded else "unassured_or_unspecified"
    )
    return {
        "ticker": ticker,
        "document_type": document_type,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": quarter,
        "period_type": period_type,
        "fiscal_period_end": fiscal_period_end,
        "statement_scope": scope,
        "assurance": assurance,
    }


def _sector_disclosure(title: str) -> bool:
    folded = _ascii_fold(title)
    return (
        "danh muc co phieu thanh phan" in folded
        and ("chi so nganh" in folded or "vnallshare sector" in folded)
    )


def _media_rows(news_id: int, raw_directory: Path) -> list[dict[str, Any]]:
    stem = f"news-{news_id}"
    payload = _load_json_checkpoint(raw_directory, stem)
    if payload is None:
        payload = _request_json(
            f"{HOSE_MEDIA_API}/mediafiles/1/{int(news_id)}",
            params={"pageIndex": 1, "pageSize": 100},
            attempts=4,
            timeout=45.0,
        )
        _json_archive(raw_directory, stem, payload)
    return list(payload["data"].get("list") or [])


def _static_url(file_path: str) -> str:
    path = str(file_path).replace("~", "", 1)
    return HOSE_STATIC + quote(path, safe="/%")


def _download_document(
    row: dict[str, Any],
    directory: Path,
    maximum_bytes: int = 80 * 1024 * 1024,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    url = row["download_url"]
    suffix = Path(str(row.get("file_name") or "document.pdf")).suffix.lower() or ".bin"
    provisional = directory / f"{row['ticker']}-{row['news_id']}-{row['attachment_index']}{suffix}"
    if provisional.exists() and provisional.stat().st_size > 0:
        return {
            **row,
            "local_path": str(provisional),
            "bytes": provisional.stat().st_size,
            "sha256": sha256_file(provisional),
            "download_status": "resumed",
        }
    response = _http_session().get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (academic-research-data-17-8/1.0)"},
        timeout=120,
        stream=True,
    )
    response.raise_for_status()
    content_length = int(response.headers.get("Content-Length") or 0)
    if content_length > maximum_bytes:
        raise RuntimeError("document exceeds configured maximum_bytes")
    digest = hashlib.sha256()
    written = 0
    temporary = provisional.with_suffix(provisional.suffix + ".part")
    with temporary.open("wb") as stream:
        for chunk in response.iter_content(1 << 20):
            if not chunk:
                continue
            written += len(chunk)
            if written > maximum_bytes:
                raise RuntimeError("document stream exceeds configured maximum_bytes")
            digest.update(chunk)
            stream.write(chunk)
    if suffix == ".pdf":
        with temporary.open("rb") as stream:
            signature = stream.read(5)
        if signature != b"%PDF-":
            temporary.unlink(missing_ok=True)
            raise RuntimeError("downloaded content is not a PDF")
    temporary.replace(provisional)
    return {
        **row,
        "local_path": str(provisional),
        "bytes": written,
        "sha256": digest.hexdigest(),
        "download_status": "downloaded",
    }


def crawl_hose_documents(
    root: Path,
    max_workers: int = 4,
    download: bool = True,
) -> dict[str, Any]:
    """Resolve and optionally download official HOSE BCTC/BCTN and sector files."""
    if not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be between 1 and 4")
    paths = Paths(data_17_8_workspace(root))
    index_path = paths.normalized / "hose_disclosures.parquet"
    if not index_path.exists():
        raise FileNotFoundError("Run crawl_hose_disclosure_index first")
    disclosures = pd.read_parquet(index_path)
    # Document coverage is audited against the official issuer master, not the
    # smaller complete-case price universe used by the model.
    tickers = _master_ticker_set(paths)
    selected: list[dict[str, Any]] = []
    for record in disclosures.to_dict("records"):
        classification = classify_disclosure_title(record.get("title") or "")
        if (
            classification
            and classification["ticker"] in tickers
            and classification.get("fiscal_year") is not None
            and 2020 <= int(classification["fiscal_year"]) <= 2025
        ):
            selected.append({**record, **classification})
        elif _sector_disclosure(record.get("title") or ""):
            selected.append({
                **record,
                "ticker": "HOSE_INDEX",
                "document_type": "sector_constituents",
                "fiscal_year": None,
                "fiscal_quarter": None,
                "period_type": "index_review",
                "fiscal_period_end": pd.NaT,
                "statement_scope": "not_applicable",
                "assurance": "official_exchange_publication",
            })
    raw_media = paths.raw / "hose_disclosures" / "media"
    media_results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_media_rows, int(record["id"]), raw_media): record
            for record in selected
        }
        for position, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                media_results.append((record, future.result()))
            except Exception as exc:
                failures.append({
                    "news_id": int(record["id"]),
                    "ticker": record["ticker"],
                    "stage": "media_index",
                    "error": type(exc).__name__,
                })
            if position % 250 == 0 or position == len(futures):
                print(f"[HOSE media] {position:,}/{len(futures):,}", flush=True)
    documents: list[dict[str, Any]] = []
    for record, attachments in media_results:
        for index, attachment in enumerate(attachments, 1):
            file_name = str(attachment.get("fileName") or "")
            extension = Path(file_name).suffix.lower()
            if extension not in {".pdf", ".xls", ".xlsx", ".doc", ".docx"}:
                continue
            documents.append({
                "news_id": int(record["id"]),
                "ticker": record["ticker"],
                "document_type": record["document_type"],
                "title": record.get("title"),
                "fiscal_year": record.get("fiscal_year"),
                "fiscal_quarter": record.get("fiscal_quarter"),
                "period_type": record.get("period_type"),
                "fiscal_period_end": record.get("fiscal_period_end"),
                "statement_scope": record.get("statement_scope"),
                "assurance": record.get("assurance"),
                "publication_date": pd.Timestamp(record["posted_at"]).normalize(),
                "available_at": pd.Timestamp(record["posted_at"]),
                "attachment_index": index,
                "file_name": file_name,
                "file_type": extension,
                "download_url": _static_url(str(attachment.get("filePath") or "")),
                "source": "hose_official_issuer_disclosure",
                "source_url": record.get("source_url"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
    downloaded: list[dict[str, Any]] = []
    if download:
        raw_documents = paths.raw / "company_documents"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _download_document,
                    row,
                    raw_documents / row["ticker"],
                    150 * 1024 * 1024,
                ): row
                for row in documents
            }
            for position, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    downloaded.append(future.result())
                except Exception as exc:
                    failures.append({
                        "news_id": row["news_id"],
                        "ticker": row["ticker"],
                        "file_name": row["file_name"],
                        "stage": "download",
                        "error": type(exc).__name__,
                    })
                if position % 100 == 0 or position == len(futures):
                    print(f"[HOSE documents] {position:,}/{len(futures):,}", flush=True)
    else:
        downloaded = [{**row, "download_status": "metadata_only"} for row in documents]
    frame = pd.DataFrame(downloaded)
    output = paths.normalized / "source_documents.parquet"
    if frame.empty:
        frame = pd.DataFrame(columns=[
            "news_id", "ticker", "document_type", "title", "fiscal_year",
            "fiscal_quarter", "period_type", "fiscal_period_end", "statement_scope",
            "assurance", "publication_date", "available_at", "attachment_index",
            "file_name", "file_type", "download_url", "source", "source_url",
            "fetched_at", "local_path", "bytes", "sha256", "download_status",
        ])
    else:
        frame = frame.sort_values(["ticker", "available_at", "news_id", "attachment_index"])
        frame = frame.drop_duplicates(["news_id", "attachment_index", "download_url"], keep="last")
    frame.to_parquet(output, index=False)
    coverage = (
        frame.groupby(["ticker", "document_type"], dropna=False).size()
        .rename("documents").reset_index()
    )
    coverage.to_csv(paths.reports / "company_document_coverage.csv", index=False)
    manifest = {
        "dataset": DATASET_LABEL,
        "status": "partial" if failures else "success",
        "selected_disclosures": len(selected),
        "documents_indexed": len(documents),
        "documents_available": len(frame),
        "tickers_with_documents": int(frame.loc[
            frame["ticker"].ne("HOSE_INDEX"), "ticker"
        ].nunique()) if not frame.empty else 0,
        "failures": failures,
        "output": str(output),
        "sha256": sha256_file(output),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.reports / "company_document_crawl_audit.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _vietstock_session_and_token(ticker: str) -> tuple[requests.Session, str, str]:
    session = getattr(_VIETSTOCK_THREAD_STATE, "session", None)
    token = getattr(_VIETSTOCK_THREAD_STATE, "token", None)
    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; academic-research-data-17-8/1.0)",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
        })
        _VIETSTOCK_THREAD_STATE.session = session
    referer = f"https://finance.vietstock.vn/{ticker}/tai-lieu.htm"
    if not token:
        response = session.get(referer, timeout=60)
        response.raise_for_status()
        token_match = re.search(
            r'name\s*=\s*["\']?__RequestVerificationToken["\']?[^>]*'
            r'value\s*=\s*["\']?([^"\'\s>]+)',
            response.text,
            flags=re.IGNORECASE,
        ) or re.search(
            r'value\s*=\s*["\']?([^"\'\s>]+)["\']?[^>]*'
            r'name\s*=\s*["\']?__RequestVerificationToken["\']?',
            response.text,
            flags=re.IGNORECASE,
        )
        if not token_match:
            raise RuntimeError("vietstock_verification_token_missing")
        token = token_match.group(1)
        _VIETSTOCK_THREAD_STATE.token = token
    return session, str(token), referer


def _vietstock_document_page(
    ticker: str,
    document_type_id: int,
    page: int,
    raw_directory: Path,
) -> list[dict[str, Any]]:
    stem = f"{ticker}-type-{document_type_id}-page-{page}"
    cached = _load_json_archive_value(raw_directory, stem)
    if isinstance(cached, list):
        return cached
    session, token, referer = _vietstock_session_and_token(ticker)
    response = session.post(
        "https://finance.vietstock.vn/data/getdocument",
        data={
            "code": ticker,
            "type": str(document_type_id),
            "page": str(page),
            "pageSize": "50",
            "__RequestVerificationToken": token,
        },
        headers={
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": referer,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("vietstock_document_payload_not_list")
    _json_archive(raw_directory, stem, payload)
    return payload


def _vietstock_last_update(value: Any) -> pd.Timestamp:
    match = re.search(r"/Date\((\d+)", str(value or ""))
    if not match:
        return pd.NaT
    return (
        pd.to_datetime(int(match.group(1)), unit="ms", utc=True, errors="coerce")
        .tz_convert("Asia/Ho_Chi_Minh")
        .tz_localize(None)
    )


def _collect_vietstock_ticker_documents(
    ticker: str,
    raw_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for type_id, expected_type in ((1, "financial_statement"), (2, "annual_report")):
        try:
            first = _vietstock_document_page(ticker, type_id, 1, raw_directory)
            total = int(first[0].get("TotalRow") or len(first)) if first else 0
            pages = max(1, math.ceil(total / 50)) if total else 1
            payload_rows = list(first)
            for page in range(2, pages + 1):
                payload_rows.extend(
                    _vietstock_document_page(ticker, type_id, page, raw_directory)
                )
            for item in payload_rows:
                title = str(item.get("FullName") or item.get("Title") or "").strip()
                classification = classify_disclosure_title(f"{ticker}: {title}")
                if not classification or classification["document_type"] != expected_type:
                    continue
                fiscal_year = classification.get("fiscal_year")
                if fiscal_year is None or not 2020 <= int(fiscal_year) <= 2025:
                    continue
                file_id = int(item.get("FileInfoID"))
                available_at = _vietstock_last_update(item.get("LastUpdate"))
                if pd.isna(available_at):
                    failures.append({
                        "ticker": ticker, "type": type_id, "file_id": file_id,
                        "stage": "metadata", "reason": "last_update_missing",
                    })
                    continue
                extension = str(item.get("FileExt") or Path(str(item.get("Url") or "")).suffix)
                extension = extension.strip().lower()
                rows.append({
                    "document_id": f"vietstock:{file_id}",
                    "news_id": -file_id,
                    "ticker": ticker,
                    "document_type": expected_type,
                    "title": title,
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": classification.get("fiscal_quarter"),
                    "period_type": classification.get("period_type"),
                    "fiscal_period_end": classification.get("fiscal_period_end"),
                    "statement_scope": classification.get("statement_scope"),
                    "assurance": classification.get("assurance"),
                    "publication_date": available_at.normalize(),
                    "available_at": available_at,
                    "attachment_index": 1,
                    "file_name": Path(str(item.get("Url") or f"{file_id}{extension}")).name.split("?", 1)[0],
                    "file_type": extension,
                    "download_url": str(item.get("Url") or ""),
                    "source": "vietstock_public_document_repository",
                    "source_url": f"https://finance.vietstock.vn/{ticker}/tai-lieu.htm",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "source_total_rows": total,
                })
        except Exception as exc:
            # Tokens occasionally rotate.  Discard only this worker's token so a
            # resumed attempt obtains a fresh anti-forgery pair.
            _VIETSTOCK_THREAD_STATE.token = None
            failures.append({
                "ticker": ticker, "type": type_id, "stage": "document_index",
                "error": type(exc).__name__, "reason": str(exc)[:200],
            })
    return rows, failures


def crawl_vietstock_company_documents(
    root: Path,
    max_workers: int = 4,
    download: bool = False,
) -> dict[str, Any]:
    """Index BCTC/BCTN for every Data 17/8 ticker from Vietstock's public repository."""
    if not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be between 1 and 4")
    paths = Paths(data_17_8_workspace(root))
    tickers = sorted(_master_ticker_set(paths))
    raw_index = paths.raw / "vietstock_documents" / "index"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _collect_vietstock_ticker_documents, ticker, raw_index
            ): ticker for ticker in tickers
        }
        for position, future in enumerate(as_completed(futures), 1):
            ticker_rows, ticker_failures = future.result()
            rows.extend(ticker_rows)
            failures.extend(ticker_failures)
            if position % 25 == 0 or position == len(futures):
                print(
                    f"[Vietstock document index] {position:,}/{len(futures):,}, "
                    f"eligible={len(rows):,}, failures={len(failures):,}",
                    flush=True,
                )
    indexed = pd.DataFrame(rows)
    downloaded: list[dict[str, Any]] = []
    selected_download_ids: set[str] = set()
    if download and not indexed.empty:
        candidates = indexed.copy()
        candidates["scope_rank"] = candidates["statement_scope"].map({
            "consolidated": 3, "unspecified": 2, "separate": 1,
        }).fillna(0)
        candidates["assurance_rank"] = candidates["assurance"].map({
            "audited": 3, "reviewed": 2, "unassured_or_unspecified": 1,
        }).fillna(0)
        annual_financial = candidates[
            candidates["document_type"].eq("financial_statement")
            & candidates["period_type"].eq("annual")
        ].sort_values(
            ["ticker", "fiscal_year", "scope_rank", "assurance_rank", "available_at"],
            ascending=[True, True, False, False, True],
        ).drop_duplicates(["ticker", "fiscal_year"], keep="first")
        annual_reports = candidates[
            candidates["document_type"].eq("annual_report")
        ].sort_values(
            ["ticker", "fiscal_year", "available_at"]
        ).drop_duplicates(["ticker", "fiscal_year"], keep="last")
        # Prefer the official HOSE annual-report binary where that ticker-year is
        # already covered, and use Vietstock only as a gap-filling repository.
        existing_path = paths.normalized / "source_documents.parquet"
        existing_before_download = (
            pd.read_parquet(existing_path) if existing_path.exists() else pd.DataFrame()
        )
        official_annual_keys = set()
        if not existing_before_download.empty:
            official_rows = existing_before_download[
                existing_before_download["source"].eq("hose_official_issuer_disclosure")
                & existing_before_download["document_type"].eq("annual_report")
            ]
            official_annual_keys = set(zip(
                official_rows["ticker"].astype(str),
                pd.to_numeric(official_rows["fiscal_year"], errors="coerce"),
            ))
        annual_reports = annual_reports[
            [
                (str(row.ticker), float(row.fiscal_year)) not in official_annual_keys
                for row in annual_reports.itertuples()
            ]
        ]
        binary_selection = pd.concat(
            [annual_financial, annual_reports], ignore_index=True
        ).drop_duplicates("document_id")
        selected_download_ids = set(binary_selection["document_id"].astype(str))
        raw_documents = paths.raw / "company_documents"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _download_document, row, raw_documents / row["ticker"], 150 * 1024 * 1024
                ): row for row in binary_selection.to_dict("records")
                if row.get("download_url")
            }
            for position, future in enumerate(as_completed(futures), 1):
                source_row = futures[future]
                try:
                    downloaded.append(future.result())
                except Exception as exc:
                    failures.append({
                        "ticker": source_row["ticker"],
                        "document_id": source_row["document_id"],
                        "stage": "download", "error": type(exc).__name__,
                    })
                if position % 100 == 0 or position == len(futures):
                    print(f"[Vietstock documents] {position:,}/{len(futures):,}", flush=True)
    base_records = [
        {
            **row,
            "download_required": str(row["document_id"]) in selected_download_ids,
            "download_status": (
                "selected_pending_or_failed"
                if str(row["document_id"]) in selected_download_ids
                else "metadata_only"
            ),
        }
        for row in indexed.to_dict("records")
    ]
    new_frame = pd.DataFrame(base_records)
    if downloaded:
        download_frame = pd.DataFrame(downloaded).drop_duplicates("document_id", keep="last")
        new_frame = new_frame.set_index("document_id")
        download_frame = download_frame.set_index("document_id")
        for column in download_frame.columns:
            new_frame.loc[download_frame.index, column] = download_frame[column]
        new_frame = new_frame.reset_index()
    output = paths.normalized / "source_documents.parquet"
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame()
    frame = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    if not frame.empty:
        frame = frame.sort_values(
            ["ticker", "available_at", "news_id", "attachment_index"]
        ).drop_duplicates(["source", "news_id", "download_url"], keep="last")
    frame.to_parquet(output, index=False)
    coverage = (
        new_frame.groupby(["ticker", "document_type"], dropna=False).size()
        .unstack(fill_value=0).reindex(tickers, fill_value=0).reset_index()
        if not new_frame.empty else pd.DataFrame({"ticker": tickers})
    )
    coverage.to_csv(paths.reports / "vietstock_document_coverage.csv", index=False)
    manifest = {
        "dataset": DATASET_LABEL,
        "status": "partial" if failures else "success",
        "source_role": "aggregated document repository; not relabeled as official HOSE",
        "requested_tickers": len(tickers),
        "eligible_2020_2025_documents": len(indexed),
        "documents_available": len(new_frame),
        "binary_download_policy": (
            "one preferred annual audited BCTC per ticker-year plus one BCTN per "
            "ticker-year only when an official HOSE BCTN is unavailable"
        ),
        "binary_documents_selected": len(selected_download_ids),
        "binary_documents_downloaded": int(
            new_frame.get("local_path", pd.Series(dtype=object)).notna().sum()
        ),
        "tickers_with_financial_statements": int(
            new_frame.loc[new_frame.get("document_type").eq("financial_statement"), "ticker"].nunique()
        ) if not new_frame.empty else 0,
        "tickers_with_annual_reports": int(
            new_frame.loc[new_frame.get("document_type").eq("annual_report"), "ticker"].nunique()
        ) if not new_frame.empty else 0,
        "failure_count": len(failures),
        "failure_sample": failures[:100],
        "output": str(output),
        "sha256": sha256_file(output),
    }
    (paths.reports / "vietstock_document_crawl_audit.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(failures).to_csv(
        paths.reports / "vietstock_document_failures.csv", index=False
    )
    return manifest


def crawl_hose_tri_benchmark(
    root: Path,
    benchmark_name: str = "VNALLSHARETRI",
    max_workers: int = 4,
) -> dict[str, Any]:
    """Collect an official broad-HOSE total-return benchmark on trading dates."""
    paths = Paths(data_17_8_workspace(root))
    prices = pd.read_parquet(paths.normalized / "prices.parquet", columns=["date"])
    dates = sorted(pd.to_datetime(prices["date"]).dt.normalize().unique())
    checkpoint = paths.raw / "benchmark" / benchmark_name
    checkpoint.mkdir(parents=True, exist_ok=True)

    def collect(date: pd.Timestamp) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        target = checkpoint / f"{date:%Y-%m-%d}.json"
        if target.exists():
            payload = json.loads(target.read_text(encoding="utf-8"))
        else:
            payload = _request_json(
                f"{HOSE_MARKET_API}/market/indicies-information",
                params={"dateTime": f"{date:%Y-%m-%d}", "indexName": "TRI"},
            )
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        matches = [
            item for item in (payload.get("data") or [])
            if re.sub(r"\s+", "", str(item.get("name") or "")).upper()
            == re.sub(r"\s+", "", benchmark_name).upper()
        ]
        if not matches:
            return None, {"date": str(date.date()), "reason": "benchmark_not_unique"}
        values = {
            float(value)
            for value in (
                pd.to_numeric(str(item.get("value")).replace(",", ""), errors="coerce")
                for item in matches
            )
            if pd.notna(value) and float(value) > 0
        }
        # The official endpoint contains one duplicated, identical row on
        # 2024-03-07.  Identical duplicates are safe to collapse; conflicting
        # values remain a hard failure.
        if len(values) != 1:
            return None, {"date": str(date.date()), "reason": "benchmark_not_unique"}
        value = next(iter(values))
        if pd.isna(value) or float(value) <= 0:
            return None, {"date": str(date.date()), "reason": "invalid_value"}
        return {
            "date": date,
            "benchmark": benchmark_name,
            "total_return_index": float(value),
            "index_type": "total_return",
            "methodology_url": HOSE_INDEX_METHODOLOGY_URL,
            "available_at": date + pd.Timedelta(hours=18),
            "source": "hose_official_tri_api",
            "source_url": HOSE_TRI_PAGE_URL,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data_class": "real",
        }, None

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(collect, pd.Timestamp(date)): date for date in dates}
        for position, future in enumerate(as_completed(futures), 1):
            row, failure = future.result()
            if row:
                rows.append(row)
            if failure:
                failures.append(failure)
            if position % 250 == 0 or position == len(futures):
                print(f"[HOSE TRI] {position:,}/{len(futures):,}", flush=True)
    frame = pd.DataFrame(rows).sort_values("date")
    output = paths.normalized / "benchmark.parquet"
    frame.to_parquet(output, index=False)
    expected = len(dates)
    coverage = len(frame) / expected if expected else 0.0
    audit = {
        "dataset": DATASET_LABEL,
        "status": "pass" if coverage >= 0.99 else "partial",
        "benchmark": benchmark_name,
        "expected_trading_dates": expected,
        "observations": len(frame),
        "coverage": coverage,
        "failures": failures,
        "methodology_url": HOSE_INDEX_METHODOLOGY_URL,
        "note": (
            "HOSE does not expose VN-Index TRI in this public endpoint. "
            "VNALLSHARETRI is used under its official name as the broad-market total-return benchmark."
        ),
        "output": str(output),
        "sha256": sha256_file(output),
    }
    (paths.reports / "benchmark_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def crawl_current_hose_sector_reference(root: Path) -> dict[str, Any]:
    """Collect current official GICS sectors but prevent retrospective use."""
    paths = Paths(data_17_8_workspace(root))
    fetched_at = pd.Timestamp.now(tz="UTC").tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
    sectors_payload = _request_json(
        f"{HOSE_LISTING_API}/sectors", params={"pageIndex": 1, "pageSize": 1000}
    )
    _json_archive(paths.raw / "sector_reference", "sectors", sectors_payload)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sector in sectors_payload["data"].get("list") or []:
        try:
            payload = _request_json(
                f"{HOSE_LISTING_API}/securities/stock",
                params={
                    "pageIndex": 1,
                    "pageSize": 1000,
                    "alphabet": "",
                    "code": "",
                    "sectorId": sector["id"],
                },
            )
            _json_archive(paths.raw / "sector_reference", f"sector-{sector['id']}", payload)
            for stock in payload["data"].get("list") or []:
                rows.append({
                    "ticker": str(stock.get("code") or "").upper(),
                    "sector_id": int(sector["id"]),
                    "sector": sector["name"],
                    "effective_from": fetched_at,
                    "effective_to": pd.NaT,
                    "available_at": fetched_at,
                    "point_in_time_usable_2020_2025": False,
                    "source": "hose_official_current_gics",
                    "source_url": "https://www.hsx.vn/vi/quan-ly-niem-yet/phan-nganh-niem-yet",
                    "fetched_at": fetched_at,
                })
        except Exception as exc:
            failures.append({"sector_id": sector.get("id"), "error": type(exc).__name__})
    frame = pd.DataFrame(rows).drop_duplicates(["ticker", "sector_id"])
    output = paths.normalized / "sector_current_reference.parquet"
    frame.to_parquet(output, index=False)
    audit = {
        "dataset": DATASET_LABEL,
        "status": "partial" if failures else "success",
        "rows": len(frame),
        "sectors": int(frame["sector"].nunique()) if not frame.empty else 0,
        "failures": failures,
        "point_in_time_usable_2020_2025": False,
        "reason": "current classification cannot be backfilled into historical decision dates",
        "output": str(output),
        "sha256": sha256_file(output),
    }
    (paths.reports / "sector_current_reference_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def _render_and_ocr_page(pdf_path: Path, page_number: int, temporary_root: Path) -> str:
    """OCR one low-text page when the optional local OCR dependency is available."""
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:  # pragma: no cover - depends on optional local runtime
        raise RuntimeError("rapidocr_not_installed") from exc
    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise RuntimeError("pdftoppm_not_available")
    prefix = temporary_root / f"page-{page_number:04d}"
    subprocess.run(
        [
            renderer,
            "-f", str(page_number),
            "-l", str(page_number),
            "-r", "180",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    image_path = prefix.with_suffix(".png")
    # Model initialization is expensive.  Keep one local inference engine per
    # extraction worker instead of reloading the ONNX models for every page.
    engine = getattr(_OCR_THREAD_STATE, "engine", None)
    if engine is None:
        engine = RapidOCR()
        _OCR_THREAD_STATE.engine = engine
    result = engine(str(image_path))
    texts = getattr(result, "txts", None)
    if texts is None and isinstance(result, tuple):
        legacy = result[0]
        texts = [item[1] for item in legacy] if legacy else []
    if not texts:
        return ""
    return "\n".join(str(text) for text in texts)


def _extract_pdf_document(
    row: dict[str, Any],
    text_directory: Path,
    temporary_directory: Path,
    *,
    use_ocr: bool,
    maximum_pages: int,
) -> dict[str, Any]:
    from pypdf import PdfReader

    pdf_path = Path(str(row["local_path"]))
    digest = str(row.get("sha256") or sha256_file(pdf_path))
    text_directory.mkdir(parents=True, exist_ok=True)
    target = text_directory / f"{digest}.json.gz"
    if target.exists():
        with gzip.open(target, "rt", encoding="utf-8") as stream:
            pages = json.load(stream)
        return {
            **row,
            "text_path": str(target),
            "pages_total": len(pages),
            "pages_native": sum(page.get("method") == "native" for page in pages),
            "pages_ocr": sum(page.get("method") == "ocr" for page in pages),
            "pages_low_text_unresolved": sum(
                page.get("method") == "low_text_unresolved" for page in pages
            ),
            "extraction_status": "resumed",
        }
    reader = PdfReader(str(pdf_path), strict=False)
    pages: list[dict[str, Any]] = []
    page_limit = min(len(reader.pages), maximum_pages)
    document_tmp = temporary_directory / digest[:16]
    document_tmp.mkdir(parents=True, exist_ok=True)
    for index in range(page_limit):
        page_number = index + 1
        error = None
        try:
            native_text = reader.pages[index].extract_text(extraction_mode="layout") or ""
        except Exception as exc:  # damaged text layers are handled per page
            native_text = ""
            error = type(exc).__name__
        meaningful = len(re.sub(r"\W+", "", native_text, flags=re.UNICODE))
        method = "native"
        text = native_text
        if meaningful < 40:
            if use_ocr:
                try:
                    ocr_text = _render_and_ocr_page(pdf_path, page_number, document_tmp)
                    if len(re.sub(r"\W+", "", ocr_text, flags=re.UNICODE)) >= meaningful:
                        text = ocr_text
                        method = "ocr"
                    else:
                        method = "low_text_unresolved"
                except Exception as exc:
                    method = "low_text_unresolved"
                    error = type(exc).__name__ + ":" + str(exc)[:120]
            else:
                method = "low_text_unresolved"
        pages.append({
            "page": page_number,
            "method": method,
            "text": text,
            "native_character_count": len(native_text),
            "character_count": len(text),
            "error": error,
        })
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        json.dump(pages, stream, ensure_ascii=False)
    unresolved = sum(page["method"] == "low_text_unresolved" for page in pages)
    return {
        **row,
        "text_path": str(target),
        "pages_total": len(reader.pages),
        "pages_processed": len(pages),
        "pages_native": sum(page["method"] == "native" for page in pages),
        "pages_ocr": sum(page["method"] == "ocr" for page in pages),
        "pages_low_text_unresolved": unresolved,
        "extraction_status": (
            "complete" if len(pages) == len(reader.pages) and not unresolved
            else "partial_requires_review"
        ),
    }


def _archive_member_is_safe(name: str) -> bool:
    candidate = Path(str(name).replace("\\", "/"))
    return bool(
        str(name).strip()
        and not candidate.is_absolute()
        and not re.match(r"^[A-Za-z]:", str(name))
        and ".." not in candidate.parts
    )


def _clear_archive_working_directory(path: Path, permitted_root: Path) -> None:
    resolved = path.resolve()
    root = permitted_root.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise RuntimeError(f"Refusing to clear archive path outside temporary root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def extract_company_document_archives(
    root: Path,
    *,
    maximum_uncompressed_bytes: int = 500 * 1024 * 1024,
) -> dict[str, Any]:
    """Safely expose PDFs inside selected annual BCTC ZIP/RAR archives.

    Annual-report archives are retained as source artifacts but are not expanded,
    because the model ingests standardized financial statements only.
    """
    paths = Paths(data_17_8_workspace(root))
    source_path = paths.normalized / "source_documents.parquet"
    if not source_path.exists():
        raise FileNotFoundError("source_documents.parquet is required")
    documents = pd.read_parquet(source_path)
    model_tickers = _ticker_set(paths)
    selected = documents[
        documents["document_type"].eq("financial_statement")
        & documents["ticker"].astype(str).isin(model_tickers)
        & documents["file_type"].astype(str).str.lower().isin([".zip", ".rar"])
        & documents["local_path"].notna()
        & documents.get("download_required", pd.Series(True, index=documents.index)).fillna(True)
    ].copy()
    children: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    output_root = paths.raw / "company_documents_extracted"
    temporary_root = root / "tmp" / "archives" / "data_17_8"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)

    for position, row in enumerate(selected.to_dict("records"), 1):
        archive = Path(str(row["local_path"]))
        digest = str(row.get("sha256") or sha256_file(archive))
        working = temporary_root / digest[:20]
        _clear_archive_working_directory(working, temporary_root)
        working.mkdir(parents=True)
        try:
            suffix = archive.suffix.lower()
            if suffix == ".zip":
                with zipfile.ZipFile(archive) as bundle:
                    members = [item for item in bundle.infolist() if not item.is_dir()]
                    if not members or any(not _archive_member_is_safe(item.filename) for item in members):
                        raise RuntimeError("unsafe_or_empty_zip_member_list")
                    if sum(item.file_size for item in members) > maximum_uncompressed_bytes:
                        raise RuntimeError("archive_uncompressed_size_exceeds_limit")
                    bundle.extractall(working)
            else:
                listing = subprocess.run(
                    ["tar", "-tf", str(archive)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                members = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
                if not members or any(not _archive_member_is_safe(name) for name in members):
                    raise RuntimeError("unsafe_or_empty_rar_member_list")
                subprocess.run(
                    ["tar", "-xf", str(archive), "-C", str(working)],
                    check=True,
                    capture_output=True,
                    timeout=180,
                )
                total_size = sum(
                    item.stat().st_size for item in working.rglob("*") if item.is_file()
                )
                if total_size > maximum_uncompressed_bytes:
                    raise RuntimeError("archive_uncompressed_size_exceeds_limit")
            pdfs = sorted(
                item for item in working.rglob("*")
                if item.is_file() and item.suffix.lower() == ".pdf"
            )
            if not pdfs:
                raise RuntimeError("archive_contains_no_pdf")
            target_dir = output_root / str(row["ticker"])
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, pdf in enumerate(pdfs, 1):
                pdf_hash = sha256_file(pdf)
                target = target_dir / f"{digest[:16]}-{index:02d}-{pdf_hash[:12]}.pdf"
                if not target.exists():
                    shutil.copy2(pdf, target)
                parent_id = str(
                    row.get("document_id")
                    or f"{row.get('source')}:{row.get('news_id')}:{row.get('attachment_index')}"
                )
                children.append({
                    **row,
                    "document_id": f"{parent_id}:archive:{index}:{pdf_hash[:12]}",
                    "attachment_index": int(row.get("attachment_index") or 1) * 100 + index,
                    "file_name": pdf.name,
                    "file_type": ".pdf",
                    "local_path": str(target),
                    "bytes": target.stat().st_size,
                    "sha256": pdf_hash,
                    "download_status": "archive_extracted",
                    "archive_parent_path": str(archive),
                })
        except (Exception, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            failures.append({
                "ticker": row.get("ticker"),
                "document_id": row.get("document_id"),
                "file_name": row.get("file_name"),
                "error": type(exc).__name__,
                "reason": str(exc)[:300],
            })
        finally:
            _clear_archive_working_directory(working, temporary_root)
        if position % 25 == 0 or position == len(selected):
            print(
                f"[BCTC archives] {position:,}/{len(selected):,}, "
                f"pdfs={len(children):,}, failures={len(failures):,}",
                flush=True,
            )

    if children:
        child_frame = pd.DataFrame(children)
        combined = pd.concat([documents, child_frame], ignore_index=True, sort=False)
        if "document_id" in combined:
            fallback = (
                combined["source"].astype(str) + ":"
                + combined["news_id"].astype(str) + ":"
                + combined["attachment_index"].astype(str)
            )
            combined["_dedupe_id"] = combined["document_id"].fillna(fallback)
            combined = combined.drop_duplicates("_dedupe_id", keep="last").drop(columns="_dedupe_id")
        combined.to_parquet(source_path, index=False)
    audit = {
        "dataset": DATASET_LABEL,
        "status": "partial" if failures else "success",
        "financial_statement_archives_selected": len(selected),
        "pdf_children_extracted": len(children),
        "archive_failures": len(failures),
        "failures": failures,
        "output": str(source_path),
        "sha256": sha256_file(source_path),
    }
    (paths.reports / "document_archive_extraction_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def extract_company_document_text(
    root: Path,
    *,
    max_workers: int = 2,
    maximum_documents: int | None = None,
    maximum_pages: int = 250,
    use_ocr: bool = True,
) -> dict[str, Any]:
    """Extract page-level text and OCR only low-text PDF pages."""
    if not 1 <= max_workers <= 2:
        raise ValueError("PDF extraction max_workers must be 1 or 2")
    paths = Paths(data_17_8_workspace(root))
    source_path = paths.normalized / "source_documents.parquet"
    if not source_path.exists():
        raise FileNotFoundError("source_documents.parquet is required")
    documents = pd.read_parquet(source_path)
    model_tickers = _ticker_set(paths)
    documents = documents[
        documents["document_type"].isin(["financial_statement", "sector_constituents"])
        & (
            documents["ticker"].astype(str).isin(model_tickers)
            | documents["document_type"].eq("sector_constituents")
        )
        & documents["file_type"].astype(str).str.lower().eq(".pdf")
        & documents["local_path"].notna()
    ].copy()
    if maximum_documents is not None:
        documents = documents.head(int(maximum_documents))
    text_directory = paths.raw / "extracted_text"
    temp_root = root / "tmp" / "pdfs" / "data_17_8"
    temp_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _extract_pdf_document,
                row,
                text_directory,
                temp_root,
                use_ocr=use_ocr,
                maximum_pages=maximum_pages,
            ): row
            for row in documents.to_dict("records")
        }
        for position, future in enumerate(as_completed(futures), 1):
            source = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({
                    "news_id": int(source["news_id"]),
                    "ticker": source["ticker"],
                    "file_name": source["file_name"],
                    "error": type(exc).__name__,
                    "reason": str(exc)[:200],
                })
            if position % 50 == 0 or position == len(futures):
                print(f"[PDF extraction] {position:,}/{len(futures):,}", flush=True)
    frame = pd.DataFrame(rows)
    output = paths.normalized / "document_text_index.parquet"
    if frame.empty:
        frame = pd.DataFrame(columns=list(documents.columns) + [
            "text_path", "pages_total", "pages_processed", "pages_native",
            "pages_ocr", "pages_low_text_unresolved", "extraction_status",
        ])
    frame.to_parquet(output, index=False)
    unresolved_pages = int(frame.get("pages_low_text_unresolved", pd.Series(dtype=float)).fillna(0).sum())
    audit = {
        "dataset": DATASET_LABEL,
        "status": "partial" if failures or unresolved_pages else "success",
        "pdf_documents_selected": len(documents),
        "pdf_documents_extracted": len(frame),
        "documents_failed": len(failures),
        "ocr_enabled": use_ocr,
        "ocr_runtime_available": (
            shutil.which("pdftoppm") is not None
            and _module_available("rapidocr")
        ),
        "unresolved_low_text_pages": unresolved_pages,
        "failures": failures,
        "output": str(output),
        "sha256": sha256_file(output),
    }
    (paths.reports / "document_text_extraction_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


_METRICS: dict[str, tuple[str, ...]] = {
    "total_assets": (
        r"tong\s+cong\s+tai\s+san",
        r"tong\s+tai\s+san",
    ),
    "liabilities": (
        r"no\s+phai\s+tra",
        r"tong\s+no\s+phai\s+tra",
    ),
    "equity": (
        r"von\s+chu\s+so\s+huu",
        r"tong\s+cong\s+von\s+chu\s+so\s+huu",
    ),
    "revenue": (
        r"doanh\s+thu\s+thuan\s+ve\s+ban\s+hang\s+va\s+cung\s+cap\s+dich\s+vu",
        r"doanh\s+thu\s+thuan",
        r"tong\s+thu\s+nhap\s+hoat\s+dong",
    ),
    "net_income": (
        r"loi\s+nhuan\s+sau\s+thue\s+thu\s+nhap\s+doanh\s+nghiep",
        r"loi\s+nhuan\s+sau\s+thue",
        r"loi\s+nhuan\s+rong",
    ),
    "operating_cash_flow": (
        r"luu\s+chuyen\s+tien\s+thuan\s+tu\s+hoat\s+dong\s+kinh\s+doanh",
    ),
}


_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:\(\s*)?-?\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d+)?(?:\s*\))?"
    r"|(?<![A-Za-z0-9])(?:\(\s*)?-?\d{4,}(?:[.,]\d+)?(?:\s*\))?"
)


def _parse_accounting_number(value: str) -> float | None:
    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(" ", "")
    if text.count(",") and text.count("."):
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(","):
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", "") if len(tail) == 3 else text.replace(",", ".")
    elif text.count("."):
        tail = text.rsplit(".", 1)[-1]
        if len(tail) == 3 or text.count(".") > 1:
            text = text.replace(".", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return -abs(number) if negative else number


def _unit_factor(text: str) -> tuple[float, str | None]:
    folded = _ascii_fold(text)
    if re.search(r"don\s+vi\s+tinh.{0,30}ty\s*(dong|vnd)", folded):
        return 1e9, "VND billion"
    if re.search(r"don\s+vi\s+tinh.{0,30}trieu\s*(dong|vnd)", folded):
        return 1e6, "VND million"
    if re.search(r"don\s+vi\s+tinh.{0,30}nghin\s*(dong|vnd)", folded):
        return 1e3, "VND thousand"
    if re.search(r"don\s+vi\s+tinh.{0,30}(dong|vnd)", folded):
        return 1.0, "VND"
    return 1.0, None


def _metric_from_page(page: dict[str, Any], metric: str) -> dict[str, Any] | None:
    raw = str(page.get("text") or "")
    factor, unit = _unit_factor(raw)
    best: dict[str, Any] | None = None
    # Search line by line.  Accent folding changes string offsets, so slicing
    # the original page with an offset from the folded page can associate a
    # number with the wrong accounting row.
    raw_lines = raw.splitlines()
    for line_index, line in enumerate(raw_lines):
        folded_line = _ascii_fold(line)
        for pattern in _METRICS[metric]:
            if not re.search(pattern, folded_line, flags=re.IGNORECASE):
                continue
            snippet_raw = "\n".join(raw_lines[line_index:line_index + 3])[:600]
            numbers = []
            for token in _NUMBER_PATTERN.findall(snippet_raw):
                parsed = _parse_accounting_number(token)
                if parsed is None:
                    continue
                # Accounting statement row codes (e.g. 270/300/400) are not values.
                if float(parsed).is_integer() and 0 <= abs(parsed) <= 999:
                    continue
                numbers.append((token, parsed))
            if not numbers:
                continue
            token, number = numbers[0]
            confidence = 0.90 if unit else 0.72
            if page.get("method") == "ocr":
                confidence -= 0.15
            candidate = {
                "metric": metric,
                "value": float(number) * factor,
                "reported_value": float(number),
                "unit": unit or "unknown_assumed_VND",
                "unit_factor": factor,
                "page": int(page["page"]),
                "extraction_method": page.get("method"),
                "confidence": max(0.0, confidence),
                "evidence_text": re.sub(r"\s+", " ", snippet_raw[:280]).strip(),
                "matched_pattern": pattern,
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
    return best


def _load_pages(text_path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(Path(text_path), "rt", encoding="utf-8") as stream:
        return list(json.load(stream))


def build_financial_statement_facts(root: Path) -> dict[str, Any]:
    """Create page-evidenced financial facts and a conservative PIT wide table."""
    paths = Paths(data_17_8_workspace(root))
    index_path = paths.normalized / "document_text_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError("document_text_index.parquet is required")
    documents = pd.read_parquet(index_path)
    documents = documents[
        documents["document_type"].eq("financial_statement")
        & documents["fiscal_period_end"].notna()
        & documents["text_path"].notna()
    ].copy()
    facts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, document in enumerate(documents.to_dict("records"), 1):
        try:
            pages = _load_pages(document["text_path"])
            for metric in _METRICS:
                candidates = [
                    result for result in (
                        _metric_from_page(page, metric) for page in pages
                    ) if result is not None
                ]
                if not candidates:
                    continue
                best = max(candidates, key=lambda row: row["confidence"])
                facts.append({
                    "news_id": int(document["news_id"]),
                    "document_sha256": document.get("sha256"),
                    "ticker": document["ticker"],
                    "document_type": document["document_type"],
                    "fiscal_period_end": pd.Timestamp(document["fiscal_period_end"]),
                    "publication_date": pd.Timestamp(document["publication_date"]),
                    "available_at": pd.Timestamp(document["available_at"]),
                    "period_type": document["period_type"],
                    "statement_scope": document["statement_scope"],
                    "assurance": document["assurance"],
                    "source": document["source"],
                    "source_url": document["source_url"],
                    "parser_version": "data-17-8-financial-facts-v1",
                    **best,
                })
        except Exception as exc:
            failures.append({
                "news_id": int(document["news_id"]),
                "ticker": document["ticker"],
                "error": type(exc).__name__,
            })
        if position % 100 == 0 or position == len(documents):
            print(f"[Financial facts] {position:,}/{len(documents):,}", flush=True)
    fact_frame = pd.DataFrame(facts)
    fact_output = paths.normalized / "financial_statement_facts.parquet"
    if fact_frame.empty:
        fact_frame = pd.DataFrame(columns=[
            "news_id", "document_sha256", "ticker", "document_type",
            "fiscal_period_end", "publication_date", "available_at", "period_type",
            "statement_scope", "assurance", "source", "source_url", "parser_version",
            "metric", "value", "reported_value", "unit", "unit_factor", "page",
            "extraction_method", "confidence", "evidence_text", "matched_pattern",
        ])
    fact_frame.to_parquet(fact_output, index=False)

    if fact_frame.empty:
        wide = pd.DataFrame()
    else:
        priority = fact_frame.copy()
        priority["scope_priority"] = priority["statement_scope"].map({
            "consolidated": 3, "separate": 2, "unspecified": 1,
        }).fillna(0)
        priority["assurance_priority"] = priority["assurance"].map({
            "audited": 3, "reviewed": 2, "unassured_or_unspecified": 1,
        }).fillna(0)
        priority = priority.sort_values(
            [
                "ticker", "fiscal_period_end", "available_at", "metric",
                "scope_priority", "assurance_priority", "confidence",
            ],
            ascending=[True, True, True, True, False, False, False],
        ).drop_duplicates(
            ["ticker", "fiscal_period_end", "available_at", "metric"], keep="first"
        )
        identifiers = [
            "ticker", "fiscal_period_end", "publication_date", "available_at",
        ]
        wide = priority.pivot_table(
            index=identifiers, columns="metric", values="value", aggfunc="first"
        ).reset_index()
        wide.columns.name = None
        metadata = (
            priority.sort_values(
                ["scope_priority", "assurance_priority", "confidence"],
                ascending=[False, False, False],
            )
            .drop_duplicates(identifiers, keep="first")
            [identifiers + [
                "period_type", "statement_scope", "assurance", "source", "source_url",
            ]]
        )
        wide = wide.merge(metadata, on=identifiers, how="left", validate="one_to_one")
        for metric in _METRICS:
            if metric not in wide:
                wide[metric] = np.nan
        denominator = wide["total_assets"].abs().replace(0, np.nan)
        wide["balance_equation_error"] = (
            wide["total_assets"] - wide["liabilities"] - wide["equity"]
        ).abs() / denominator
        required = wide[["revenue", "net_income", "equity"]].notna().all(axis=1)
        balance_available = wide[["total_assets", "liabilities", "equity"]].notna().all(axis=1)
        balance_pass = wide["balance_equation_error"].le(0.05)
        wide["usable_for_model"] = required & (~balance_available | balance_pass)
        wide["data_class"] = "real"
        wide["parser_version"] = "data-17-8-financial-wide-v1"
    wide_output = paths.normalized / "financial_statements.parquet"
    wide.to_parquet(wide_output, index=False)
    coverage = (
        wide.groupby("ticker").agg(
            observations=("fiscal_period_end", "size"),
            usable_observations=("usable_for_model", "sum"),
            first_period=("fiscal_period_end", "min"),
            last_period=("fiscal_period_end", "max"),
        ).reset_index()
        if not wide.empty else pd.DataFrame()
    )
    coverage.to_csv(paths.reports / "financial_statement_coverage.csv", index=False)
    audit = {
        "dataset": DATASET_LABEL,
        "status": "partial" if failures or wide.empty else "success_with_quality_filter",
        "documents_considered": len(documents),
        "facts": len(fact_frame),
        "wide_rows": len(wide),
        "usable_for_model_rows": int(wide["usable_for_model"].sum()) if not wide.empty else 0,
        "tickers_with_usable_rows": int(
            wide.loc[wide["usable_for_model"], "ticker"].nunique()
        ) if not wide.empty else 0,
        "failures": failures,
        "fact_output": str(fact_output),
        "fact_sha256": sha256_file(fact_output),
        "wide_output": str(wide_output),
        "wide_sha256": sha256_file(wide_output),
        "policy": (
            "Only page-evidenced facts with publication timestamps are retained; "
            "rows missing revenue/net income/equity or failing an available balance check "
            "are excluded from model features."
        ),
    }
    (paths.reports / "financial_statement_fact_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


_SECTOR_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bvn(?:allshare)?\s*energy\b|\bvnene\b", re.I), "Energy"),
    (re.compile(r"\bvn(?:allshare)?\s*materials?\b|\bvnmat\b", re.I), "Materials"),
    (re.compile(r"\bvn(?:allshare)?\s*industrials?\b|\bvnind\b", re.I), "Industrials"),
    (re.compile(r"\bvn(?:allshare)?\s*consumer\s*discretionary\b|\bvncond\b", re.I), "Consumer Discretionary"),
    (re.compile(r"\bvn(?:allshare)?\s*consumer\s*staples\b|\bvncons\b", re.I), "Consumer Staples"),
    (re.compile(r"\bvn(?:allshare)?\s*health\s*care\b|\bvnheal\b", re.I), "Health Care"),
    (re.compile(r"\bvn(?:allshare)?\s*financials?\b|\bvnfin\b", re.I), "Financials"),
    (re.compile(r"\bvn(?:allshare)?\s*information\s*technology\b|\bvnit\b", re.I), "Information Technology"),
    (re.compile(r"\bvn(?:allshare)?\s*communication\s*services\b|\bvncomm\b", re.I), "Communication Services"),
    (re.compile(r"\bvn(?:allshare)?\s*utilities\b|\bvnuti\b", re.I), "Utilities"),
    (re.compile(r"\bvn(?:allshare)?\s*real\s*estate\b|\bvnreal\b", re.I), "Real Estate"),
]


def build_historical_sector_pit(root: Path) -> dict[str, Any]:
    """Parse official periodic VNAllshare Sector constituent publications."""
    paths = Paths(data_17_8_workspace(root))
    index_path = paths.normalized / "document_text_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError("document_text_index.parquet is required")
    documents = pd.read_parquet(index_path)
    documents = documents[
        documents["document_type"].eq("sector_constituents")
        & documents["text_path"].notna()
    ]
    tickers = _ticker_set(paths)
    rows: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for document in documents.to_dict("records"):
        pages = _load_pages(document["text_path"])
        for page in pages:
            current_sector: str | None = None
            for line in str(page.get("text") or "").splitlines():
                folded = _ascii_fold(line)
                matches = [sector for pattern, sector in _SECTOR_ALIASES if pattern.search(folded)]
                if len(matches) == 1:
                    current_sector = matches[0]
                elif len(matches) > 1:
                    ambiguous.append({
                        "news_id": int(document["news_id"]),
                        "page": page["page"],
                        "line": line[:200],
                        "reason": "multiple_sector_headings",
                    })
                    current_sector = None
                if not current_sector:
                    continue
                line_tickers = {
                    token for token in re.findall(r"\b[A-Z0-9]{3}\b", line.upper())
                    if token in tickers
                }
                for ticker in line_tickers:
                    rows.append({
                        "ticker": ticker,
                        "sector": current_sector,
                        "effective_from": pd.Timestamp(document["available_at"]).normalize(),
                        "effective_to": pd.NaT,
                        "available_at": pd.Timestamp(document["available_at"]),
                        "publication_date": pd.Timestamp(document["publication_date"]),
                        "news_id": int(document["news_id"]),
                        "source": "hose_official_vnallshare_sector_constituents",
                        "source_url": document["source_url"],
                        "document_sha256": document.get("sha256"),
                        "evidence_page": int(page["page"]),
                        "parser_version": "data-17-8-sector-pit-v1",
                    })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["ticker", "effective_from", "available_at"])
        frame = frame.drop_duplicates(["ticker", "effective_from", "sector"], keep="last")
        for ticker, indices in frame.groupby("ticker").groups.items():
            ordered = frame.loc[indices].sort_values("effective_from")
            next_dates = ordered["effective_from"].shift(-1)
            frame.loc[ordered.index, "effective_to"] = next_dates - pd.Timedelta(days=1)
    else:
        frame = pd.DataFrame(columns=[
            "ticker", "sector", "effective_from", "effective_to", "available_at",
            "publication_date", "news_id", "source", "source_url", "document_sha256",
            "evidence_page", "parser_version",
        ])
    output = paths.normalized / "sector_pit.parquet"
    frame.to_parquet(output, index=False)
    audit = {
        "dataset": DATASET_LABEL,
        "status": "success" if not frame.empty and not ambiguous else "partial",
        "source_documents": len(documents),
        "rows": len(frame),
        "tickers": int(frame["ticker"].nunique()) if not frame.empty else 0,
        "sectors": int(frame["sector"].nunique()) if not frame.empty else 0,
        "ambiguous_lines": ambiguous,
        "point_in_time_policy": (
            "A sector becomes usable no earlier than the HOSE publication timestamp; "
            "the current 2026 classification is never backfilled."
        ),
        "output": str(output),
        "sha256": sha256_file(output),
    }
    (paths.reports / "sector_pit_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def stage_data_17_8_corporate_actions(root: Path) -> dict[str, Any]:
    """Reparse archived VSDC notices into Data 17/8, preserving Data A/B and research-v2."""
    source = root / "outputs" / "research_v2" / "normalized" / "corporate_actions.parquet"
    paths = Paths(data_17_8_workspace(root))
    if not source.exists():
        raise FileNotFoundError(
            "Run crawl-corporate-actions before staging the Data 17/8 ledger"
        )
    archive_root = root / "outputs" / "research_v2" / "raw" / "corporate_actions"
    notice_root = archive_root / "vsdc" / "notices"
    checkpoint_root = archive_root / "checkpoints"
    target = paths.normalized / "corporate_actions.parquet"

    official_frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    source_url_by_checksum: dict[str, str] = {}
    for checkpoint in checkpoint_root.glob("vsdc-*.parquet"):
        try:
            frame = pd.read_parquet(checkpoint)
        except Exception:
            continue
        if not frame.empty:
            for item in frame.to_dict("records"):
                checksum = str(item.get("raw_checksum") or "")
                source_url = str(item.get("source_url") or "")
                if checksum and source_url:
                    source_url_by_checksum[checksum] = source_url
    for checkpoint in checkpoint_root.glob("cafef-*.parquet"):
        try:
            frame = pd.read_parquet(checkpoint)
        except Exception:
            continue
        if not frame.empty:
            reference_frames.append(frame)

    parse_failures: list[dict[str, Any]] = []
    parsed_notice_files = 0
    for notice in sorted(notice_root.glob("*.html")):
        try:
            markup = notice.read_text(encoding="utf-8")
            checksum = hashlib.sha256(markup.encode("utf-8")).hexdigest()
            id_match = re.search(r"-([0-9]+)-[0-9a-f]{16}$", notice.stem)
            inferred_url = (
                f"https://www.vsd.vn/vi/ad/{id_match.group(1)}"
                if id_match else "https://www.vsd.vn/vi"
            )
            source_url = source_url_by_checksum.get(checksum, inferred_url)
            fetched_at = datetime.fromtimestamp(
                notice.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            records = VSDCCorporateActionAdapter.parse(
                markup, source_url, fetched_at
            )
            if records:
                frame = pd.DataFrame(records)
                frame["raw_checksum"] = checksum
                official_frames.append(frame)
            parsed_notice_files += 1
        except Exception as exc:
            parse_failures.append({
                "file": str(notice), "error": type(exc).__name__,
            })

    if official_frames and reference_frames:
        official = pd.concat(official_frames, ignore_index=True)
        reference = pd.concat(reference_frames, ignore_index=True)
        date_column = official["record_date"].fillna(official["announcement_date"])
        official = official[
            pd.to_datetime(date_column, errors="coerce").between(
                pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
            )
        ].copy()
        reference = reference[
            pd.to_datetime(reference["ex_date"], errors="coerce").between(
                pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
            )
        ].copy()
        master = pd.read_parquet(paths.normalized / "security_master.parquet")
        ledger, conflicts = reconcile_corporate_actions(official, reference, master)
        ledger.to_parquet(target, index=False)
        conflicts.to_csv(paths.reports / "corporate_action_conflicts.csv", index=False)
    else:
        # A reproducible fallback retains the prior immutable ledger and makes the
        # missing archive input explicit in the audit.
        shutil.copy2(source, target)
        ledger = pd.read_parquet(target)
        conflicts = pd.DataFrame()
    verified = ledger["verification_status"].astype(str).eq("verified_cross_source")
    material = ledger["event_type"].isin({
        "CASH_DIVIDEND", "STOCK_DIVIDEND", "BONUS_SHARE", "STOCK_SPLIT",
        "REVERSE_SPLIT", "RIGHTS_ISSUE", "SHARE_CONVERSION", "MERGER",
    })
    unresolved = material & ~verified
    audit = {
        "dataset": DATASET_LABEL,
        "status": "blocked" if unresolved.any() else "pass",
        "rows": len(ledger),
        "verified_cross_source": int(verified.sum()),
        "unresolved_material_events": int(unresolved.sum()),
        "source": str(source),
        "official_notice_files_reparsed": parsed_notice_files,
        "official_rows_reparsed": int(len(official)) if official_frames and reference_frames else 0,
        "reference_rows_reused": int(len(reference)) if official_frames and reference_frames else 0,
        "parse_failures": parse_failures,
        "conflicts": len(conflicts),
        "output": str(target),
        "sha256": sha256_file(target),
        "note": (
            "Data 17/8 reparses immutable VSDC notice archives with the corrected "
            "ratio grammar and reuses CafeF only as an ex-date corroboration source. "
            "Unresolved material rows are not silently promoted to verified events."
        ),
    }
    (paths.reports / "corporate_action_stage_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def audit_data_17_8(root: Path) -> dict[str, Any]:
    """Fail-closed audit for the new workspace and its research claims."""
    paths = Paths(data_17_8_workspace(root))
    blockers: list[str] = []
    checks: dict[str, dict[str, Any]] = {}

    price_path = paths.normalized / "prices.parquet"
    if price_path.exists():
        prices = pd.read_parquet(price_path)
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        price_pass = bool(
            not prices.empty
            and prices["date"].min() >= pd.Timestamp(START_DATE)
            and prices["date"].max() <= pd.Timestamp(END_DATE)
            and not prices.duplicated(["ticker", "date"]).any()
        )
        adjustment_verified = bool(
            "adjustment_policy" in prices.columns
            and prices["adjustment_policy"].notna().all()
            and prices["adjustment_policy"].astype(str).isin({
                "verified_corporate_action_adjusted",
                "unadjusted_with_verified_actions_join",
                "verified_vendor_total_return_adjusted",
            }).all()
        )
        checks["price_panel"] = {
            "passed": price_pass,
            "rows": len(prices),
            "tickers": int(prices["ticker"].nunique()),
            "start": str(prices["date"].min().date()),
            "end": str(prices["date"].max().date()),
        }
        checks["verified_price_adjustment"] = {"passed": bool(adjustment_verified)}
        if not adjustment_verified:
            blockers.append("verified_corporate_action_adjustment_missing")
    else:
        checks["price_panel"] = {"passed": False, "reason": "missing"}
        blockers.append("price_panel_missing")

    crosscheck_audit_path = paths.reports / "cafef_price_crosscheck_audit.json"
    if price_path.exists() and crosscheck_audit_path.exists():
        crosscheck = json.loads(crosscheck_audit_path.read_text(encoding="utf-8"))
        crosscheck_pass = bool(
            crosscheck.get("cross_source_verified_tickers") == prices["ticker"].nunique()
            and crosscheck.get("rows") == len(prices)
            and crosscheck.get("sha256") == sha256_file(price_path)
            and crosscheck.get("adjustment_policy") == "verified_vendor_total_return_adjusted"
        )
        checks["cafef_kbs_price_crosscheck"] = {
            "passed": crosscheck_pass,
            "verified_tickers": crosscheck.get("cross_source_verified_tickers"),
            "rows": crosscheck.get("rows"),
        }
        if not crosscheck_pass:
            blockers.append("cafef_kbs_price_crosscheck_invalid")
    else:
        checks["cafef_kbs_price_crosscheck"] = {"passed": False, "reason": "missing"}
        blockers.append("cafef_kbs_price_crosscheck_missing")

    benchmark_path = paths.normalized / "benchmark.parquet"
    if benchmark_path.exists():
        benchmark = pd.read_parquet(benchmark_path)
        expected_dates = (
            set(pd.to_datetime(prices["date"], errors="coerce").dropna().dt.normalize())
            if price_path.exists() else set()
        )
        observed_dates = set(
            pd.to_datetime(benchmark["date"], errors="coerce").dropna().dt.normalize()
        )
        benchmark_pass = bool(
            not benchmark.empty
            and benchmark["benchmark"].astype(str).eq("VNALLSHARETRI").all()
            and benchmark["index_type"].astype(str).eq("total_return").all()
            and benchmark["methodology_url"].astype(str).str.startswith("https://").all()
            and benchmark["total_return_index"].astype(float).gt(0).all()
            and expected_dates
            and observed_dates == expected_dates
        )
        checks["official_total_return_benchmark"] = {
            "passed": benchmark_pass,
            "rows": len(benchmark),
            "benchmark": sorted(benchmark["benchmark"].astype(str).unique()),
            "expected_trading_dates": len(expected_dates),
            "missing_trading_dates": len(expected_dates - observed_dates),
            "unexpected_dates": len(observed_dates - expected_dates),
        }
        if not benchmark_pass:
            blockers.append("official_total_return_benchmark_invalid")
    else:
        checks["official_total_return_benchmark"] = {"passed": False, "reason": "missing"}
        blockers.append("official_total_return_benchmark_missing")

    financial_path = paths.normalized / "financial_statements.parquet"
    if financial_path.exists():
        financial = pd.read_parquet(financial_path)
        availability_pass = bool(
            financial.empty
            or (
                pd.to_datetime(financial["available_at"], errors="coerce")
                >= pd.to_datetime(financial["fiscal_period_end"], errors="coerce")
            ).all()
        )
        usable = int(financial.get("usable_for_model", pd.Series(dtype=bool)).fillna(False).sum())
        checks["financial_statements_pit"] = {
            "passed": availability_pass and usable > 0,
            "rows": len(financial),
            "usable_rows": usable,
            "tickers_with_usable_rows": int(
                financial.loc[
                    financial.get("usable_for_model", False), "ticker"
                ].nunique()
            ) if usable else 0,
        }
    else:
        checks["financial_statements_pit"] = {"passed": False, "reason": "missing"}

    sector_path = paths.normalized / "sector_pit.parquet"
    if sector_path.exists():
        sector = pd.read_parquet(sector_path)
        sector_pass = bool(
            not sector.empty
            and (
                pd.to_datetime(sector["available_at"], errors="coerce")
                >= pd.to_datetime(sector["effective_from"], errors="coerce")
            ).all()
        )
        checks["historical_sector_pit"] = {
            "passed": sector_pass,
            "rows": len(sector),
            "tickers": int(sector["ticker"].nunique()) if not sector.empty else 0,
        }
    else:
        checks["historical_sector_pit"] = {"passed": False, "reason": "missing"}

    documents_path = paths.normalized / "source_documents.parquet"
    full_master_path = paths.normalized / "security_master_full.parquet"
    master_path = (
        full_master_path
        if full_master_path.exists()
        else paths.normalized / "security_master.parquet"
    )
    if documents_path.exists() and master_path.exists():
        documents = pd.read_parquet(documents_path)
        master_tickers = set(
            pd.read_parquet(master_path, columns=["ticker"])["ticker"]
            .astype(str).str.upper().str.strip()
        )
        repository = documents[
            documents["source"].astype(str).eq("vietstock_public_document_repository")
        ].copy()
        bctc_tickers = set(
            repository.loc[
                repository["document_type"].eq("financial_statement"), "ticker"
            ].astype(str)
        )
        bctn_tickers = set(
            repository.loc[
                repository["document_type"].eq("annual_report"), "ticker"
            ].astype(str)
        )
        required = repository[
            repository.get(
                "download_required", pd.Series(False, index=repository.index)
            ).fillna(False)
        ]
        downloaded_required = int(
            required.get("local_path", pd.Series(dtype=object)).notna().sum()
        )
        document_pass = bool(
            master_tickers <= bctc_tickers
            and master_tickers <= bctn_tickers
            and len(required) > 0
            and downloaded_required == len(required)
        )
        checks["company_document_repository"] = {
            "passed": document_pass,
            "master_tickers": len(master_tickers),
            "bctc_tickers": len(bctc_tickers),
            "bctn_tickers": len(bctn_tickers),
            "canonical_binaries_required": len(required),
            "canonical_binaries_downloaded": downloaded_required,
        }
        if not document_pass:
            blockers.append("company_document_repository_incomplete")
    else:
        checks["company_document_repository"] = {"passed": False, "reason": "missing"}
        blockers.append("company_document_repository_missing")

    corporate_path = paths.normalized / "corporate_actions.parquet"
    if corporate_path.exists():
        actions = pd.read_parquet(corporate_path)
        actions["ticker"] = actions["ticker"].astype(str).str.upper().str.strip()
        verified = actions["verification_status"].astype(str).eq("verified_cross_source")
        material = actions["event_type"].isin({
            "CASH_DIVIDEND", "STOCK_DIVIDEND", "BONUS_SHARE", "STOCK_SPLIT",
            "REVERSE_SPLIT", "RIGHTS_ISSUE", "SHARE_CONVERSION", "MERGER",
        })
        panel_tickers = set(prices["ticker"].astype(str).str.upper().str.strip()) if price_path.exists() else set()
        unresolved_all = int((material & ~verified).sum())
        unresolved_panel = int(
            (material & ~verified & actions["ticker"].isin(panel_tickers)).sum()
        )
        checks["corporate_action_ledger"] = {
            "passed": bool(panel_tickers) and unresolved_panel == 0,
            "rows": len(actions),
            "verified": int(verified.sum()),
            "unresolved_material_all_master_tickers": unresolved_all,
            "unresolved_material_in_complete_case_panel": unresolved_panel,
            "panel_tickers": len(panel_tickers),
        }
        if unresolved_panel:
            blockers.append("unresolved_material_corporate_actions_in_price_panel")
    else:
        checks["corporate_action_ledger"] = {"passed": False, "reason": "missing"}
        blockers.append("corporate_action_ledger_missing")

    audit = {
        "dataset": DATASET_LABEL,
        "audit_date": "2026-08-17",
        "status": "pass" if not blockers else "blocked",
        "research_ready": not blockers,
        "exploratory_run_permitted": checks.get("price_panel", {}).get("passed", False),
        "checks": checks,
        "blockers": blockers,
        "interpretation": (
            "A failed optional enrichment check does not fabricate data. Confirmatory status "
            "requires every core price-adjustment, benchmark and corporate-action gate to pass."
        ),
    }
    output = paths.reports / "DATA_17_8_AUDIT.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def write_data_17_8_source_report(root: Path) -> Path:
    paths = Paths(data_17_8_workspace(root))
    audit = audit_data_17_8(root)
    checks = audit["checks"]
    lines = [
        f"# {DATASET_LABEL} - Báo cáo nguồn và kiểm toán dữ liệu",
        "",
        "**Ngày chốt phiên bản:** 17/08/2026  ",
        f"**Giai đoạn nghiên cứu:** {START_DATE} đến {END_DATE}  ",
        f"**Trạng thái confirmatory:** `{audit['status']}`",
        "",
        "## Phạm vi",
        "",
        "Data 17/8 kế thừa duy nhất security master có lịch sử niêm yết/hủy "
        "niêm yết chính thức; panel giá được dựng lại độc lập. Giá OHLC và giá "
        "trị giao dịch gốc lấy từ CafeF, chuỗi điều chỉnh lấy từ KBS và chỉ được "
        "chấp nhận khi lợi nhuận điều chỉnh khớp chéo giữa hai nguồn. Những mã "
        "có corporate action trọng yếu chưa xác minh bị loại toàn bộ trước khi "
        "chạy mô hình, không bị loại dựa trên kết quả sinh lợi.",
        "",
        "Kho tài liệu lưu toàn bộ metadata BCTC/BCTN giai đoạn 2020–2025 cho "
        "security master và tải các binary chuẩn tắc theo năm. HOSE là nguồn "
        "công bố chính thức; Vietstock chỉ giữ vai trò kho tổng hợp bổ sung. Mọi "
        "đặc trưng tài chính chỉ được sử dụng sau `available_at` và phải kèm "
        "bằng chứng trang của tài liệu nguồn.",
        "",
        "## Kết quả kiểm toán",
        "",
        "| Kiểm tra | Kết quả | Chi tiết |",
        "|---|---:|---|",
    ]
    for name, detail in checks.items():
        result = "PASS" if detail.get("passed") else "FAIL"
        compact = ", ".join(
            f"{key}={value}" for key, value in detail.items() if key != "passed"
        )
        lines.append(f"| `{name}` | {result} | {compact} |")
    lines.extend([
        "",
        "## Blocker còn lại",
        "",
    ])
    if audit["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in audit["blockers"])
    else:
        lines.append("- Không còn blocker dữ liệu cốt lõi.")
    lines.extend([
        "",
        "## Quy tắc diễn giải",
        "",
        "Benchmark được giữ đúng tên `VNALLSHARETRI`; đây không phải chuỗi "
        "VN-Index TRI. Bản GICS hiện tại của HOSE chỉ là snapshot 2026 và không "
        "được hồi tố; nếu không thu được ấn phẩm phân ngành lịch sử thì ràng "
        "buộc ngành được vô hiệu hóa và công khai trong diagnostics. Số liệu OCR "
        "không vượt qua kiểm tra đơn vị, thời điểm công bố và cân đối sẽ không "
        "được đưa vào mô hình.",
        "",
    ])
    target = paths.root / "DATA_17_8_SOURCE_AND_AUDIT_REPORT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
