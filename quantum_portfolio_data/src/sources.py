from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .data_pipeline import (
    PRICE_COLUMNS, Paths, create_staging_run, promote_staged_file, sha256_file,
)


class SourceConfigurationError(RuntimeError):
    pass


def _secret_file_from_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SourceConfigurationError(f"{name} points to a missing file.")
    return path.read_text(encoding="utf-8").strip()


def _stage_and_promote_price_panel(paths: Paths, prices: pd.DataFrame, source: str) -> dict:
    """Validate a candidate panel in an isolated run before atomic promotion."""
    missing = sorted(set(PRICE_COLUMNS) - set(prices.columns))
    if missing:
        raise ValueError(f"Staged price panel missing columns: {missing}")
    duplicates = prices.duplicated(["date", "ticker"], keep=False)
    bad_ohlc = (
        (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
        | (prices[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    if duplicates.any() or bad_ohlc.any():
        raise ValueError(
            f"Staged panel failed validation: duplicates={int(duplicates.sum())}, "
            f"bad_ohlc={int(bad_ohlc.sum())}."
        )
    staging = create_staging_run(paths, source)
    parquet = staging / "prices.parquet"
    csv = staging / "prices.csv"
    prices.sort_values(["ticker", "date"]).to_parquet(parquet, index=False)
    prices.sort_values(["ticker", "date"]).to_csv(csv, index=False)
    manifest = {
        "source": source, "records": len(prices), "tickers": prices["ticker"].nunique(),
        "start": str(pd.to_datetime(prices["date"]).min().date()),
        "end": str(pd.to_datetime(prices["date"]).max().date()),
        "sha256": sha256_file(parquet), "validated": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    promoted = promote_staged_file(paths, parquet, "prices.parquet")
    promote_staged_file(paths, csv, "prices.csv")
    return {**manifest, "staging": str(staging), "promotion": promoted}


def _archive_raw_frame(directory: Path, ticker: str, frame: pd.DataFrame) -> dict:
    """Persist a content-addressed source payload without overwriting an earlier response."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = frame.to_json(orient="records", date_format="iso").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    target = directory / f"{ticker.upper()}-{digest[:16]}.json"
    if not target.exists():
        target.write_bytes(payload)
    return {"path": str(target), "sha256": digest, "records": len(frame)}


@dataclass
class RetryPolicy:
    attempts: int = 4
    backoff_seconds: float = 1.0
    timeout_seconds: float = 30.0


class JsonHttpClient:
    def __init__(self, retry: RetryPolicy | None = None):
        self.retry = retry or RetryPolicy()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "quantum-portfolio-research/0.1"

    def request(self, method: str, url: str, **kwargs) -> Any:
        last: Exception | None = None
        for attempt in range(self.retry.attempts):
            try:
                response = self.session.request(
                    method, url, timeout=self.retry.timeout_seconds, **kwargs
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt + 1 < self.retry.attempts:
                    time.sleep(self.retry.backoff_seconds * (2 ** attempt))
        # Avoid rendering URLs containing query-string API credentials.
        raise RuntimeError(
            f"Source request failed after retries ({type(last).__name__ if last else 'unknown'})."
        )


class SSIFastConnectAdapter:
    """Official SSI FastConnect Data v2 adapter.

    Documentation:
    https://guide.ssi.com.vn/ssi-products/fastconnect-data/api-specs
    Credentials are deliberately required and never guessed.
    """

    base_url = "https://fc-data.ssi.com.vn/api/v2"

    def __init__(self, consumer_id: str | None = None, consumer_secret: str | None = None):
        self.consumer_id = consumer_id or os.getenv("SSI_CONSUMER_ID")
        self.consumer_secret = consumer_secret or os.getenv("SSI_CONSUMER_SECRET")
        if not self.consumer_id or not self.consumer_secret:
            raise SourceConfigurationError(
                "SSI_CONSUMER_ID and SSI_CONSUMER_SECRET are required. Create an "
                "authorized key in SSI iBoard/FastConnect; the pipeline will not guess credentials."
            )
        self.http = JsonHttpClient()
        self._token: str | None = None

    def token(self) -> str:
        if not self._token:
            payload = self.http.request(
                "POST", f"{self.base_url}/Market/AccessToken",
                json={"consumerID": self.consumer_id, "consumerSecret": self.consumer_secret},
            )
            self._token = payload["data"]["accessToken"]
        return self._token

    def _get(self, path: str, params: dict[str, Any]) -> dict:
        return self.http.request(
            "GET", f"{self.base_url}/{path}", params=params,
            headers={"Authorization": f"Bearer {self.token()}"},
        )

    def securities(self, market: str = "HOSE") -> pd.DataFrame:
        rows = []
        page = 1
        while True:
            payload = self._get(
                "Market/Securities",
                {"market": market, "pageIndex": page, "pageSize": 1000},
            )
            batch = payload.get("data") or []
            rows.extend(batch)
            if len(batch) < 1000:
                break
            page += 1
        return pd.DataFrame(rows)

    def daily_ohlc(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        rows = []
        page = 1
        while True:
            payload = self._get(
                "Market/DailyOhlc",
                {
                    "symbol": symbol, "fromDate": start.replace("-", "/"),
                    "toDate": end.replace("-", "/"), "pageIndex": page, "pageSize": 1000,
                },
            )
            batch = payload.get("data") or []
            rows.extend(batch)
            if len(batch) < 1000:
                break
            page += 1
        return pd.DataFrame(rows)

    def index_components(self, index_code: str = "VN30") -> pd.DataFrame:
        payload = self._get(
            "Market/IndexComponents",
            {"indexCode": index_code, "pageIndex": 1, "pageSize": 1000},
        )
        return pd.DataFrame(payload.get("data") or [])


def normalize_ssi_ohlc(df: pd.DataFrame, ticker: str, source_url: str) -> pd.DataFrame:
    aliases = {
        "TradingDate": "date", "tradingDate": "date",
        "Open": "open", "open": "open", "OpenPrice": "open",
        "High": "high", "high": "high", "HighPrice": "high",
        "Low": "low", "low": "low", "LowPrice": "low",
        "Close": "close", "close": "close", "ClosePrice": "close",
        "Volume": "volume", "volume": "volume",
        "Value": "trading_value", "value": "trading_value",
    }
    out = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns}).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"SSI response missing normalized fields: {missing}")
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = ticker.upper()
    out["security_id"] = f"HOSE:{ticker.upper()}"
    out["adjusted_close"] = out["close"]
    out["trading_value"] = out.get("trading_value", out["volume"] * out["close"])
    raw = out.to_csv(index=False).encode()
    out["source"] = "ssi_fastconnect_official"
    out["source_url"] = source_url
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    out["available_at"] = out["date"] + pd.Timedelta(days=1)
    out["raw_checksum"] = hashlib.sha256(raw).hexdigest()
    out["parser_version"] = "ssi-v2-normalizer-v1"
    out["data_class"] = "real"
    out["adjustment_policy"] = "unverified"
    return out[PRICE_COLUMNS]


def crawl_ssi_stage1(paths: Paths, tickers: list[str], start: str, end: str) -> dict:
    adapter = SSIFastConnectAdapter()
    paths.ensure()
    frames = []
    failures = []
    raw_archives = []
    checkpoint_dir = paths.raw / "ssi_ohlcv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{adapter.base_url}/Market/DailyOhlc"
    for ticker in tickers:
        try:
            ticker = ticker.strip().upper()
            checkpoint = checkpoint_dir / f"{ticker}.parquet"
            if checkpoint.exists():
                cached = pd.read_parquet(checkpoint)
                if not cached.empty and pd.to_datetime(cached["date"]).max() >= pd.Timestamp(end) - pd.Timedelta(days=10):
                    frames.append(cached)
                    continue
            raw = adapter.daily_ohlc(ticker, start, end)
            raw_archives.append(_archive_raw_frame(paths.raw / "ssi_responses", ticker, raw))
            normalized = normalize_ssi_ohlc(raw, ticker, endpoint)
            normalized.to_parquet(checkpoint, index=False)
            frames.append(normalized)
        except Exception as exc:
            failures.append({"ticker": ticker, "error": type(exc).__name__})
    if not frames:
        raise RuntimeError(f"No SSI data collected. Failures: {failures}")
    prices = pd.concat(frames, ignore_index=True)
    promotion = _stage_and_promote_price_panel(paths, prices, "ssi")
    securities = adapter.securities("HOSE")
    symbol_col = "Symbol" if "Symbol" in securities else "symbol"
    name_col = "StockName" if "StockName" in securities else "stockName"
    master = pd.DataFrame({
        "ticker": securities[symbol_col].astype(str).str.upper(),
        "security_id": securities[symbol_col].astype(str).str.upper().map(lambda x: f"HOSE:{x}"),
        "company_name": securities[name_col].astype(str),
        "exchange": "HOSE", "industry": pd.NA, "sector": pd.NA,
        "listing_date": pd.NaT, "delisting_date": pd.NaT,
        "effective_from": pd.NaT, "effective_to": pd.NaT,
        "available_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
        "source": "ssi_fastconnect_official", "data_class": "real",
        "source_url": f"{adapter.base_url}/Market/Securities",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_checksum": promotion["sha256"], "history_method": "current_snapshot_only",
    })
    master.to_parquet(Path(promotion["staging"]) / "security_master_candidate.parquet", index=False)
    manifest = {
        "status": "partial" if failures else "success", "data_class": "real",
        "source": "ssi_fastconnect_official", "records": len(prices),
        "tickers_requested": len(tickers), "tickers_collected": prices.ticker.nunique(),
        "failures": failures, "raw_archives": raw_archives,
        "checkpoint_dir": str(checkpoint_dir), "promotion": promotion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.raw / "ssi_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class VietstockAdapter:
    """Vietstock Finance historical-price adapter.

    Authentication, when required, is loaded from a user-owned file outside Git.
    Secrets are never returned by the adapter or written to artifacts.
    """

    base_url = "https://finance.vietstock.vn"
    history_endpoint = f"{base_url}/data/gettradingresult"

    def __init__(
        self,
        retry: RetryPolicy | None = None,
        verify_tls: bool = True,
        exchange_id: int = 1,
    ):
        self.retry = retry or RetryPolicy()
        self.verify_tls = verify_tls
        self.exchange_id = exchange_id
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        })
        header_payload = _secret_file_from_env("VIETSTOCK_AUTH_HEADER_FILE")
        cookie_payload = _secret_file_from_env("VIETSTOCK_COOKIE_FILE")
        if header_payload:
            try:
                headers = json.loads(header_payload)
            except json.JSONDecodeError as exc:
                raise SourceConfigurationError(
                    "VIETSTOCK_AUTH_HEADER_FILE must contain a JSON object of request headers."
                ) from exc
            allowed = {"Accept", "Accept-Language", "Origin", "Referer", "X-Requested-With"}
            self.session.headers.update({k: str(v) for k, v in headers.items() if k in allowed})
            if "Cookie" in headers:
                self.session.headers["Cookie"] = str(headers["Cookie"])
        elif cookie_payload:
            self.session.headers["Cookie"] = cookie_payload

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        last: Exception | None = None
        for attempt in range(self.retry.attempts):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.retry.timeout_seconds,
                    verify=self.verify_tls,
                    **kwargs,
                )
                response.raise_for_status()
                if "login" in response.url.lower() and url != response.url:
                    raise SourceConfigurationError("Vietstock authentication expired.")
                return response
            except requests.RequestException as exc:
                last = exc
                if attempt + 1 < self.retry.attempts:
                    time.sleep(self.retry.backoff_seconds * (2 ** attempt))
        raise RuntimeError(f"Vietstock request failed after retries: {last}")

    @staticmethod
    def _anti_forgery_token(html: str) -> str:
        form = re.search(
            r"<form[^>]+id=[\"']?__CHART_AjaxAntiForgeryForm[\"']?.*?</form>",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not form:
            raise RuntimeError("Vietstock chart anti-forgery form was not found.")
        token = re.search(
            r"name=[\"']?__RequestVerificationToken[\"']?[^>]*"
            r"value=[\"']?([^\s\"'>]+)",
            form.group(0),
            flags=re.IGNORECASE,
        )
        if not token:
            raise RuntimeError("Vietstock anti-forgery token was not found.")
        return token.group(1)

    def daily_ohlc(
        self, symbol: str, start: str, end: str, page_size: int = 100
    ) -> pd.DataFrame:
        symbol = symbol.strip().upper()
        page_url = f"{self.base_url}/{symbol}/lich-su-giao-dich.htm"
        page = self._request("GET", page_url)
        token = self._anti_forgery_token(page.text)
        start_text = pd.Timestamp(start).strftime("%d/%m/%Y")
        end_text = pd.Timestamp(end).strftime("%d/%m/%Y")
        rows: list[dict[str, Any]] = []
        page_index = 1
        while True:
            response = self._request(
                "POST",
                self.history_endpoint,
                data={
                    "Code": symbol,
                    "OrderBy": "TradingDate",
                    "OrderDirection": "desc",
                    "PageIndex": str(page_index),
                    "PageSize": str(page_size),
                    "FromDate": start_text,
                    "ToDate": end_text,
                    "ExportType": "default",
                    "Cols": "TKLGD,TGTGD,VHTT,TGG,DC,TGPTG,KLGDKL,GTGDKL",
                    "ExchangeID": str(self.exchange_id),
                    "__RequestVerificationToken": token,
                },
                headers={"Referer": page_url, "X-Requested-With": "XMLHttpRequest"},
            )
            payload = response.json()
            batch = payload.get("Data") or []
            rows.extend(batch)
            total = int(payload.get("Rows") or len(rows))
            if not batch or len(rows) >= total or len(batch) < page_size:
                break
            page_index += 1
        return pd.DataFrame(rows)


def normalize_vietstock_ohlc(
    df: pd.DataFrame, ticker: str, source_url: str
) -> pd.DataFrame:
    aliases = {
        "TradingDate": "date",
        "OpenPrice": "open",
        "HighestPrice": "high",
        "LowestPrice": "low",
        "ClosePrice": "close",
        "AdjustPrice": "adjusted_close",
        "TotalVol": "volume",
        "TotalVal": "trading_value",
    }
    out = df.rename(columns=aliases).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Vietstock response missing normalized fields: {missing}")
    date_ms = out["date"].astype(str).str.extract(r"/Date\((\d+)", expand=False)
    out["date"] = pd.to_datetime(pd.to_numeric(date_ms), unit="ms").dt.normalize()
    for column in ["open", "high", "low", "close", "adjusted_close", "volume", "trading_value"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["ticker"] = ticker.upper()
    out["security_id"] = f"HOSE:{ticker.upper()}"
    out["adjusted_close"] = out.get("adjusted_close", out["close"]).fillna(out["close"])
    out["trading_value"] = out.get(
        "trading_value", out["volume"] * out["close"]
    ).fillna(out["volume"] * out["close"])
    raw = out.to_csv(index=False).encode()
    out["source"] = "vietstock_finance"
    out["source_url"] = source_url
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    # Conservatively expose an EOD observation to models on the next calendar day.
    out["available_at"] = out["date"] + pd.Timedelta(days=1)
    out["raw_checksum"] = hashlib.sha256(raw).hexdigest()
    out["parser_version"] = "vietstock-history-v1"
    out["data_class"] = "real"
    out["adjustment_policy"] = "unverified"
    return out[PRICE_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)


def crawl_vietstock_stage1(
    paths: Paths,
    tickers: list[str],
    start: str,
    end: str,
    verify_tls: bool = True,
) -> dict:
    adapter = VietstockAdapter(verify_tls=verify_tls)
    paths.ensure()
    frames = []
    failures = []
    raw_archives = []
    checkpoint_dir = paths.raw / "vietstock_ohlcv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for ticker in [item.strip().upper() for item in tickers if item.strip()]:
        try:
            checkpoint = checkpoint_dir / f"{ticker}.parquet"
            if checkpoint.exists():
                cached = pd.read_parquet(checkpoint)
                if not cached.empty and pd.to_datetime(cached["date"]).max() >= pd.Timestamp(end) - pd.Timedelta(days=10):
                    frames.append(cached)
                    continue
            raw_frame = adapter.daily_ohlc(ticker, start, end)
            if raw_frame.empty:
                raise RuntimeError("empty response")
            raw_archives.append(_archive_raw_frame(paths.raw / "vietstock_responses", ticker, raw_frame))
            normalized = normalize_vietstock_ohlc(
                raw_frame,
                ticker,
                f"{adapter.base_url}/{ticker}/lich-su-giao-dich.htm",
            )
            normalized.to_parquet(checkpoint, index=False)
            frames.append(normalized)
            time.sleep(1.0)
        except Exception as exc:
            failures.append({"ticker": ticker, "error": type(exc).__name__})
    if not frames:
        raise RuntimeError(f"No Vietstock data collected. Failures: {failures}")
    prices = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
    promotion = _stage_and_promote_price_panel(paths, prices, "vietstock")
    collected = sorted(prices["ticker"].unique())
    master = pd.DataFrame({
        "ticker": collected,
        "security_id": [f"HOSE:{ticker}" for ticker in collected],
        "company_name": collected,
        "exchange": "HOSE",
        "industry": pd.NA,
        "sector": pd.NA,
        "listing_date": pd.NaT,
        "delisting_date": pd.NaT,
        "effective_from": pd.Timestamp(start),
        "effective_to": pd.NaT,
        "available_at": pd.Timestamp(start),
        "source": "vietstock_finance",
        "source_url": adapter.base_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_checksum": promotion["sha256"], "history_method": "current_snapshot_only",
        "data_class": "real",
    })
    master.to_parquet(Path(promotion["staging"]) / "security_master_candidate.parquet", index=False)
    manifest = {
        "status": "partial" if failures else "success",
        "data_class": "real",
        "source": "vietstock_finance",
        "records": len(prices),
        "tickers_requested": len(tickers),
        "tickers_collected": len(collected),
        "date_start": str(prices["date"].min().date()),
        "date_end": str(prices["date"].max().date()),
        "failures": failures,
        "raw_archives": raw_archives,
        "checkpoint_dir": str(checkpoint_dir),
        "tls_verification": verify_tls,
        "promotion": promotion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.raw / "vietstock_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _normalize_vnstock_ohlc(
    df: pd.DataFrame, ticker: str, source_url: str
) -> pd.DataFrame:
    out = df.rename(columns={"time": "date"}).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"vnstock response missing normalized fields: {missing}")
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    # vnstock's Vietnam equity daily bars are expressed in thousands of VND.
    for column in ["open", "high", "low", "close"]:
        out[column] *= 1000.0
    out["ticker"] = ticker.upper()
    out["security_id"] = f"HOSE:{ticker.upper()}"
    out["adjusted_close"] = out["close"]
    out["trading_value"] = out["volume"] * out["close"]
    raw = out.to_csv(index=False).encode()
    out["source"] = "vnstock_kbs"
    out["source_url"] = source_url
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    out["available_at"] = out["date"] + pd.Timedelta(days=1)
    out["raw_checksum"] = hashlib.sha256(raw).hexdigest()
    out["parser_version"] = "vnstock-v4-kbs-v1"
    out["data_class"] = "real"
    out["adjustment_policy"] = "unverified"
    return (
        out[PRICE_COLUMNS]
        .dropna(subset=["date", "open", "high", "low", "close", "volume"])
        .drop_duplicates(["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def crawl_vnstock_hose(
    paths: Paths,
    start: str,
    end: str,
    max_tickers: int = 300,
    tickers: list[str] | None = None,
    pause_seconds: float = 3.4,
) -> dict:
    """Crawl HOSE daily bars with per-symbol checkpointing.

    FinanceDataReader supplies the current HOSE equity listing (ordered by the
    provider); vnstock/KBS supplies daily OHLCV. Existing valid checkpoints are
    reused, making the command safe to resume.
    """
    import FinanceDataReader as fdr
    from vnstock import Market

    paths.ensure()
    checkpoint_dir = paths.raw / "vnstock_ohlcv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    listing = fdr.StockListing("HOSE").copy()
    listing["Symbol"] = listing["Symbol"].astype(str).str.upper().str.strip()
    listing = listing[
        listing["Symbol"].str.fullmatch(r"[A-Z0-9]{3}")
    ].drop_duplicates("Symbol")
    if tickers:
        requested = [item.strip().upper() for item in tickers if item.strip()]
        symbols = [symbol for symbol in requested if symbol in set(listing["Symbol"])]
    else:
        symbols = listing["Symbol"].head(max_tickers).tolist()
    if not symbols:
        raise RuntimeError("No valid HOSE equity symbols were selected.")

    failures: list[dict[str, str]] = []
    completed: list[str] = []
    source_url = "https://github.com/thinh-vu/vnstock"
    market = Market()
    for position, symbol in enumerate(symbols, start=1):
        checkpoint = checkpoint_dir / f"{symbol}.parquet"
        try:
            if checkpoint.exists():
                cached = pd.read_parquet(checkpoint)
                cached_dates = pd.to_datetime(cached["date"])
                if (
                    not cached.empty
                    and cached_dates.max() >= pd.Timestamp(end) - pd.Timedelta(days=10)
                ):
                    completed.append(symbol)
                    print(f"[{position:03d}/{len(symbols):03d}] {symbol}: checkpoint")
                    continue
            last: Exception | None = None
            raw = pd.DataFrame()
            for attempt in range(4):
                try:
                    raw = market.equity(symbol=symbol).ohlcv(
                        start=start, end=end, interval="1D", count=5000, source="kbs"
                    )
                    if raw is None or raw.empty:
                        raise RuntimeError("empty response")
                    break
                except Exception as exc:
                    last = exc
                    if attempt < 3:
                        time.sleep(1.5 * (2 ** attempt))
            if raw is None or raw.empty:
                raise RuntimeError(str(last or "empty response"))
            normalized = _normalize_vnstock_ohlc(raw, symbol, source_url)
            normalized.to_parquet(checkpoint, index=False)
            completed.append(symbol)
            print(
                f"[{position:03d}/{len(symbols):03d}] {symbol}: "
                f"{len(normalized):,} rows "
                f"{normalized['date'].min().date()}..{normalized['date'].max().date()}"
            )
            time.sleep(pause_seconds)
        except Exception as exc:
            failures.append({"ticker": symbol, "error": str(exc)})
            print(f"[{position:03d}/{len(symbols):03d}] {symbol}: FAILED {exc}")
        progress = {
            "requested": len(symbols),
            "completed": completed,
            "failures": failures,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (paths.raw / "vnstock_progress.json").write_text(
            json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    frames = []
    for symbol in completed:
        checkpoint = checkpoint_dir / f"{symbol}.parquet"
        if checkpoint.exists():
            frames.append(pd.read_parquet(checkpoint))
    if not frames:
        raise RuntimeError(f"No vnstock data collected. Failures: {failures}")
    prices = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    promotion = _stage_and_promote_price_panel(paths, prices, "vnstock")

    selected_listing = listing[listing["Symbol"].isin(completed)].copy()
    selected_listing = selected_listing.set_index("Symbol").reindex(completed).reset_index()
    master = pd.DataFrame({
        "ticker": selected_listing["Symbol"],
        "security_id": selected_listing["Symbol"].map(lambda x: f"HOSE:{x}"),
        "company_name": selected_listing.get("Name", selected_listing["Symbol"]),
        "exchange": "HOSE",
        "industry": selected_listing.get("Industry", pd.Series(pd.NA, index=selected_listing.index)),
        "sector": pd.NA,
        "listing_date": pd.NaT,
        "delisting_date": pd.NaT,
        "effective_from": pd.Timestamp(start),
        "effective_to": pd.NaT,
        "available_at": pd.Timestamp(start),
        "source": "finance_datareader_hose_listing",
        "source_url": "https://github.com/FinanceData/FinanceDataReader",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_checksum": promotion["sha256"], "history_method": "current_snapshot_only",
        "data_class": "real",
    })
    master.to_parquet(Path(promotion["staging"]) / "security_master_candidate.parquet", index=False)

    coverage = prices.groupby("ticker").agg(
        records=("date", "size"),
        start=("date", "min"),
        end=("date", "max"),
        missing_close=("close", lambda values: int(values.isna().sum())),
    ).reset_index()
    coverage["calendar_span_days"] = (
        pd.to_datetime(coverage["end"]) - pd.to_datetime(coverage["start"])
    ).dt.days
    coverage.to_csv(paths.normalized / "vnstock_coverage.csv", index=False)
    manifest = {
        "status": "partial" if failures else "success",
        "data_class": "real",
        "source": "vnstock_kbs",
        "listing_source": "finance_datareader_hose",
        "records": len(prices),
        "tickers_requested": len(symbols),
        "tickers_collected": len(completed),
        "date_start": str(prices["date"].min().date()),
        "date_end": str(prices["date"].max().date()),
        "failures": failures,
        "checkpoint_dir": str(checkpoint_dir),
        "promotion": promotion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.raw / "vnstock_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _normalize_fdr_ohlc(
    df: pd.DataFrame, ticker: str, source_url: str
) -> pd.DataFrame:
    out = df.reset_index().rename(columns={
        "Date": "date",
        "index": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    })
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"FinanceDataReader response missing normalized fields: {missing}")
    # Yahoo/FDR labels these HOSE bars on the prior provider business date. A
    # calendar-day shift incorrectly places Monday sessions on Saturday; roll to
    # the next business day instead. The exact exchange calendar remains a source
    # limitation and is cross-checked against the official listing intervals.
    out["date"] = (
        pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
        + pd.offsets.BDay(1)
    )
    for column in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["ticker"] = ticker.upper()
    out["security_id"] = f"HOSE:{ticker.upper()}"
    out["adjusted_close"] = out.get("adjusted_close", out["close"]).fillna(out["close"])
    out["trading_value"] = out["volume"] * out["close"]
    raw = out.to_csv(index=False).encode()
    out["source"] = "finance_datareader_yahoo_hose"
    out["source_url"] = source_url
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    out["available_at"] = out["date"] + pd.Timedelta(days=1)
    out["raw_checksum"] = hashlib.sha256(raw).hexdigest()
    out["parser_version"] = "finance-datareader-yahoo-hose-v1"
    out["data_class"] = "real"
    out["adjustment_policy"] = "unverified"
    return (
        out[PRICE_COLUMNS]
        .dropna(subset=["date", "open", "high", "low", "close", "volume"])
        .drop_duplicates(["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def crawl_fdr_hose(
    paths: Paths,
    start: str,
    end: str,
    max_tickers: int = 300,
    tickers: list[str] | None = None,
) -> dict:
    import FinanceDataReader as fdr

    paths.ensure()
    checkpoint_dir = paths.raw / "fdr_ohlcv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    listing = fdr.StockListing("HOSE").copy()
    listing["Symbol"] = listing["Symbol"].astype(str).str.upper().str.strip()
    listing = listing[
        listing["Symbol"].str.fullmatch(r"[A-Z0-9]{3}")
    ].drop_duplicates("Symbol")
    if tickers:
        requested = [item.strip().upper() for item in tickers if item.strip()]
        symbols = [symbol for symbol in requested if symbol in set(listing["Symbol"])]
    else:
        symbols = listing["Symbol"].head(max_tickers).tolist()
    if not symbols:
        raise RuntimeError("No valid HOSE equity symbols were selected.")

    completed: list[str] = []
    failures: list[dict[str, str]] = []
    source_url = "https://github.com/FinanceData/FinanceDataReader"
    for position, symbol in enumerate(symbols, start=1):
        checkpoint = checkpoint_dir / f"{symbol}.parquet"
        try:
            if checkpoint.exists():
                cached = pd.read_parquet(checkpoint)
                if (
                    not cached.empty
                    and pd.to_datetime(cached["date"]).max()
                    >= pd.Timestamp(end) - pd.Timedelta(days=10)
                ):
                    completed.append(symbol)
                    print(f"[{position:03d}/{len(symbols):03d}] {symbol}: checkpoint")
                    continue
            last: Exception | None = None
            raw = pd.DataFrame()
            for attempt in range(4):
                try:
                    raw = fdr.DataReader(f"HOSE:{symbol}", start, end)
                    if raw is None or raw.empty:
                        raise RuntimeError("empty response")
                    break
                except Exception as exc:
                    last = exc
                    if attempt < 3:
                        time.sleep(2 ** attempt)
            if raw is None or raw.empty:
                raise RuntimeError(str(last or "empty response"))
            normalized = _normalize_fdr_ohlc(raw, symbol, source_url)
            normalized.to_parquet(checkpoint, index=False)
            completed.append(symbol)
            print(
                f"[{position:03d}/{len(symbols):03d}] {symbol}: "
                f"{len(normalized):,} rows "
                f"{normalized['date'].min().date()}..{normalized['date'].max().date()}"
            )
        except Exception as exc:
            failures.append({"ticker": symbol, "error": str(exc)})
            print(f"[{position:03d}/{len(symbols):03d}] {symbol}: FAILED {exc}")
        (paths.raw / "fdr_progress.json").write_text(
            json.dumps({
                "requested": len(symbols),
                "completed": completed,
                "failures": failures,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    frames = [
        pd.read_parquet(checkpoint_dir / f"{symbol}.parquet")
        for symbol in completed
        if (checkpoint_dir / f"{symbol}.parquet").exists()
    ]
    if not frames:
        raise RuntimeError(f"No FinanceDataReader data collected. Failures: {failures}")
    prices = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    promotion = _stage_and_promote_price_panel(paths, prices, "finance-datareader")
    selected = listing.set_index("Symbol").reindex(completed).reset_index()
    master = pd.DataFrame({
        "ticker": selected["Symbol"],
        "security_id": selected["Symbol"].map(lambda x: f"HOSE:{x}"),
        "company_name": selected.get("Name", selected["Symbol"]),
        "exchange": "HOSE",
        "industry": selected.get("Industry", pd.Series(pd.NA, index=selected.index)),
        "sector": pd.NA,
        "listing_date": pd.NaT,
        "delisting_date": pd.NaT,
        "effective_from": pd.Timestamp(start),
        "effective_to": pd.NaT,
        "available_at": pd.Timestamp(start),
        "source": "finance_datareader_hose_listing",
        "source_url": "https://github.com/FinanceData/FinanceDataReader",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_checksum": promotion["sha256"], "history_method": "current_snapshot_only",
        "data_class": "real",
    })
    master.to_parquet(Path(promotion["staging"]) / "security_master_candidate.parquet", index=False)
    coverage = prices.groupby("ticker").agg(
        records=("date", "size"),
        start=("date", "min"),
        end=("date", "max"),
        missing_close=("close", lambda values: int(values.isna().sum())),
    ).reset_index()
    coverage.to_csv(paths.normalized / "fdr_coverage.csv", index=False)
    manifest = {
        "status": "partial" if failures else "success",
        "data_class": "real",
        "source": "finance_datareader_yahoo_hose",
        "records": len(prices),
        "tickers_requested": len(symbols),
        "tickers_collected": len(completed),
        "date_start": str(prices["date"].min().date()),
        "date_end": str(prices["date"].max().date()),
        "failures": failures,
        "checkpoint_dir": str(checkpoint_dir),
        "promotion": promotion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.raw / "fdr_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def merge_hose_checkpoints(paths: Paths, target_tickers: int = 300) -> dict:
    """Merge FDR primary checkpoints with vnstock fallback checkpoints."""
    import FinanceDataReader as fdr

    paths.ensure()
    listing = fdr.StockListing("HOSE").copy()
    listing["Symbol"] = listing["Symbol"].astype(str).str.upper().str.strip()
    listing = listing[
        listing["Symbol"].str.fullmatch(r"[A-Z0-9]{3}")
    ].drop_duplicates("Symbol")
    fdr_dir = paths.raw / "fdr_ohlcv"
    vnstock_dir = paths.raw / "vnstock_ohlcv"
    candidates: list[dict[str, Any]] = []
    for symbol in listing["Symbol"]:
        fdr_path = fdr_dir / f"{symbol}.parquet"
        vnstock_path = vnstock_dir / f"{symbol}.parquet"
        if fdr_path.exists():
            path = fdr_path
            source = "finance_datareader_yahoo_hose"
        elif vnstock_path.exists():
            path = vnstock_path
            source = "vnstock_kbs"
        else:
            continue
        dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
        candidates.append({
            "ticker": symbol,
            "source": source,
            "records": len(dates),
            "start": dates.min(),
            "end": dates.max(),
        })
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame["ends_at_target"] = (
        candidate_frame["end"] >= pd.Timestamp("2025-12-25")
    )
    candidate_frame = candidate_frame.sort_values(
        ["ends_at_target", "start", "records"],
        ascending=[False, True, False],
    )
    selected = candidate_frame["ticker"].head(target_tickers).tolist()
    source_for = dict(zip(candidate_frame["ticker"], candidate_frame["source"]))
    if len(selected) < target_tickers:
        raise RuntimeError(
            f"Only {len(selected)} usable HOSE checkpoints; target is {target_tickers}."
        )

    frames = []
    for symbol in selected:
        directory = fdr_dir if source_for[symbol].startswith("finance") else vnstock_dir
        frame = pd.read_parquet(directory / f"{symbol}.parquet")
        frames.append(frame)
    prices = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    bad_ohlc = (
        (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
        | (prices[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    quarantine = prices.loc[bad_ohlc].copy()
    quarantine.to_csv(paths.normalized / "ohlc_quarantine.csv", index=False)
    prices = prices.loc[~bad_ohlc].copy()
    # Preserve the source adapter's availability timestamp. Replacing it with the
    # observation date would erase the observation/availability distinction.
    prices["available_at"] = pd.to_datetime(prices["available_at"], errors="coerce")
    paths.normalized.mkdir(parents=True, exist_ok=True)
    promotion = _stage_and_promote_price_panel(paths, prices, "merged-market-sources")

    selected_listing = listing.set_index("Symbol").reindex(selected).reset_index()
    first_price = prices.groupby("ticker").first().reindex(selected)
    master = pd.DataFrame({
        "ticker": selected_listing["Symbol"],
        "security_id": selected_listing["Symbol"].map(lambda x: f"HOSE:{x}"),
        "company_name": selected_listing.get("Name", selected_listing["Symbol"]),
        "exchange": "HOSE",
        "industry": selected_listing.get(
            "Industry", pd.Series(pd.NA, index=selected_listing.index)
        ),
        "sector": pd.NA,
        "listing_date": prices.groupby("ticker")["date"].min().reindex(selected).values,
        "delisting_date": pd.NaT,
        "effective_from": prices.groupby("ticker")["date"].min().reindex(selected).values,
        "effective_to": pd.NaT,
        "available_at": prices.groupby("ticker")["date"].min().reindex(selected).values,
        "source": [source_for[symbol] for symbol in selected],
        "source_url": first_price["source_url"].values,
        "fetched_at": first_price["fetched_at"].values,
        "raw_checksum": first_price["raw_checksum"].values,
        "history_method": "first_price_observation_proxy",
        "data_class": "real",
    })
    master.to_parquet(Path(promotion["staging"]) / "security_master_candidate.parquet", index=False)
    coverage = prices.groupby("ticker").agg(
        records=("date", "size"),
        start=("date", "min"),
        end=("date", "max"),
        missing_open=("open", lambda values: int(values.isna().sum())),
        missing_close=("close", lambda values: int(values.isna().sum())),
        nonpositive_close=("close", lambda values: int((values <= 0).sum())),
    ).reset_index()
    coverage["source"] = coverage["ticker"].map(source_for)
    coverage.to_csv(paths.normalized / "hose300_coverage.csv", index=False)

    overlap_stats = []
    overlap_symbols = sorted(
        set(path.stem for path in fdr_dir.glob("*.parquet"))
        & set(path.stem for path in vnstock_dir.glob("*.parquet"))
        & set(selected)
    )
    for symbol in overlap_symbols:
        left = pd.read_parquet(fdr_dir / f"{symbol}.parquet")[["date", "adjusted_close"]]
        right = pd.read_parquet(vnstock_dir / f"{symbol}.parquet")[["date", "adjusted_close"]]
        joined = left.merge(right, on="date", suffixes=("_fdr", "_vnstock"))
        if not joined.empty:
            denominator = joined["adjusted_close_vnstock"].abs().replace(0, pd.NA)
            ape = (
                (joined["adjusted_close_fdr"] - joined["adjusted_close_vnstock"]).abs()
                / denominator
            )
            overlap_stats.append({
                "ticker": symbol,
                "overlap_rows": len(joined),
                "median_abs_pct_diff": float(ape.median()),
                "mean_abs_pct_diff": float(ape.mean()),
            })
    pd.DataFrame(overlap_stats).to_csv(
        paths.normalized / "source_crosscheck.csv", index=False
    )
    manifest = {
        "status": "success",
        "data_class": "real",
        "records": len(prices),
        "tickers": len(selected),
        "date_start": str(prices["date"].min().date()),
        "date_end": str(prices["date"].max().date()),
        "source_counts": pd.Series(
            [source_for[symbol] for symbol in selected]
        ).value_counts().to_dict(),
        "selected_tickers": selected,
        "crosschecked_tickers": len(overlap_stats),
        "quarantined_ohlc_rows": len(quarantine),
        "promotion": promotion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.raw / "hose300_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


class TradingEconomicsAdapter:
    """Official Trading Economics API used only as a market-data cross-check.

    Historical market responses document OHLC but not volume, exchange listing events,
    corporate-action adjustment policy or total-return semantics. Consequently this
    adapter never promotes its output to the primary research price panel.
    """

    base_url = "https://api.tradingeconomics.com"
    documentation_url = "https://docs.tradingeconomics.com/markets/historical/"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TRADING_ECONOMICS_API_KEY")
        if not self.api_key:
            raise SourceConfigurationError(
                "TRADING_ECONOMICS_API_KEY is required for the official Trading Economics API."
            )
        self.http = JsonHttpClient()

    def stocks_by_country(self, country: str = "vietnam") -> pd.DataFrame:
        payload = self.http.request(
            "GET", f"{self.base_url}/markets/stocks/country/{country}",
            params={"c": self.api_key, "f": "json"},
        )
        return pd.DataFrame(payload or [])

    def historical(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        payload = self.http.request(
            "GET", f"{self.base_url}/markets/historical/{symbol}",
            params={"c": self.api_key, "d1": start, "d2": end, "f": "json"},
        )
        return pd.DataFrame(payload or [])


def normalize_trading_economics_ohlc(
    df: pd.DataFrame, ticker: str, provider_symbol: str
) -> pd.DataFrame:
    aliases = {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"}
    out = df.rename(columns=aliases).copy()
    required = {"date", "open", "high", "low", "close"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Trading Economics response missing fields: {missing}")
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["ticker"] = ticker.upper()
    out["security_id"] = f"HOSE:{ticker.upper()}"
    out["provider_symbol"] = provider_symbol
    out["available_at"] = out["date"] + pd.Timedelta(days=1)
    out["source"] = "trading_economics_official_api"
    out["source_url"] = TradingEconomicsAdapter.documentation_url
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    raw = out.to_csv(index=False).encode()
    out["raw_checksum"] = hashlib.sha256(raw).hexdigest()
    out["parser_version"] = "trading-economics-market-history-v1"
    out["data_class"] = "real_crosscheck"
    return out[[
        "date", "ticker", "security_id", "provider_symbol", "open", "high", "low", "close",
        "available_at", "source", "source_url", "fetched_at", "raw_checksum",
        "parser_version", "data_class",
    ]].dropna(subset=["date", "close"]).drop_duplicates(["date", "ticker"])


def crawl_trading_economics_crosscheck(
    paths: Paths, tickers: list[str], start: str, end: str
) -> dict:
    adapter = TradingEconomicsAdapter()
    paths.ensure()
    staging = create_staging_run(paths, "trading-economics-crosscheck")
    listing = adapter.stocks_by_country("vietnam")
    if listing.empty or not {"Ticker", "Symbol"} <= set(listing.columns):
        raise RuntimeError("Trading Economics did not return a usable Vietnam stock symbol map.")
    listing["Ticker"] = listing["Ticker"].astype(str).str.upper().str.strip()
    symbol_map = dict(zip(listing["Ticker"], listing["Symbol"].astype(str)))
    requested = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for ticker in requested:
        provider_symbol = symbol_map.get(ticker)
        if not provider_symbol:
            failures.append({"ticker": ticker, "error": "symbol_not_found_for_vietnam"})
            continue
        try:
            frame = normalize_trading_economics_ohlc(
                adapter.historical(provider_symbol, start, end), ticker, provider_symbol
            )
            if frame.empty:
                raise RuntimeError("empty response")
            frame.to_parquet(staging / f"{ticker}.parquet", index=False)
            frames.append(frame)
        except Exception as exc:
            failures.append({"ticker": ticker, "error": type(exc).__name__})
    if not frames:
        raise RuntimeError(f"No Trading Economics cross-check rows collected; failures={failures}")
    result = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    output = staging / "trading_economics_crosscheck.parquet"
    result.to_parquet(output, index=False)
    csv_output = paths.reports / "trading_economics_crosscheck.csv"
    result.to_csv(csv_output, index=False)
    manifest = {
        "status": "partial" if failures else "success", "role": "crosscheck_only",
        "source": "trading_economics_official_api", "records": len(result),
        "tickers_requested": len(requested), "tickers_collected": result["ticker"].nunique(),
        "failures": failures, "dataset_sha256": sha256_file(output),
        "documentation_url": adapter.documentation_url,
        "limitations": [
            "no_volume", "no_verified_adjustment_policy", "no_listing_history",
            "not_a_total_return_benchmark",
        ],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (paths.raw / "trading_economics_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


class CafeFPublicHistoryAdapter:
    """Public CafeF price-history endpoint used only as a last-resort gap source."""

    endpoint = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    page_url = "https://cafef.vn/du-lieu/lich-su-giao-dich-{ticker}-1.chn"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (compatible; academic-research/0.1)"

    def daily_ohlc(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        symbol = symbol.upper().strip()
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        rows: list[dict[str, Any]] = []
        # The public UI is reliable only for bounded date ranges.
        left = start_ts
        while left <= end_ts:
            # The web UI enforces a roughly three-month window even if a longer
            # range is sent to the endpoint. Chunk explicitly to avoid silent loss.
            right = min(end_ts, left + pd.Timedelta(days=79))
            page_index = 1
            chunk_rows = 0
            while True:
                response = self.session.get(
                    self.endpoint,
                    params={
                        "ExchangeType": "HOSE", "Symbol": symbol,
                        "StartDate": left.strftime("%m/%d/%Y"),
                        "EndDate": right.strftime("%m/%d/%Y"),
                        "PageIndex": page_index, "PageSize": 40,
                    },
                    headers={"Referer": self.page_url.format(ticker=symbol.lower())},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                data = (payload or {}).get("Data") or {}
                batch = data.get("Data") or []
                rows.extend(batch)
                chunk_rows += len(batch)
                total = int(data.get("TotalCount") or 0)
                # The endpoint currently returns 20 records even when PageSize is
                # larger, so termination must use observed rows, not requested size.
                if not batch or chunk_rows >= total:
                    break
                page_index += 1
            left = right + pd.Timedelta(days=1)
        return pd.DataFrame(rows)


def _normalize_cafef_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    aliases = {
        "Ngay": "date", "GiaMoCua": "open", "GiaCaoNhat": "high",
        "GiaThapNhat": "low", "GiaDongCua": "close", "GiaDieuChinh": "adjusted_close",
        "KhoiLuongKhopLenh": "volume", "GiaTriKhopLenh": "trading_value",
    }
    out = df.rename(columns=aliases).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"CafeF response missing normalized fields: {missing}")
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "adjusted_close"]:
        out[column] = pd.to_numeric(out.get(column), errors="coerce") * 1000.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    # CafeF reports matched value in billions of VND.
    reported_value = pd.to_numeric(out.get("trading_value"), errors="coerce") * 1_000_000_000.0
    out["trading_value"] = reported_value.fillna(out["volume"] * out["close"])
    out["ticker"] = ticker.upper()
    out["security_id"] = f"HOSE:{ticker.upper()}"
    out["adjusted_close"] = out["adjusted_close"].fillna(out["close"])
    out["source"] = "cafef_public_history"
    out["source_url"] = CafeFPublicHistoryAdapter.page_url.format(ticker=ticker.lower())
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    out["available_at"] = out["date"] + pd.Timedelta(days=1)
    raw = out.to_csv(index=False).encode("utf-8")
    out["raw_checksum"] = hashlib.sha256(raw).hexdigest()
    out["parser_version"] = "cafef-public-history-v1"
    out["data_class"] = "real"
    out["adjustment_policy"] = "unverified"
    return out[PRICE_COLUMNS].dropna(
        subset=["date", "open", "high", "low", "close", "volume"]
    ).drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"])


DEFAULT_CAFEF_SYSTEM_TICKERS = (
    "VCB", "BID", "CTG", "MBB", "HPG", "FPT", "VNM", "VIC",
    "GAS", "MSN", "MWG", "SSI",
)


def crawl_cafef_standalone_workspace(
    paths: Paths,
    start: str,
    end: str,
    tickers: list[str] | None = None,
    max_workers: int = 3,
    workspace_name: str | None = None,
) -> tuple[Path, dict]:
    """Build a versioned CafeF-only price workspace without promoting it.

    CafeF is treated as an aggregated public reference source. Each source response is
    content-addressed, the official HOSE security master supplies identity/listing
    intervals, and failed symbols remain explicit in the manifest. This function never
    changes the canonical normalized panel.
    """
    master_all = _historically_relevant_hose_master(paths, start, end)
    requested_input = (
        sorted(master_all["ticker"].astype(str).unique())
        if tickers is None else tickers
    )
    requested = list(dict.fromkeys(
        str(ticker).upper().strip()
        for ticker in requested_input
        if str(ticker).strip()
    ))
    if len(requested) < 8:
        raise ValueError("A standalone CafeF system panel requires at least 8 tickers.")
    if max_workers < 1 or max_workers > 4:
        raise ValueError("max_workers must be between 1 and 4 to limit source load.")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    master = master_all
    master = master[master["ticker"].isin(requested)].copy()
    master_tickers = set(master["ticker"])
    unknown = sorted(set(requested) - master_tickers)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(
        ("|".join(requested) + str(start_ts.date()) + str(end_ts.date())).encode("utf-8")
    ).hexdigest()[:10]
    if workspace_name:
        if not re.fullmatch(r"[\w .-]+", workspace_name, flags=re.UNICODE):
            raise ValueError("workspace_name may contain only letters, numbers, spaces, dots and dashes")
        workspace = paths.root / "outputs" / workspace_name
    else:
        workspace = paths.root / "outputs" / "cafef_workspaces" / f"{stamp}-{digest}"
    target = Paths(workspace)
    target.ensure()
    raw_response_dir = target.raw / "responses"
    checkpoint_dir = target.raw / "normalized_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def collect(symbol: str) -> dict[str, Any]:
        raw = CafeFPublicHistoryAdapter().daily_ohlc(symbol, start, end)
        if raw.empty:
            raise RuntimeError("CafeF returned no observations")
        archive = _archive_raw_frame(raw_response_dir, symbol, raw)
        normalized = _normalize_cafef_ohlc(raw, symbol)
        candidates = master[master["ticker"].eq(symbol)]
        if len(candidates) != 1:
            raise RuntimeError("official security identity is missing or ambiguous")
        security = candidates.iloc[0]
        active_end = (
            min(end_ts, security["delisting_date"])
            if pd.notna(security["delisting_date"]) else end_ts
        )
        normalized = normalized[
            pd.to_datetime(normalized["date"]).between(
                max(start_ts, security["listing_date"]), active_end
            )
        ].copy()
        if normalized.empty:
            raise RuntimeError("CafeF observations do not overlap the official HOSE interval")
        normalized["security_id"] = security["security_id"]
        normalized["raw_checksum"] = archive["sha256"]
        normalized.to_parquet(checkpoint_dir / f"{symbol}.parquet", index=False)
        return {
            "ticker": symbol, "records": int(len(normalized)),
            "start": str(pd.to_datetime(normalized["date"]).min().date()),
            "end": str(pd.to_datetime(normalized["date"]).max().date()),
            "raw_response": archive,
        }

    existing_checkpoints = {
        path.stem.upper(): path for path in checkpoint_dir.glob("*.parquet")
    }
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = [
        {"ticker": ticker, "error": "not_in_official_hose_master"} for ticker in unknown
    ]
    collectable = [
        ticker for ticker in requested
        if ticker in master_tickers and ticker not in existing_checkpoints
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(collect, ticker): ticker for ticker in collectable}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                item = future.result()
                completed.append(item)
                print(
                    f"[CafeF {len(completed):02d}/{len(collectable):02d}] "
                    f"{ticker}: {item['records']:,} rows",
                    flush=True,
                )
            except Exception as exc:
                failures.append({
                    "ticker": ticker,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                })
                print(f"[CafeF] {ticker}: rejected ({type(exc).__name__})", flush=True)

    checkpoint_files = sorted(checkpoint_dir.glob("*.parquet"))
    if not checkpoint_files:
        manifest = {
            "status": "rejected", "source": "cafef_public_history",
            "requested_tickers": requested, "failures": failures,
            "start": start, "end": end,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (target.raw / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return workspace, manifest

    prices = pd.concat(
        [pd.read_parquet(path) for path in checkpoint_files], ignore_index=True
    ).sort_values(["ticker", "date"])
    target_master = master[master["ticker"].isin(prices["ticker"].unique())].copy()
    prices.to_parquet(target.normalized / "prices.parquet", index=False)
    target_master.to_parquet(target.normalized / "security_master.parquet", index=False)
    actions_path = paths.normalized / "corporate_actions.parquet"
    if actions_path.exists():
        # The current table may be empty; copying it preserves the schema without
        # claiming CafeF supplied or verified corporate actions.
        target_actions = pd.read_parquet(actions_path)
        if not target_actions.empty and "ticker" in target_actions:
            target_actions = target_actions[target_actions["ticker"].isin(target_master["ticker"])]
        target_actions.to_parquet(target.normalized / "corporate_actions.parquet", index=False)

    manifest = {
        "status": "collected_pending_quality_gate",
        "source": "cafef_public_history",
        "source_role": "standalone_exploratory_price_panel",
        "requested_tickers": requested,
        "requested_count": len(requested),
        "collected_count": int(prices["ticker"].nunique()),
        "resumed_checkpoint_count": int(len(existing_checkpoints)),
        "newly_collected_count": int(len(completed)),
        "records": int(len(prices)),
        "start": str(pd.to_datetime(prices["date"]).min().date()),
        "end": str(pd.to_datetime(prices["date"]).max().date()),
        "completed": sorted(completed, key=lambda row: row["ticker"]),
        "failures": sorted(failures, key=lambda row: row["ticker"]),
        "price_dataset_sha256": sha256_file(target.normalized / "prices.parquet"),
        "security_master_sha256": sha256_file(target.normalized / "security_master.parquet"),
        "adjustment_policy": "unverified",
        "limitations": [
            "aggregated_public_reference_source_not_exchange_official",
            "corporate_action_adjustment_semantics_not_certified",
            "full_period_sample_selection_is_exploratory",
            "not_a_full_hose_panel",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (target.raw / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return workspace, manifest


def _historically_relevant_hose_master(
    paths: Paths, start: str, end: str
) -> pd.DataFrame:
    """Return official HOSE identities active at any time in the requested interval."""
    master_path = paths.normalized / "security_master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(
            "Official security_master.parquet is required; run crawl-hose-security-master first."
        )
    master = pd.read_parquet(master_path).copy()
    if not master.get("history_method", pd.Series(dtype=str)).astype(str).isin(
        {"exchange_listing_history", "official_event_history"}
    ).all():
        raise ValueError("Security master is not entirely backed by official event history.")
    master["ticker"] = master["ticker"].astype(str).str.upper().str.strip()
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], errors="coerce")
    relevant = master[
        master["ticker"].str.fullmatch(r"[A-Z0-9]{3}")
        & master["listing_date"].le(pd.Timestamp(end))
        & (master["delisting_date"].isna() | master["delisting_date"].ge(pd.Timestamp(start)))
    ].copy()
    if relevant.empty:
        raise RuntimeError("Official HOSE master has no securities in the requested interval.")
    return relevant.sort_values(["ticker", "listing_date", "security_id"])


def crawl_historical_hose_price_gaps(
    paths: Paths,
    start: str,
    end: str,
    try_vnstock_fallback: bool = True,
    pause_seconds: float = 0.35,
) -> dict:
    """Checkpoint missing historical HOSE symbols without replacing the primary panel.

    The official HOSE event master defines the requested symbols. FinanceDataReader/Yahoo
    is attempted first and vnstock/KBS is an optional fallback. Collection never promotes
    a partial subset; use ``merge-historical-price-checkpoints`` afterwards.
    """
    import FinanceDataReader as fdr

    paths.ensure()
    master = _historically_relevant_hose_master(paths, start, end)
    requested = sorted(master["ticker"].unique())
    fdr_dir = paths.raw / "fdr_ohlcv"
    vnstock_dir = paths.raw / "vnstock_ohlcv"
    cafef_dir = paths.raw / "cafef_ohlcv"
    fdr_dir.mkdir(parents=True, exist_ok=True)
    vnstock_dir.mkdir(parents=True, exist_ok=True)
    cafef_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.stem.upper() for path in fdr_dir.glob("*.parquet")} | {
        path.stem.upper() for path in vnstock_dir.glob("*.parquet")
    } | {path.stem.upper() for path in cafef_dir.glob("*.parquet")}
    missing = [symbol for symbol in requested if symbol not in existing]
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    fdr_url = "https://github.com/FinanceData/FinanceDataReader"

    for position, symbol in enumerate(missing, start=1):
        error_messages: list[str] = []
        try:
            raw = fdr.DataReader(f"HOSE:{symbol}", start, end)
            if raw is None or raw.empty:
                raise RuntimeError("empty response")
            normalized = _normalize_fdr_ohlc(raw, symbol, fdr_url)
            normalized.to_parquet(fdr_dir / f"{symbol}.parquet", index=False)
            completed.append({"ticker": symbol, "source": "finance_datareader_yahoo_hose", "records": len(normalized)})
            print(f"[{position:03d}/{len(missing):03d}] {symbol}: FDR {len(normalized):,} rows")
            continue
        except Exception as exc:
            error_messages.append(f"fdr:{type(exc).__name__}")

        if try_vnstock_fallback:
            try:
                from vnstock import Market

                # Guest access is documented by the provider as 20 requests/minute.
                # Pace calls conservatively and preserve every completed symbol.
                if pause_seconds > 0:
                    time.sleep(max(pause_seconds, 3.2))
                raw = Market().equity(symbol=symbol).ohlcv(
                    start=start, end=end, interval="1D", count=5000, source="kbs"
                )
                if raw is None or raw.empty:
                    raise RuntimeError("empty response")
                normalized = _normalize_vnstock_ohlc(
                    raw, symbol, "https://github.com/thinh-vu/vnstock"
                )
                normalized.to_parquet(vnstock_dir / f"{symbol}.parquet", index=False)
                completed.append({"ticker": symbol, "source": "vnstock_kbs", "records": len(normalized)})
                print(f"[{position:03d}/{len(missing):03d}] {symbol}: vnstock {len(normalized):,} rows")
                continue
            except (Exception, SystemExit) as exc:
                error_messages.append(f"vnstock:{type(exc).__name__}")
        try:
            raw = CafeFPublicHistoryAdapter().daily_ohlc(symbol, start, end)
            if raw.empty:
                raise RuntimeError("empty response")
            _archive_raw_frame(paths.raw / "cafef_responses", symbol, raw)
            normalized = _normalize_cafef_ohlc(raw, symbol)
            normalized.to_parquet(cafef_dir / f"{symbol}.parquet", index=False)
            completed.append({"ticker": symbol, "source": "cafef_public_history", "records": len(normalized)})
            print(f"[{position:03d}/{len(missing):03d}] {symbol}: CafeF {len(normalized):,} rows")
            continue
        except Exception as exc:
            error_messages.append(f"cafef:{type(exc).__name__}")
        failures.append({"ticker": symbol, "error": ";".join(error_messages)})
        print(f"[{position:03d}/{len(missing):03d}] {symbol}: unavailable")

    manifest = {
        "status": "partial" if failures else "success",
        "role": "checkpoint_only",
        "requested_historical_tickers": len(requested),
        "already_checkpointed": len(existing & set(requested)),
        "newly_collected": len(completed),
        "completed": completed,
        "failures": failures,
        "start": start,
        "end": end,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "note": "No normalized price panel was promoted by this command.",
    }
    (paths.raw / "historical_price_gap_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def merge_historical_hose_checkpoints(paths: Paths, start: str, end: str) -> dict:
    """Merge every available historically relevant checkpoint using official identities."""
    paths.ensure()
    master = _historically_relevant_hose_master(paths, start, end)
    fdr_dir = paths.raw / "fdr_ohlcv"
    vnstock_dir = paths.raw / "vnstock_ohlcv"
    cafef_dir = paths.raw / "cafef_ohlcv"
    frames: list[pd.DataFrame] = []
    source_for: dict[str, str] = {}
    missing: list[str] = []
    identity_ambiguous: list[str] = []
    for symbol in sorted(master["ticker"].unique()):
        fdr_path = fdr_dir / f"{symbol}.parquet"
        vnstock_path = vnstock_dir / f"{symbol}.parquet"
        cafef_path = cafef_dir / f"{symbol}.parquet"
        path = fdr_path if fdr_path.exists() else (vnstock_path if vnstock_path.exists() else cafef_path)
        if not path.exists():
            missing.append(symbol)
            continue
        frame = pd.read_parquet(path).copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        candidates = master[master["ticker"].eq(symbol)]
        matched_parts: list[pd.DataFrame] = []
        for _, security in candidates.iterrows():
            active = frame[
                frame["date"].ge(max(pd.Timestamp(start), security["listing_date"]))
                & frame["date"].le(
                    min(
                        pd.Timestamp(end),
                        security["delisting_date"] if pd.notna(security["delisting_date"]) else pd.Timestamp(end),
                    )
                )
            ].copy()
            if not active.empty:
                active["security_id"] = security["security_id"]
                matched_parts.append(active)
        if not matched_parts:
            missing.append(symbol)
            continue
        combined = pd.concat(matched_parts, ignore_index=True)
        if combined.duplicated(["date", "ticker"], keep=False).any():
            identity_ambiguous.append(symbol)
            continue
        frames.append(combined)
        source_for[symbol] = str(combined["source"].iloc[0])
    if not frames:
        raise RuntimeError("No historical checkpoints matched official HOSE intervals.")
    prices = pd.concat(frames, ignore_index=True)
    bad_ohlc = (
        (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
        | (prices[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    quarantine = prices.loc[bad_ohlc].copy()
    quarantine.to_csv(paths.normalized / "ohlc_quarantine_historical.csv", index=False)
    prices = prices.loc[~bad_ohlc].drop_duplicates(["date", "ticker"]).sort_values(["ticker", "date"])
    promotion = _stage_and_promote_price_panel(paths, prices, "historical-hose-checkpoints")
    coverage = prices.groupby("ticker").agg(
        records=("date", "size"), start=("date", "min"), end=("date", "max")
    ).reset_index()
    coverage["source"] = coverage["ticker"].map(source_for)
    coverage.to_csv(paths.normalized / "historical_hose_coverage.csv", index=False)
    manifest = {
        "status": "partial" if missing or identity_ambiguous else "success",
        "records": len(prices),
        "historically_relevant_tickers": int(master["ticker"].nunique()),
        "tickers_promoted": int(prices["ticker"].nunique()),
        "missing_tickers": sorted(set(missing)),
        "identity_ambiguous_tickers": identity_ambiguous,
        "quarantined_ohlc_rows": len(quarantine),
        "source_counts": pd.Series(source_for).value_counts().to_dict(),
        "promotion": promotion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.raw / "historical_hose_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


class HOSEOfficialListingAdapter:
    """Public HOSE listing service for current stocks and delisting events.

    The routes are the same official JSON resources used by the HOSE listing pages.
    They provide identity and event history, not a licensed historical OHLC panel.
    """

    base_url = "https://api.hsx.vn/l/api/v1/1"
    current_stocks_url = "https://www.hsx.vn/vi/quan-ly-niem-yet/co-phieu"
    delistings_url = "https://www.hsx.vn/vi/quan-ly-niem-yet/huy-niem-yet"

    def __init__(self):
        self.http = JsonHttpClient(RetryPolicy(attempts=4, backoff_seconds=0.5, timeout_seconds=60))

    @staticmethod
    def _data(payload: dict, context: str) -> dict:
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise RuntimeError(f"HOSE official listing response failed for {context}.")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"HOSE official listing response has no data for {context}.")
        return data

    def current_stocks(self) -> pd.DataFrame:
        payload = self.http.request(
            "GET", f"{self.base_url}/securities/stock",
            params={
                "pageIndex": 1, "pageSize": 1000, "alphabet": "", "code": "",
                "sectorId": "",
            },
        )
        return pd.DataFrame(self._data(payload, "current stocks").get("list") or [])

    def delistings(self, year: int) -> pd.DataFrame:
        payload = self.http.request(
            "GET", f"{self.base_url}/securities/cancel",
            params={"pageIndex": 1, "pageSize": 1000, "year": int(year)},
        )
        return pd.DataFrame(self._data(payload, f"delistings {year}").get("list") or [])

    def security_detail(self, security_id: int) -> dict:
        payload = self.http.request(
            "GET", f"{self.base_url}/securities/{int(security_id)}"
        )
        data = self._data(payload, f"security {security_id}")
        return data


def _hose_epoch(value: Any) -> pd.Timestamp:
    """Convert HOSE epoch seconds and reject its year-0001 sentinel."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or float(number) <= 0:
        return pd.NaT
    return pd.to_datetime(float(number), unit="s", utc=True).tz_localize(None).normalize()


def normalize_hose_security_master(
    current: pd.DataFrame,
    delisted_details: list[dict],
    delisting_events: pd.DataFrame,
    fetched_at: str,
) -> pd.DataFrame:
    """Build an event-time security master without using first observed price dates."""
    cancellation_by_id: dict[int, pd.Timestamp] = {}
    if not delisting_events.empty:
        for row in delisting_events.to_dict("records"):
            try:
                security_id = int(row.get("securityId"))
            except (TypeError, ValueError):
                continue
            cancellation_by_id[security_id] = _hose_epoch(row.get("cancelDate"))

    rows: list[dict[str, Any]] = []
    inputs = [*current.to_dict("records"), *delisted_details]
    for item in inputs:
        ticker = str(item.get("code") or "").strip().upper()
        isin = str(item.get("isin") or "").strip().upper()
        detail_id = item.get("id")
        if not re.fullmatch(r"[A-Z0-9]{3}", ticker) or not isin.startswith("VN000000"):
            continue
        if item.get("securityTypeId") not in (None, 1) and item.get("securitiesType") != 1:
            continue
        first_trade = _hose_epoch(item.get("ftdate"))
        registration = _hose_epoch(item.get("regDate"))
        listing_record = _hose_epoch(item.get("listDate"))
        listing_date = first_trade if pd.notna(first_trade) else registration
        if pd.isna(listing_date):
            listing_date = listing_record
        if pd.isna(listing_date):
            # Do not infer listing dates from price observations.
            continue
        delisting_date = cancellation_by_id.get(int(detail_id), pd.NaT)
        known_dates = [
            value for value in [
                _hose_epoch(item.get("acceptDate")), registration, listing_record, first_trade
            ] if pd.notna(value)
        ]
        available_at = min(known_dates) if known_dates else listing_date
        rows.append({
            "security_id": isin,
            "ticker": ticker,
            "company_name": item.get("name"),
            "exchange": "HOSE",
            "isin": isin,
            "figi": item.get("bloomberg"),
            "hose_security_id": int(detail_id),
            "listing_date": listing_date,
            "delisting_date": delisting_date,
            "effective_from": listing_date,
            "effective_to": delisting_date,
            "available_at": available_at,
            "source": "hose_official_listing_service",
            "source_url": (
                HOSEOfficialListingAdapter.delistings_url
                if pd.notna(delisting_date) else HOSEOfficialListingAdapter.current_stocks_url
            ),
            "fetched_at": fetched_at,
            "history_method": "exchange_listing_history",
            "data_class": "real",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("HOSE official listing service produced no valid equity history.")
    out = out.sort_values(["ticker", "effective_from", "security_id"]).drop_duplicates(
        ["security_id", "effective_from"], keep="last"
    )
    digest = hashlib.sha256(
        out.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    ).hexdigest()
    out["raw_checksum"] = digest
    return out.reset_index(drop=True)


def reconcile_price_security_ids_from_master(paths: Paths, master: pd.DataFrame) -> dict:
    """Attach stable ISIN identity to prices only inside verified listing intervals."""
    price_path = paths.normalized / "prices.parquet"
    if not price_path.exists():
        return {"status": "skipped", "reason": "prices.parquet_missing"}
    prices = pd.read_parquet(price_path).copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    reference = master.copy()
    for column in ["listing_date", "delisting_date"]:
        reference[column] = pd.to_datetime(reference[column], errors="coerce")
    updates = 0
    unmatched_tickers: list[str] = []
    ambiguous_rows = 0
    outside_hose_interval_indices: list[int] = []
    for ticker, indices in prices.groupby("ticker").groups.items():
        candidates = reference[reference["ticker"].astype(str).eq(str(ticker))]
        if candidates.empty:
            unmatched_tickers.append(str(ticker))
            continue
        if len(candidates) == 1:
            candidate = candidates.iloc[0]
            group_dates = prices.loc[indices, "date"]
            in_interval = group_dates.ge(candidate["listing_date"]) & (
                pd.isna(candidate["delisting_date"])
                | group_dates.le(candidate["delisting_date"])
            )
            selected_indices = group_dates.index[in_interval]
            outside_hose_interval_indices.extend(group_dates.index[~in_interval].tolist())
            changed = prices.loc[selected_indices, "security_id"].astype(str).ne(
                str(candidate["security_id"])
            )
            changed_indices = selected_indices[changed]
            prices.loc[changed_indices, "security_id"] = candidate["security_id"]
            updates += len(changed_indices)
            continue
        for index in indices:
            date = prices.at[index, "date"]
            active = candidates[
                (candidates["listing_date"] <= date)
                & (candidates["delisting_date"].isna() | (candidates["delisting_date"] >= date))
            ]
            if len(active) == 1:
                stable_id = active.iloc[0]["security_id"]
                if str(prices.at[index, "security_id"]) != str(stable_id):
                    prices.at[index, "security_id"] = stable_id
                    updates += 1
            elif len(active) > 1:
                ambiguous_rows += 1
            else:
                outside_hose_interval_indices.append(index)
    quarantined_rows = 0
    quarantine_path = None
    if outside_hose_interval_indices:
        outside = prices.loc[sorted(set(outside_hose_interval_indices))].copy()
        quarantined_rows = len(outside)
        quarantine = (
            paths.root / "outputs" / "quarantine" / "outside_verified_hose_interval"
            / datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        quarantine.mkdir(parents=True, exist_ok=False)
        quarantine_path = quarantine / "prices.parquet"
        outside.to_parquet(quarantine_path, index=False)
        prices = prices.drop(index=outside.index).reset_index(drop=True)
    promotion = _stage_and_promote_price_panel(
        paths, prices, "hose-official-security-id-reconciliation"
    )
    audit = {
        "status": "pass" if not unmatched_tickers and ambiguous_rows == 0 else "partial",
        "rows_updated": updates,
        "rows_quarantined_outside_verified_hose_interval": quarantined_rows,
        "quarantine_path": str(quarantine_path) if quarantine_path else None,
        "unmatched_tickers": sorted(unmatched_tickers),
        "ambiguous_rows": ambiguous_rows,
        "identity_source": HOSEOfficialListingAdapter.current_stocks_url,
        "promotion": promotion,
    }
    (paths.reports / "price_security_identity_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def crawl_hose_official_security_master(
    paths: Paths, start_year: int = 2015, end_year: int = 2025,
    pause_seconds: float = 0.05,
) -> dict:
    """Collect official current stocks plus equity delistings with checkpointed raw data."""
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    paths.ensure()
    adapter = HOSEOfficialListingAdapter()
    staging = create_staging_run(paths, "hose-official-security-master")
    raw_dir = paths.raw / "hose_official_listing"
    current = adapter.current_stocks()
    _archive_raw_frame(raw_dir, "current-stocks", current)

    cancellation_frames: list[pd.DataFrame] = []
    for year in range(int(start_year), int(end_year) + 1):
        frame = adapter.delistings(year)
        if not frame.empty:
            frame = frame.copy()
            frame["event_year"] = year
            cancellation_frames.append(frame)
            _archive_raw_frame(raw_dir, f"delistings-{year}", frame)
    cancellations = (
        pd.concat(cancellation_frames, ignore_index=True)
        if cancellation_frames else pd.DataFrame()
    )
    equity_events = cancellations[
        cancellations.get("code", pd.Series(dtype=str)).astype(str).str.fullmatch(r"[A-Z0-9]{3}")
        & cancellations.get("isin", pd.Series(dtype=str)).astype(str).str.startswith("VN000000")
    ].drop_duplicates("securityId") if not cancellations.empty else pd.DataFrame()

    details: list[dict] = []
    failures: list[dict[str, Any]] = []
    for row in equity_events.to_dict("records"):
        security_id = row.get("securityId")
        try:
            detail = adapter.security_detail(int(security_id))
            details.append(detail)
            _archive_raw_frame(raw_dir, f"security-{security_id}", pd.DataFrame([detail]))
        except Exception as exc:
            failures.append({"security_id": security_id, "error": type(exc).__name__})
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    fetched_at = datetime.now(timezone.utc).isoformat()
    master = normalize_hose_security_master(current, details, equity_events, fetched_at)
    staged = staging / "security_master.parquet"
    master.to_parquet(staged, index=False)
    master.to_csv(staging / "security_master.csv", index=False)
    manifest = {
        "status": "partial" if failures else "success",
        "source": "hose_official_listing_service",
        "current_equities": int(current["code"].astype(str).str.fullmatch(r"[A-Z0-9]{3}").sum()),
        "delisted_equities": int(master["delisting_date"].notna().sum()),
        "master_rows": len(master),
        "history_year_start": int(start_year),
        "history_year_end": int(end_year),
        "detail_failures": failures,
        "dataset_sha256": sha256_file(staged),
        "current_listing_url": adapter.current_stocks_url,
        "delisting_url": adapter.delistings_url,
        "fetched_at": fetched_at,
        "limitation": (
            "This source certifies listing identity/events only. It does not provide or certify "
            "historical OHLC, corporate-action adjustment factors, or a total-return benchmark."
        ),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    promotion = promote_staged_file(paths, staged, "security_master.parquet")
    manifest["promotion"] = promotion
    manifest["price_identity_reconciliation"] = reconcile_price_security_ids_from_master(
        paths, master
    )
    (paths.raw / "hose_official_security_master_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


class FREDAdapter:
    """Official FRED API adapter for international macro series."""

    endpoint = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise SourceConfigurationError("FRED_API_KEY is required by the official FRED API.")
        self.http = JsonHttpClient()

    def series(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        payload = self.http.request("GET", self.endpoint, params={
            "series_id": series_id, "api_key": self.api_key, "file_type": "json",
            "observation_start": start, "observation_end": end,
        })
        df = pd.DataFrame(payload.get("observations") or [])
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["series_id"] = series_id
        df["available_at"] = pd.to_datetime(df.get("realtime_start", df["date"]))
        df["source"] = "fred_official_api"
        return df[["date", "available_at", "series_id", "value", "source"]]


class WorldBankIndicatorsAdapter:
    """Official, keyless World Bank Indicators API adapter.

    The API exposes the source's latest update date, not a vintage/release timestamp
    for each historical observation. Output is therefore archived as a current
    snapshot and is deliberately not promoted to the point-in-time macro feature table.
    """

    base_url = "https://api.worldbank.org/v2"
    documentation_url = (
        "https://datahelpdesk.worldbank.org/knowledgebase/articles/889392"
    )

    def __init__(self):
        self.http = JsonHttpClient(RetryPolicy(attempts=4, backoff_seconds=0.5, timeout_seconds=60))

    def indicator(
        self, country: str, indicator: str, start_year: int, end_year: int
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        payload = self.http.request(
            "GET",
            f"{self.base_url}/country/{country}/indicator/{indicator}",
            params={
                "format": "json", "date": f"{start_year}:{end_year}",
                "per_page": 20000, "source": 2,
            },
        )
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"World Bank returned an invalid payload for {indicator}.")
        return dict(payload[0] or {}), pd.DataFrame(payload[1] or [])


WORLD_BANK_VIETNAM_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "PA.NUS.FCRF": "Official exchange rate (LCU per USD, period average)",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% of labor force)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
    "FS.AST.DOMS.GD.ZS": "Domestic credit provided by financial sector (% of GDP)",
    "CM.MKT.LCAP.GD.ZS": "Market capitalization of listed domestic companies (% of GDP)",
}


def crawl_world_bank_vietnam_snapshot(
    paths: Paths, start_year: int = 2015, end_year: int = 2025
) -> dict:
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    paths.ensure()
    adapter = WorldBankIndicatorsAdapter()
    fetched_at = pd.Timestamp.now(tz="UTC")
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    source_updates: dict[str, Any] = {}
    for indicator, label in WORLD_BANK_VIETNAM_INDICATORS.items():
        try:
            metadata, raw = adapter.indicator("VNM", indicator, start_year, end_year)
            source_updates[indicator] = metadata.get("lastupdated")
            if raw.empty:
                raise RuntimeError("empty response")
            out = pd.DataFrame({
                "series_id": indicator,
                "series_name": label,
                "country": raw.get("countryiso3code", pd.Series("VNM", index=raw.index)),
                "observation_date": pd.to_datetime(raw["date"].astype(str) + "-12-31", errors="coerce"),
                "value": pd.to_numeric(raw["value"], errors="coerce"),
                "unit": raw.get("unit", pd.Series(pd.NA, index=raw.index)),
                "source_last_updated": pd.to_datetime(metadata.get("lastupdated"), errors="coerce"),
                "release_date": pd.NaT,
                "available_at": fetched_at,
                "fetched_at": fetched_at,
                "source": "world_bank_indicators_api_v2",
                "source_url": adapter.documentation_url,
                "data_class": "real_snapshot_non_pit",
                "pit_eligible": False,
                "pit_exclusion_reason": (
                    "API supplies the current revised observation and source-level last update, "
                    "not the observation's historical release/vintage timestamp."
                ),
            }).dropna(subset=["observation_date", "value"])
            if out.empty:
                raise RuntimeError("no non-null observations in requested period")
            frames.append(out)
        except Exception as exc:
            failures.append({"indicator": indicator, "error": type(exc).__name__})
    if not frames:
        raise RuntimeError(f"No World Bank observations collected; failures={failures}")
    result = pd.concat(frames, ignore_index=True).sort_values(["series_id", "observation_date"])
    raw_bytes = result.to_csv(index=False).encode("utf-8")
    result["raw_checksum"] = hashlib.sha256(raw_bytes).hexdigest()
    staging = create_staging_run(paths, "world-bank-vietnam-snapshot")
    parquet = staging / "macro_world_bank_snapshot.parquet"
    result.to_parquet(parquet, index=False)
    result.to_csv(staging / "macro_world_bank_snapshot.csv", index=False)
    # This is intentionally a separately named snapshot, never normalized/macro.parquet.
    promotion = promote_staged_file(paths, parquet, "macro_world_bank_snapshot.parquet")
    report_csv = paths.reports / "world_bank_vietnam_macro_snapshot.csv"
    result.to_csv(report_csv, index=False)
    manifest = {
        "status": "partial" if failures or result["series_id"].nunique() < len(WORLD_BANK_VIETNAM_INDICATORS) else "success",
        "records": len(result),
        "indicators_requested": len(WORLD_BANK_VIETNAM_INDICATORS),
        "indicators_collected": int(result["series_id"].nunique()),
        "source_updates": source_updates,
        "failures": failures,
        "pit_eligible": False,
        "role": "descriptive_and_crosscheck_only",
        "documentation_url": adapter.documentation_url,
        "promotion": promotion,
        "fetched_at": fetched_at.isoformat(),
    }
    (paths.raw / "world_bank_vietnam_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def audit_available_data_sources(paths: Paths) -> dict:
    """Write a reproducible source/data-gap inventory without exposing credentials."""
    paths.ensure()
    files = {
        "prices": paths.normalized / "prices.parquet",
        "security_master": paths.normalized / "security_master.parquet",
        "corporate_actions": paths.normalized / "corporate_actions.parquet",
        "benchmark": paths.normalized / "benchmark.parquet",
        "macro_pit": paths.normalized / "macro.parquet",
        "macro_world_bank_snapshot": paths.normalized / "macro_world_bank_snapshot.parquet",
        "financial_statements": paths.normalized / "financial_statements.parquet",
        "foreign_flow": paths.normalized / "foreign_flow.parquet",
        "index_membership": paths.normalized / "index_membership.parquet",
        "research_v2_corporate_actions": (
            paths.root / "outputs" / "research_v2" / "normalized" /
            "corporate_actions.parquet"
        ),
        "research_v2_total_return_candidate": (
            paths.root / "outputs" / "research_v2" / "normalized" /
            "prices_total_return.parquet"
        ),
    }
    datasets: dict[str, Any] = {}
    for name, path in files.items():
        entry: dict[str, Any] = {"exists": path.exists(), "path": str(path)}
        if path.exists():
            table = pd.read_parquet(path)
            entry.update({"records": len(table), "columns": list(table.columns)})
            if "ticker" in table:
                entry["tickers"] = int(table["ticker"].nunique())
            if "date" in table and not table.empty:
                dates = pd.to_datetime(table["date"], errors="coerce")
                entry["date_start"] = str(dates.min().date())
                entry["date_end"] = str(dates.max().date())
        datasets[name] = entry
    sources = [
        {
            "source": "HOSE official website/API", "role": "security identity and listing/delisting history",
            "access": "public", "usable_now": True,
            "limitation": "historical EOD OHLC and total-return index feeds are licensed services",
            "url": HOSEOfficialListingAdapter.current_stocks_url,
        },
        {
            "source": "VSDC public notices", "role": "corporate-action evidence",
            "access": "public notices", "usable_now": True,
            "limitation": (
                "official event terms are parsed, but confirmatory use still requires "
                "independent ex-date corroboration and zero unresolved material events"
            ),
            "url": "https://www.vsd.vn/",
        },
        {
            "source": "Vietstock Finance", "role": "authenticated OHLCV cross-check",
            "access": "public page plus session-bound authenticated requests",
            "usable_now": bool(
                os.getenv("VIETSTOCK_COOKIE_FILE") or os.getenv("VIETSTOCK_AUTH_HEADER_FILE")
            ),
            "limitation": (
                None if (os.getenv("VIETSTOCK_COOKIE_FILE") or os.getenv("VIETSTOCK_AUTH_HEADER_FILE"))
                else "no reusable cookie/header file configured; no credential is guessed or persisted"
            ),
            "url": VietstockAdapter.base_url,
        },
        {
            "source": "SSI FastConnect", "role": "official broker OHLC and index components",
            "access": "credentialed API", "usable_now": bool(os.getenv("SSI_CONSUMER_ID") and os.getenv("SSI_CONSUMER_SECRET")),
            "limitation": "consumer ID/secret absent" if not (os.getenv("SSI_CONSUMER_ID") and os.getenv("SSI_CONSUMER_SECRET")) else None,
            "url": SSIFastConnectAdapter.base_url,
        },
        {
            "source": "Trading Economics", "role": "OHLC cross-check and macro snapshot",
            "access": "credentialed API", "usable_now": bool(os.getenv("TRADING_ECONOMICS_API_KEY")),
            "limitation": "API key absent; market history lacks volume and verified adjustment semantics",
            "url": TradingEconomicsAdapter.documentation_url,
        },
        {
            "source": "World Bank Indicators API v2", "role": "Vietnam macro snapshot",
            "access": "public keyless API", "usable_now": True,
            "limitation": "current revised snapshot has no observation-specific release vintage",
            "url": WorldBankIndicatorsAdapter.documentation_url,
        },
        {
            "source": "FinanceDataReader/Yahoo", "role": "historical OHLCV checkpoint source",
            "access": "public adapter", "usable_now": True,
            "limitation": "not exchange-official; adjustment policy remains uncertified",
            "url": "https://github.com/FinanceData/FinanceDataReader",
        },
        {
            "source": "vnstock/KBS", "role": "OHLCV fallback and cross-check",
            "access": "public adapter subject to provider policy", "usable_now": True,
            "limitation": "not exchange-official; delisted coverage and adjustment semantics are incomplete",
            "url": "https://github.com/thinh-vu/vnstock",
        },
        {
            "source": "CafeF public history", "role": "last-resort historical OHLCV gap coverage",
            "access": "public website endpoint", "usable_now": True,
            "limitation": "aggregated reference data; not exchange-official and adjustment semantics remain uncertified",
            "url": "https://cafef.vn/du-lieu/lich-su-giao-dich-hose/all-1.chn",
        },
        {
            "source": "CafeF corporate-action history", "role": "independent ex-date corroboration",
            "access": "public website", "usable_now": True,
            "limitation": (
                "used to corroborate VSDC events, not treated as an exchange-official source "
                "or as a standalone adjustment authority"
            ),
            "url": "https://cafef.vn/du-lieu/",
        },
        {
            "source": "IMF SDMX", "role": "international macroeconomic series",
            "access": "public SDMX services", "usable_now": False,
            "limitation": "not integrated because a release-vintage contract for the selected Vietnam series has not been verified",
            "url": "https://sdmxcentral.imf.org/sdmx/v2/",
        },
        {
            "source": "Vietnam National Statistics Office PX-Web", "role": "official domestic macro statistics",
            "access": "public PX-Web portal", "usable_now": False,
            "limitation": "no stable dataset/API and historical publication-time contract selected for this study",
            "url": "https://www.gso.gov.vn/px-web-2/",
        },
        {
            "source": "State Bank of Vietnam", "role": "policy rates, exchange rates and banking statistics",
            "access": "public web publications", "usable_now": False,
            "limitation": "no stable bulk API/release-vintage adapter verified in this repository",
            "url": "https://sbv.gov.vn/",
        },
    ]
    blockers = [
        "complete OHLCV coverage for every historically relevant HOSE security",
        "verified corporate-action adjustment contract bound to the exact price dataset hash",
        "licensed/documented VN-Index total-return benchmark series",
    ]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "sources": sources,
        "research_blockers": blockers,
    }
    (paths.reports / "data_source_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["# Data source and gap audit", "", f"Generated: {result['generated_at']}", "", "## Sources", ""]
    for item in sources:
        lines.append(
            f"- **{item['source']}** — {item['role']}; access: {item['access']}; "
            f"usable now: {item['usable_now']}. Limitation: {item['limitation'] or 'none recorded'}."
        )
    lines.extend(["", "## Remaining research blockers", ""])
    lines.extend([f"- {item}" for item in blockers])
    (paths.reports / "DATA_SOURCE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def import_point_in_time_table(
    input_path: Path, output_path: Path, required: set[str], table_name: str
) -> dict:
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    required = set(required) | {"source", "source_url"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} missing required point-in-time fields: {missing}")
    for col in [c for c in df.columns if c.endswith("_date") or c in {
        "date", "available_at", "effective_from", "effective_to"
    }]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    temporal_origins = {
        "corporate_actions": "announcement_date",
        "financial_statements": "fiscal_period_end",
        "macro": "observation_date",
        "foreign_flow": "date",
        "benchmark": "date",
    }
    origin = temporal_origins.get(table_name)
    if origin and origin in df and (df["available_at"] < pd.to_datetime(df[origin])).any():
        raise ValueError(f"{table_name} has available_at before {origin}")
    temporal_required = [
        column for column in required
        if column.endswith("_date") or column in {"date", "available_at", "effective_from"}
    ]
    temporal_required = [
        column for column in temporal_required
        if column not in {"delisting_date", "effective_to"}
    ]
    invalid_temporal = [column for column in temporal_required if df[column].isna().any()]
    if invalid_temporal:
        raise ValueError(f"{table_name} has invalid required timestamps: {invalid_temporal}")
    if table_name == "benchmark":
        if not df["index_type"].astype(str).str.lower().eq("total_return").all():
            raise ValueError("benchmark.index_type must explicitly be total_return")
        if not df["methodology_url"].astype(str).str.startswith(("http://", "https://")).all():
            raise ValueError("benchmark requires a documented total-return methodology_url")
    if table_name == "security_master":
        listing = pd.to_datetime(df["listing_date"], errors="coerce")
        delisting = pd.to_datetime(df["delisting_date"], errors="coerce")
        if listing.isna().any() or ((delisting.notna()) & (delisting < listing)).any():
            raise ValueError("security_master has invalid listing/delisting intervals")
        if df["security_id"].isna().any() or df["security_id"].astype(str).str.strip().eq("").any():
            raise ValueError("security_master requires a stable non-empty security_id")
    checksum = hashlib.sha256(input_path.read_bytes()).hexdigest()
    df["fetched_at"] = pd.to_datetime(df.get("fetched_at", datetime.now(timezone.utc)))
    df["raw_checksum"] = df.get("raw_checksum", checksum)
    df["data_class"] = df.get("data_class", "real")
    project_root = output_path.parents[2]
    paths = Paths(project_root)
    staging = create_staging_run(paths, f"import-{table_name}")
    staged_output = staging / output_path.name
    df.to_parquet(staged_output, index=False)
    promoted = promote_staged_file(paths, staged_output, output_path.name)
    manifest = {
        "table": table_name, "records": len(df), "input_sha256": checksum,
        "output": str(output_path), "staging": str(staging), "promotion": promoted,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
