from __future__ import annotations

import hashlib
import json
import os
import re
import time
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
        listing["Symbol"].str.fullmatch(r"[A-Z]{3}")
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
    # Yahoo's HOSE timestamps are represented as the prior UTC calendar date.
    out["date"] = (
        pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
        + pd.Timedelta(days=1)
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
        listing["Symbol"].str.fullmatch(r"[A-Z]{3}")
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
        listing["Symbol"].str.fullmatch(r"[A-Z]{3}")
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
