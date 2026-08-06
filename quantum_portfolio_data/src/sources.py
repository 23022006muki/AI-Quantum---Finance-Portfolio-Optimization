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

from .data_pipeline import PRICE_COLUMNS, Paths


class SourceConfigurationError(RuntimeError):
    pass


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

    def request(self, method: str, url: str, **kwargs) -> dict:
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
        raise RuntimeError(f"Source request failed after retries: {last}")


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
    out["adjusted_close"] = out["close"]
    out["trading_value"] = out.get("trading_value", out["volume"] * out["close"])
    raw = out.to_csv(index=False).encode()
    out["source"] = "ssi_fastconnect_official"
    out["source_url"] = source_url
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    out["available_at"] = out["date"]
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
    endpoint = f"{adapter.base_url}/Market/DailyOhlc"
    for ticker in tickers:
        try:
            frames.append(normalize_ssi_ohlc(
                adapter.daily_ohlc(ticker, start, end), ticker, endpoint
            ))
        except Exception as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
    if not frames:
        raise RuntimeError(f"No SSI data collected. Failures: {failures}")
    prices = pd.concat(frames, ignore_index=True)
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    securities = adapter.securities("HOSE")
    symbol_col = "Symbol" if "Symbol" in securities else "symbol"
    name_col = "StockName" if "StockName" in securities else "stockName"
    master = pd.DataFrame({
        "ticker": securities[symbol_col].astype(str).str.upper(),
        "company_name": securities[name_col].astype(str),
        "exchange": "HOSE", "industry": pd.NA, "sector": pd.NA,
        "listing_date": pd.NaT, "delisting_date": pd.NaT,
        "effective_from": pd.NaT, "effective_to": pd.NaT,
        "available_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
        "source": "ssi_fastconnect_official", "data_class": "real",
    })
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)
    manifest = {
        "status": "partial" if failures else "success", "data_class": "real",
        "source": "ssi_fastconnect_official", "records": len(prices),
        "tickers_requested": len(tickers), "tickers_collected": prices.ticker.nunique(),
        "failures": failures, "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.raw / "ssi_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class VietstockAdapter:
    """Vietstock Finance historical-price adapter.

    The adapter creates its own anonymous HTTP session, obtains a fresh anti-forgery
    token from the stock page, and never reads or persists browser cookies.
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
    for ticker in [item.strip().upper() for item in tickers if item.strip()]:
        try:
            frame = adapter.daily_ohlc(ticker, start, end)
            if frame.empty:
                raise RuntimeError("empty response")
            frames.append(normalize_vietstock_ohlc(
                frame,
                ticker,
                f"{adapter.base_url}/{ticker}/lich-su-giao-dich.htm",
            ))
        except Exception as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
    if not frames:
        raise RuntimeError(f"No Vietstock data collected. Failures: {failures}")
    prices = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    collected = sorted(prices["ticker"].unique())
    master = pd.DataFrame({
        "ticker": collected,
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
        "data_class": "real",
    })
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)
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
        "tls_verification": verify_tls,
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
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    prices.to_csv(paths.normalized / "prices.csv", index=False)

    selected_listing = listing[listing["Symbol"].isin(completed)].copy()
    selected_listing = selected_listing.set_index("Symbol").reindex(completed).reset_index()
    master = pd.DataFrame({
        "ticker": selected_listing["Symbol"],
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
        "data_class": "real",
    })
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)

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
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    prices.to_csv(paths.normalized / "prices.csv", index=False)
    selected = listing.set_index("Symbol").reindex(completed).reset_index()
    master = pd.DataFrame({
        "ticker": selected["Symbol"],
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
        "data_class": "real",
    })
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)
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
    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    prices.to_csv(paths.normalized / "prices.csv", index=False)

    selected_listing = listing.set_index("Symbol").reindex(selected).reset_index()
    first_price = prices.groupby("ticker").first().reindex(selected)
    master = pd.DataFrame({
        "ticker": selected_listing["Symbol"],
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
    master.to_parquet(paths.normalized / "security_master.parquet", index=False)
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
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.raw / "hose300_manifest.json").write_text(
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
    checksum = hashlib.sha256(input_path.read_bytes()).hexdigest()
    df["fetched_at"] = pd.to_datetime(df.get("fetched_at", datetime.now(timezone.utc)))
    df["raw_checksum"] = df.get("raw_checksum", checksum)
    df["data_class"] = df.get("data_class", "real")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return {"table": table_name, "records": len(df), "output": str(output_path)}
