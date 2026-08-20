from __future__ import annotations

import hashlib
import html
import json
import math
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests

from .data_pipeline import Paths, sha256_file


CORPORATE_ACTION_COLUMNS = [
    "security_id", "ticker", "event_type", "announcement_date", "record_date",
    "ex_date", "effective_date", "payment_date", "cash_dividend_per_share",
    "stock_dividend_ratio", "bonus_share_ratio", "split_ratio",
    "reverse_split_ratio", "rights_ratio", "rights_subscription_price",
    "adjustment_factor", "currency", "source", "source_url",
    "corroboration_source", "corroboration_url", "fetched_at", "available_at",
    "raw_checksum", "parser_version", "verification_status", "verification_notes",
]

SUPPORTED_EVENT_TYPES = {
    "CASH_DIVIDEND", "STOCK_DIVIDEND", "BONUS_SHARE", "STOCK_SPLIT",
    "REVERSE_SPLIT", "RIGHTS_ISSUE", "SHARE_CONVERSION", "MERGER", "OTHER",
}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            value = re.sub(r"\s+", " ", data).strip()
            if value:
                self.parts.append(value)


def _visible_lines(markup: str) -> list[str]:
    parser = _VisibleTextParser()
    parser.feed(markup)
    return parser.parts


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    folded = "".join(character for character in normalized if not unicodedata.combining(character))
    return folded.replace("đ", "d").replace("Đ", "D").lower()


def _number_vi(value: str | float | int | None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().replace(" ", "")
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    elif text.count(".") > 1 or (text.count(".") == 1 and len(text.rsplit(".", 1)[1]) == 3):
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return None


def _timestamp_vi(value: str | None, with_time: bool = False) -> pd.Timestamp:
    if not value:
        return pd.NaT
    fmt = "%d/%m/%Y - %H:%M:%S" if with_time else "%d/%m/%Y"
    try:
        return pd.Timestamp(datetime.strptime(value.strip(), fmt))
    except ValueError:
        return pd.NaT


def _archive_raw_text(directory: Path, stem: str, content: str, suffix: str = ".html") -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-")[:100]
    target = directory / f"{safe}-{digest[:16]}{suffix}"
    if not target.exists():
        target.write_bytes(payload)
    return {"path": str(target), "sha256": digest, "bytes": len(payload)}


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    attempts: int = 4,
    timeout: float = 45.0,
    backoff: float = 0.75,
    **kwargs: Any,
) -> requests.Response:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Source request failed after retries ({type(last).__name__}).")


class CafeFCorporateActionAdapter:
    """CafeF ex-date history used only to corroborate official event notices."""

    page_url = "https://cafef.vn/du-lieu/{ticker}/thong-tin-chung.chn"
    legacy_page_url = (
        "https://cafef.vn/du-lieu/DuLieu.aspx?cat_id=1009&san=hose&symbol={ticker}"
    )
    parser_version = "cafef-corporate-history-v1"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; academic-research/0.2)",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
        })

    def fetch(self, ticker: str) -> tuple[str, str]:
        url = self.page_url.format(ticker=ticker.lower())
        try:
            response = _request_with_retry(self.session, "GET", url)
        except RuntimeError:
            # A small subset of CafeF symbols still resolves only through the
            # site's legacy public route. This is a same-source fallback, not a
            # bypass of authentication or access control.
            legacy = self.legacy_page_url.format(ticker=ticker.upper())
            response = _request_with_retry(self.session, "GET", legacy)
        return response.text, str(response.url)

    @staticmethod
    def parse(markup: str, ticker: str, source_url: str, fetched_at: str) -> list[dict]:
        anchor = re.search(
            r"Lịch sử trả cổ tức chia thưởng và tăng vốn.*?"
            r"<div[^>]+class=[\"']middle[\"'][^>]*>(.*?)</div>",
            markup,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not anchor:
            return []
        chunks = re.split(r"<br\s*/?>", anchor.group(1), flags=re.IGNORECASE)
        rows: list[dict] = []
        current_date: pd.Timestamp = pd.NaT
        for chunk in chunks:
            date_match = re.search(r"<b>\s*(\d{2}/\d{2}/\d{4})\s*</b>\s*:", chunk)
            if date_match:
                current_date = _timestamp_vi(date_match.group(1))
            description = html.unescape(re.sub(r"<[^>]+>", " ", chunk))
            description = re.sub(r"\s+", " ", description).strip(" -:\u2002\u2009\xa0")
            description = re.sub(r"^\d{2}/\d{2}/\d{4}\s*:\s*", "", description)
            if pd.isna(current_date) or not description:
                continue
            lower = _ascii_fold(description)
            event_type = "OTHER"
            cash = stock = bonus = split = reverse = rights = rights_price = None
            ratio_match = re.search(r"ty le\s*([\d.,]+)\s*%", lower)
            ratio = (_number_vi(ratio_match.group(1)) or 0.0) / 100.0 if ratio_match else None
            if "co tuc bang tien" in lower:
                event_type = "CASH_DIVIDEND"
                cash = ratio * 10_000.0 if ratio is not None else None
            elif "co tuc bang co phieu" in lower:
                event_type = "STOCK_DIVIDEND"
                stock = ratio
            elif "thuong bang co phieu" in lower or "co phieu thuong" in lower:
                event_type = "BONUS_SHARE"
                bonus = ratio
            elif "tach co phieu" in lower:
                event_type = "STOCK_SPLIT"
                split = ratio
            elif "gop co phieu" in lower:
                event_type = "REVERSE_SPLIT"
                reverse = ratio
            elif "quyen mua" in lower or "phat hanh cho cdhh" in lower:
                event_type = "RIGHTS_ISSUE"
                rights = ratio
                price_match = re.search(r"(?:gia|gia phat hanh)\s*([\d.,]+)", lower)
                rights_price = _number_vi(price_match.group(1)) if price_match else None
            elif "sap nhap" in lower:
                event_type = "MERGER"
            elif "hoan doi" in lower or "chuyen doi" in lower:
                event_type = "SHARE_CONVERSION"
            else:
                continue
            rows.append({
                "ticker": ticker.upper(), "event_type": event_type,
                "ex_date": current_date, "effective_date": current_date,
                "cash_dividend_per_share": cash,
                "stock_dividend_ratio": stock, "bonus_share_ratio": bonus,
                "split_ratio": split, "reverse_split_ratio": reverse,
                "rights_ratio": rights, "rights_subscription_price": rights_price,
                "currency": "VND", "source": "cafef_public_corporate_history",
                "source_url": source_url, "fetched_at": fetched_at,
                # A current summary page cannot establish historical availability.
                "available_at": pd.Timestamp(fetched_at).tz_localize(None),
                "parser_version": CafeFCorporateActionAdapter.parser_version,
                "verification_status": "reference_only_unverified",
                "verification_notes": description,
            })
        return rows


class VSDCCorporateActionAdapter:
    """VSDC public issuer-notice collector with session token and paged search."""

    base_url = "https://www.vsd.vn"
    search_path = "/vi/search"
    parser_version = "vsdc-issuer-rights-v1"
    action_keywords = (
        "cổ tức", "phát hành cổ phiếu", "cổ phiếu thưởng", "thưởng bằng cổ phiếu",
        "quyền mua", "tách cổ phiếu", "gộp cổ phiếu", "sáp nhập", "hoán đổi",
    )

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; academic-research/0.2)",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
        })

    @staticmethod
    def _result_links(markup: str, ticker: str) -> list[dict[str, str]]:
        result = []
        for match in re.finditer(
            r"<a\s+href=[\"'](?P<href>/vi/(?:ad1|ad)/\d+)[\"'][^>]*>"
            r"(?P<title>.*?)</a>", markup, flags=re.IGNORECASE | re.DOTALL,
        ):
            title = html.unescape(re.sub(r"<[^>]+>", "", match.group("title")))
            title = re.sub(r"\s+", " ", title).strip()
            lower = title.lower()
            if not re.match(rf"^{re.escape(ticker)}\s*:", title, flags=re.IGNORECASE):
                continue
            if not any(keyword in lower for keyword in VSDCCorporateActionAdapter.action_keywords):
                continue
            result.append({"title": title, "url": urljoin(VSDCCorporateActionAdapter.base_url, match.group("href"))})
        return result

    @staticmethod
    def _page_count(markup: str) -> int:
        match = re.search(r"/\s*([\d.]+)\s*b.n ghi", html.unescape(markup), re.IGNORECASE)
        total = int(match.group(1).replace(".", "")) if match else 0
        return max(1, math.ceil(total / 20))

    def search(
        self, ticker: str, start: str, end: str,
    ) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
        query = urlencode({
            "text": ticker.upper(), "type": 4, "obj": 110, "buss": 11021,
            "fdate": pd.Timestamp(start).strftime("%d/%m/%Y"),
            "tdate": pd.Timestamp(end).strftime("%d/%m/%Y"),
        })
        url = f"{self.base_url}{self.search_path}?{query}"
        first = _request_with_retry(self.session, "GET", url)
        pages = [("search-1", first.text)]
        links = self._result_links(first.text, ticker.upper())
        token_match = re.search(r'name=["\']__VPToken["\']\s+content=["\']([^"\']+)', first.text)
        token = token_match.group(1) if token_match else None
        for page in range(2, self._page_count(first.text) + 1):
            headers = {"Content-Type": "application/json;charset=utf-8"}
            if token:
                headers["__VPToken"] = token
            response = _request_with_retry(
                self.session, "POST", url,
                json={"SearchKey": 4, "CurrentPage": page}, headers=headers,
            )
            pages.append((f"search-{page}", response.text))
            links.extend(self._result_links(response.text, ticker.upper()))
        deduplicated = {item["url"]: item for item in links}
        return list(deduplicated.values()), pages

    def detail(self, url: str) -> str:
        return _request_with_retry(self.session, "GET", url).text

    @staticmethod
    def parse(markup: str, source_url: str, fetched_at: str) -> list[dict]:
        lines = _visible_lines(markup)
        try:
            lines = lines[:lines.index("Tin cùng tổ chức")]
        except ValueError:
            pass
        joined = "\n".join(lines)

        def after(label: str) -> str | None:
            for index, line in enumerate(lines):
                if line.strip().lower() == label.lower() and index + 1 < len(lines):
                    return lines[index + 1].strip()
            return None

        title = next((line for line in lines if re.match(r"^[A-Z0-9]{3}\s*:", line)), "")
        ticker = (after("Mã chứng khoán:") or title.split(":", 1)[0]).strip().upper()
        isin = (after("Mã ISIN:") or "").strip().upper()
        venue = (after("Nơi giao dịch:") or after("Sàn giao dịch:") or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{3}", ticker) or venue not in {"HOSE", "HSX"}:
            return []
        folded = _ascii_fold(joined)
        update = re.search(r"cap nhat ngay\s*(\d{2}/\d{2}/\d{4}\s*-\s*\d{2}:\d{2}:\d{2})", folded)
        announcement = _timestamp_vi(update.group(1), with_time=True) if update else pd.NaT
        record_match = re.search(r"ngay dang ky cuoi cung:\s*\n?(\d{2}/\d{2}/\d{4})", folded)
        record_date = _timestamp_vi(record_match.group(1)) if record_match else pd.NaT
        payment_match = re.search(
            r"(?:thoi gian thuc hien|ngay thanh toan):\s*\n?(\d{2}/\d{2}/\d{4})",
            folded,
        )
        payment_date = _timestamp_vi(payment_match.group(1)) if payment_match else pd.NaT
        lower = folded
        base = {
            "ticker": ticker, "security_id": isin or None,
            "announcement_date": announcement.normalize() if pd.notna(announcement) else pd.NaT,
            "record_date": record_date, "ex_date": pd.NaT, "effective_date": pd.NaT,
            "payment_date": payment_date, "currency": "VND",
            "source": "vsdc_official_notice", "source_url": source_url,
            "fetched_at": fetched_at,
            "available_at": announcement if pd.notna(announcement) else pd.Timestamp(fetched_at).tz_localize(None),
            "parser_version": VSDCCorporateActionAdapter.parser_version,
            "verification_status": "official_terms_ex_date_unresolved",
            "verification_notes": title,
        }
        rows: list[dict] = []
        cash_matches = re.findall(
            r"(?:01|1)\s*co phieu\s*(?:duoc|se duoc)\s*nhan\s*([\d.,]+)\s*dong",
            lower,
        )
        if "co tuc" in lower and "bang tien" in lower:
            cash = _number_vi(cash_matches[0]) if cash_matches else None
            rows.append({**base, "event_type": "CASH_DIVIDEND", "cash_dividend_per_share": cash})

        share_matches = re.findall(
            r"(?:so huu|co dong so huu)\s*([\d.,]+)\s*co phieu\s*"
            r"(?:duoc|se duoc)\s*nhan(?:\s+them)?\s*([\d.,]+)\s*co phieu",
            lower,
        )
        share_ratio = None
        if share_matches:
            denominator = _number_vi(share_matches[0][0])
            numerator = _number_vi(share_matches[0][1])
            if denominator and numerator is not None:
                share_ratio = numerator / denominator
        # VSDC notices consistently publish a compact old:new entitlement ratio
        # even when the explanatory sentence varies ("nhận thêm", "được hưởng
        # quyền", leading zeroes, etc.).  Use it as a documented fallback.
        execution_ratio = re.search(
            r"ty le thuc hien\s*:\s*([\d.,]+)\s*:\s*([\d.,]+)", lower
        )
        if share_ratio is None and execution_ratio:
            denominator = _number_vi(execution_ratio.group(1))
            numerator = _number_vi(execution_ratio.group(2))
            if denominator and numerator is not None:
                share_ratio = numerator / denominator
        if "co tuc" in lower and "bang co phieu" in lower:
            rows.append({**base, "event_type": "STOCK_DIVIDEND", "stock_dividend_ratio": share_ratio})
        if "co phieu thuong" in lower or "thuong bang co phieu" in lower:
            rows.append({**base, "event_type": "BONUS_SHARE", "bonus_share_ratio": share_ratio})

        if "quyen mua" in lower or ("phat hanh" in lower and "mua" in lower):
            rights_match = re.search(
                r"so huu\s*([\d.,]+)\s*co phieu\s*duoc\s*mua\s*([\d.,]+)\s*co phieu",
                lower,
            )
            rights_ratio = None
            if rights_match:
                denominator = _number_vi(rights_match.group(1))
                numerator = _number_vi(rights_match.group(2))
                if denominator and numerator is not None:
                    rights_ratio = numerator / denominator
            if rights_ratio is None:
                rights_ratio = share_ratio
            price_match = re.search(
                r"(?:gia (?:phat hanh|mua|dang ky mua))\s*:\s*([\d.,]+)\s*dong",
                lower,
            )
            rows.append({
                **base, "event_type": "RIGHTS_ISSUE", "rights_ratio": rights_ratio,
                "rights_subscription_price": _number_vi(price_match.group(1)) if price_match else None,
            })
        if "tach co phieu" in lower:
            rows.append({**base, "event_type": "STOCK_SPLIT", "split_ratio": share_ratio})
        if "gop co phieu" in lower:
            rows.append({**base, "event_type": "REVERSE_SPLIT", "reverse_split_ratio": share_ratio})
        if "sap nhap" in lower:
            rows.append({**base, "event_type": "MERGER"})
        if "hoan doi" in lower or "chuyen doi" in lower:
            rows.append({**base, "event_type": "SHARE_CONVERSION"})
        return rows


def _event_value(row: pd.Series | dict) -> float | None:
    event_type = row.get("event_type")
    field = {
        "CASH_DIVIDEND": "cash_dividend_per_share",
        "STOCK_DIVIDEND": "stock_dividend_ratio",
        "BONUS_SHARE": "bonus_share_ratio",
        "STOCK_SPLIT": "split_ratio",
        "REVERSE_SPLIT": "reverse_split_ratio",
        "RIGHTS_ISSUE": "rights_ratio",
    }.get(str(event_type))
    if not field:
        return None
    value = row.get(field)
    return float(value) if pd.notna(value) else None


def _terms_compatible(official: pd.Series, reference: pd.Series) -> tuple[bool, str]:
    left, right = _event_value(official), _event_value(reference)
    if left is None or right is None:
        return False, "material event terms incomplete"
    tolerance = max(1e-6, abs(left) * 0.011)
    if abs(left - right) > tolerance:
        return False, f"term conflict official={left:g}, reference={right:g}"
    if official["event_type"] == "RIGHTS_ISSUE":
        op = official.get("rights_subscription_price")
        if pd.isna(op):
            return False, "rights subscription price missing"
    return True, "official terms agree with independent ex-date reference"


def reconcile_corporate_actions(
    official: pd.DataFrame, reference: pd.DataFrame, security_master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame in (official, reference):
        for column in ["announcement_date", "record_date", "ex_date", "effective_date", "payment_date", "available_at"]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
    rows: list[dict] = []
    conflicts: list[dict] = []
    matched_reference: set[int] = set()
    for _, event in official.iterrows():
        candidates = reference[
            reference["ticker"].eq(event["ticker"])
            & reference["event_type"].eq(event["event_type"])
        ].copy()
        if pd.notna(event.get("record_date")) and not candidates.empty:
            candidates["date_distance"] = (
                event["record_date"] - candidates["ex_date"]
            ).dt.days
            candidates = candidates[candidates["date_distance"].between(0, 7)]
            candidates = candidates.sort_values(["date_distance", "ex_date"])
        row = event.to_dict()
        row["corroboration_source"] = None
        row["corroboration_url"] = None
        if candidates.empty:
            row["verification_status"] = "unresolved_ex_date"
            row["verification_notes"] = "No matching CafeF ex-date event within 0-7 calendar days before record date."
            rows.append(row)
            conflicts.append({
                "ticker": row["ticker"], "event_type": row["event_type"],
                "record_date": row.get("record_date"), "reason": "missing_ex_date_corroboration",
                "source_url": row.get("source_url"),
            })
            continue
        candidate_index = int(candidates.index[0])
        candidate = candidates.iloc[0]
        matched_reference.add(candidate_index)
        compatible, note = _terms_compatible(event, candidate)
        row["ex_date"] = candidate["ex_date"]
        row["effective_date"] = candidate["ex_date"]
        row["corroboration_source"] = candidate["source"]
        row["corroboration_url"] = candidate["source_url"]
        row["verification_status"] = "verified_cross_source" if compatible else "conflict"
        row["verification_notes"] = note
        row["source"] = "vsdc_official_notice+cafef_reference"
        row["raw_checksum"] = hashlib.sha256(
            f"{event.get('raw_checksum','')}|{candidate.get('raw_checksum','')}".encode("utf-8")
        ).hexdigest()
        rows.append(row)
        if not compatible:
            conflicts.append({
                "ticker": row["ticker"], "event_type": row["event_type"],
                "record_date": row.get("record_date"), "ex_date": row.get("ex_date"),
                "reason": note, "source_url": row.get("source_url"),
                "corroboration_url": row.get("corroboration_url"),
            })
    for index, event in reference.iterrows():
        if int(index) in matched_reference:
            continue
        row = event.to_dict()
        row["verification_status"] = "reference_only_unverified"
        row["verification_notes"] = "CafeF event was not matched to a VSDC official notice."
        rows.append(row)
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        ledger = pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
    master = security_master.copy()
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], errors="coerce")
    for index, row in ledger.iterrows():
        if pd.notna(row.get("security_id")) and str(row.get("security_id")).startswith("VN"):
            continue
        date = row.get("effective_date")
        candidates = master[master["ticker"].eq(row["ticker"])]
        if pd.notna(date):
            candidates = candidates[
                candidates["listing_date"].le(date)
                & (candidates["delisting_date"].isna() | candidates["delisting_date"].ge(date))
            ]
        if len(candidates) == 1:
            ledger.at[index, "security_id"] = candidates.iloc[0]["security_id"]
    for column in CORPORATE_ACTION_COLUMNS:
        if column not in ledger:
            ledger[column] = pd.NA
    ledger = ledger[CORPORATE_ACTION_COLUMNS]
    ledger = ledger.sort_values(["ticker", "effective_date", "record_date", "event_type"], na_position="last")
    dedupe_date = ledger["effective_date"].fillna(ledger["record_date"])
    ledger = ledger.assign(_dedupe_date=dedupe_date).drop_duplicates(
        ["security_id", "event_type", "_dedupe_date", "verification_status"], keep="last"
    ).drop(columns="_dedupe_date").reset_index(drop=True)
    return ledger, pd.DataFrame(conflicts)


def crawl_corporate_actions(
    paths: Paths,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    tickers: list[str] | None = None,
    max_workers: int = 3,
    pause_seconds: float = 0.20,
) -> dict:
    """Collect VSDC official notices and CafeF ex-date histories into research_v2."""
    if max_workers < 1 or max_workers > 4:
        raise ValueError("max_workers must be between 1 and 4")
    workspace = paths.root / "outputs" / "research_v2"
    raw_root = workspace / "raw" / "corporate_actions"
    normalized = workspace / "normalized"
    reports = workspace / "reports"
    checkpoint_root = raw_root / "checkpoints"
    for directory in [raw_root, normalized, reports, checkpoint_root]:
        directory.mkdir(parents=True, exist_ok=True)
    master_path = paths.normalized / "security_master.parquet"
    if not master_path.exists():
        raise FileNotFoundError("Official security_master.parquet is required.")
    master = pd.read_parquet(master_path)
    requested = sorted(set(t.upper().strip() for t in (tickers or master["ticker"].tolist()) if t.strip()))
    fetched_at = datetime.now(timezone.utc).isoformat()

    def collect_cafef(ticker: str) -> dict:
        checkpoint = checkpoint_root / f"cafef-{ticker}.parquet"
        if checkpoint.exists():
            return {"ticker": ticker, "rows": pd.read_parquet(checkpoint), "resumed": True}
        adapter = CafeFCorporateActionAdapter()
        markup, url = adapter.fetch(ticker)
        archive = _archive_raw_text(raw_root / "cafef", ticker, markup)
        parsed = adapter.parse(markup, ticker, url, fetched_at)
        frame = pd.DataFrame(parsed)
        if not frame.empty:
            frame["raw_checksum"] = archive["sha256"]
            frame = frame[
                pd.to_datetime(frame["ex_date"]).between(pd.Timestamp(start), pd.Timestamp(end))
            ]
        frame.to_parquet(checkpoint, index=False)
        return {"ticker": ticker, "rows": frame, "resumed": False, "raw": archive}

    cafef_rows: list[pd.DataFrame] = []
    cafef_failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(collect_cafef, ticker): ticker for ticker in requested}
        completed = 0
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            try:
                result = future.result()
                if not result["rows"].empty:
                    cafef_rows.append(result["rows"])
            except Exception as exc:
                cafef_failures.append({"ticker": ticker, "error": type(exc).__name__})
            if completed % 25 == 0 or completed == len(futures):
                print(f"[CafeF corporate actions] {completed}/{len(futures)}", flush=True)

    vsdc_rows: list[pd.DataFrame] = []
    vsdc_failures: list[dict] = []
    adapter = VSDCCorporateActionAdapter()
    for position, ticker in enumerate(requested, 1):
        checkpoint = checkpoint_root / f"vsdc-{ticker}.parquet"
        try:
            if checkpoint.exists():
                frame = pd.read_parquet(checkpoint)
            else:
                links, search_pages = adapter.search(ticker, start, end)
                for stem, markup in search_pages:
                    _archive_raw_text(raw_root / "vsdc" / "search", f"{ticker}-{stem}", markup)
                parsed: list[dict] = []
                for item in links:
                    markup = adapter.detail(item["url"])
                    archive = _archive_raw_text(
                        raw_root / "vsdc" / "notices", f"{ticker}-{item['url'].rsplit('/', 1)[-1]}", markup
                    )
                    events = adapter.parse(markup, item["url"], fetched_at)
                    for event in events:
                        event["raw_checksum"] = archive["sha256"]
                    parsed.extend(events)
                    if pause_seconds > 0:
                        time.sleep(pause_seconds)
                frame = pd.DataFrame(parsed)
                frame.to_parquet(checkpoint, index=False)
            if not frame.empty:
                vsdc_rows.append(frame)
        except Exception as exc:
            vsdc_failures.append({"ticker": ticker, "error": type(exc).__name__})
        if position % 25 == 0 or position == len(requested):
            print(f"[VSDC corporate actions] {position}/{len(requested)}", flush=True)
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    nonempty_vsdc = [frame for frame in vsdc_rows if not frame.empty]
    nonempty_cafef = [frame for frame in cafef_rows if not frame.empty]
    official = pd.concat(nonempty_vsdc, ignore_index=True) if nonempty_vsdc else pd.DataFrame()
    reference = pd.concat(nonempty_cafef, ignore_index=True) if nonempty_cafef else pd.DataFrame()
    for frame in (official, reference):
        for column in ["ticker", "event_type"]:
            if column not in frame:
                frame[column] = pd.Series(dtype="object")
    ledger, conflicts = reconcile_corporate_actions(official, reference, master)
    ledger_path = normalized / "corporate_actions.parquet"
    ledger.to_parquet(ledger_path, index=False)
    ledger.to_csv(normalized / "corporate_actions.csv", index=False)
    conflicts.to_csv(reports / "corporate_action_conflicts.csv", index=False)
    coverage = (
        ledger.groupby(["ticker", "verification_status"], dropna=False).size()
        .unstack(fill_value=0).reset_index()
    )
    coverage.to_csv(reports / "corporate_action_coverage.csv", index=False)
    status_counts = ledger["verification_status"].value_counts(dropna=False).to_dict()
    audit = {
        "status": "success_with_unresolved_events" if any(
            key != "verified_cross_source" and value for key, value in status_counts.items()
        ) else "success",
        "source_priority": ["VSDC official notices", "CafeF reference ex-date history"],
        "requested_tickers": len(requested),
        "official_rows": len(official), "reference_rows": len(reference),
        "ledger_rows": len(ledger), "status_counts": status_counts,
        "conflicts": len(conflicts), "cafef_failures": cafef_failures,
        "vsdc_failures": vsdc_failures, "start": start, "end": end,
        "dataset_sha256": sha256_file(ledger_path), "fetched_at": fetched_at,
        "verification_rule": (
            "verified_cross_source requires VSDC official terms plus a matching CafeF ex-date "
            "within 0-7 calendar days before the VSDC record date and compatible material terms"
        ),
        "limitations": [
            "CafeF is an aggregated corroboration source, not official",
            "current CafeF summary pages do not establish historical publication timestamps",
            "unmatched or conflicting events are excluded from total-return construction",
            "absence of a notice in the collected search result is not proof that no event occurred",
        ],
    }
    (reports / "corporate_action_source_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return audit
