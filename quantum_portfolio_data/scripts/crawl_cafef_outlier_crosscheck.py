from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sources import CafeFPublicHistoryAdapter, _normalize_cafef_ohlc


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    ledger = pd.read_csv(ROOT / "outputs" / "reports" / "outlier_resolution_ledger.csv")
    ledger["date"] = pd.to_datetime(ledger["date"])
    adapter = CafeFPublicHistoryAdapter()
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for position, (ticker, group) in enumerate(ledger.groupby("ticker"), start=1):
        for year, year_group in group.groupby(group["date"].dt.year):
            start = year_group["date"].min() - pd.Timedelta(days=20)
            end = year_group["date"].max() + pd.Timedelta(days=5)
            try:
                raw = adapter.daily_ohlc(ticker, str(start.date()), str(end.date()))
                if raw.empty:
                    raise RuntimeError("empty response")
                frame = _normalize_cafef_ohlc(raw, ticker)
                frames.append(frame)
                print(f"[{position:02d}] {ticker} {year}: {len(frame)} rows")
            except Exception as exc:
                failures.append({"ticker": ticker, "year": str(year), "error": type(exc).__name__})
    if not frames:
        raise RuntimeError("No CafeF outlier cross-check data collected")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
    result = result.sort_values(["ticker", "date"])
    result.to_parquet(ROOT / "outputs" / "reports" / "cafef_outlier_crosscheck.parquet", index=False)
    result.to_csv(ROOT / "outputs" / "reports" / "cafef_outlier_crosscheck.csv", index=False)
    pd.DataFrame(failures).to_csv(
        ROOT / "outputs" / "reports" / "cafef_outlier_crosscheck_failures.csv", index=False
    )
    print(f"records={len(result)}")
    print(f"tickers={result['ticker'].nunique()}")
    print(f"failures={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
