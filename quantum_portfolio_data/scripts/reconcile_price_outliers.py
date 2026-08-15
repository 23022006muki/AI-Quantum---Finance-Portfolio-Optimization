from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RESOLUTIONS = {
    "verified_corporate_action",
    "verified_cross_source_correction",
    "verified_vendor_adjustment",
    "unresolved",
    "genuine_market_move",
}


def reconcile(root: Path, cross_source: Path | None = None) -> dict:
    normalized = root / "outputs" / "normalized"
    reports = root / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(normalized / "prices.parquet").sort_values(["ticker", "date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["raw_return"] = prices.groupby("ticker")["close"].pct_change()
    prices["adjusted_return"] = prices.groupby("ticker")["adjusted_close"].pct_change()
    outliers = prices[prices["adjusted_return"].abs() > 0.30].copy()

    actions_path = normalized / "corporate_actions.parquet"
    actions = pd.read_parquet(actions_path) if actions_path.exists() else pd.DataFrame()
    if not actions.empty:
        actions["effective_date"] = pd.to_datetime(actions["effective_date"], errors="coerce")
    cross = pd.DataFrame()
    if cross_source and cross_source.exists():
        cross = pd.read_parquet(cross_source) if cross_source.suffix.lower() == ".parquet" else pd.read_csv(cross_source)
        cross["date"] = pd.to_datetime(cross["date"], errors="coerce")
        cross = cross.rename(columns={"close": "cross_source_value"})
        cross.to_csv(reports / "cross_source_price_comparison.csv", index=False)
    elif (reports / "trading_economics_crosscheck.csv").exists():
        cross = pd.read_csv(reports / "trading_economics_crosscheck.csv")
        cross["date"] = pd.to_datetime(cross["date"], errors="coerce")
        cross = cross.rename(columns={"close": "cross_source_value"})
        cross.to_csv(reports / "cross_source_price_comparison.csv", index=False)
    else:
        pd.DataFrame(columns=["date", "ticker", "cross_source_value", "source", "source_url"]).to_csv(
            reports / "cross_source_price_comparison.csv", index=False
        )

    rows = []
    for item in outliers.itertuples():
        matched = pd.DataFrame()
        if not actions.empty:
            matched = actions[
                actions["ticker"].astype(str).eq(str(item.ticker))
                & actions["effective_date"].between(
                    item.date - pd.Timedelta(days=3), item.date + pd.Timedelta(days=3)
                )
            ]
        cross_match = pd.DataFrame()
        if not cross.empty:
            cross_match = cross[
                cross["ticker"].astype(str).eq(str(item.ticker)) & cross["date"].eq(item.date)
            ]
        action_verified = bool(
            not matched.empty
            and {"source_url", "raw_checksum", "available_at"} <= set(matched.columns)
            and matched["source_url"].astype(str).str.startswith(("http://", "https://")).all()
        )
        resolution = "verified_corporate_action" if action_verified else "unresolved"
        event = matched.iloc[0] if not matched.empty else None
        cross_row = cross_match.iloc[0] if not cross_match.empty else None
        rows.append({
            "ticker": item.ticker,
            "date": item.date,
            "raw_return": item.raw_return,
            "adjusted_return": item.adjusted_return,
            "detected_reason": "return_above_30_percent",
            "matched_event_type": event.get("event_type") if event is not None else None,
            "event_date": event.get("effective_date") if event is not None else None,
            "source": event.get("source") if event is not None else item.source,
            "source_url": event.get("source_url") if event is not None else item.source_url,
            "cross_source_value": cross_row.get("cross_source_value") if cross_row is not None else np.nan,
            "resolution": resolution,
            "reviewer_status": "verified" if action_verified else "pending",
        })
    ledger = pd.DataFrame(rows)
    previous_path = reports / "outlier_resolution_ledger.csv"
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        previous["date"] = pd.to_datetime(previous["date"], errors="coerce")
        previous = previous[
            previous["resolution"].isin(ALLOWED_RESOLUTIONS)
            & previous["reviewer_status"].astype(str).eq("verified")
        ]
        ledger = pd.concat([ledger, previous], ignore_index=True).drop_duplicates(
            ["ticker", "date"], keep="last"
        )
    ledger.to_csv(previous_path, index=False)
    return {
        "outliers": len(ledger),
        "verified": int(ledger["reviewer_status"].eq("verified").sum()) if not ledger.empty else 0,
        "unresolved": int(ledger["resolution"].eq("unresolved").sum()) if not ledger.empty else 0,
        "ledger": str(previous_path),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Reconcile large returns without inventing evidence")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cross-source", type=Path)
    args = parser.parse_args()
    result = reconcile(args.root.resolve(), args.cross_source.resolve() if args.cross_source else None)
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
