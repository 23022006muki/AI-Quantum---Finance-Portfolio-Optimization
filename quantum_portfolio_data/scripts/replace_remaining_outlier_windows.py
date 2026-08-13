from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    reports = ROOT / "outputs" / "reports"
    checkpoints = ROOT / "outputs" / "raw" / "fdr_ohlcv"
    ledger = pd.read_csv(reports / "outlier_resolution_ledger.csv")
    cross = pd.read_parquet(reports / "cafef_outlier_crosscheck.parquet")
    ledger["date"] = pd.to_datetime(ledger["date"])
    cross["date"] = pd.to_datetime(cross["date"])
    backup = ROOT / "outputs" / "archive" / (
        datetime.now().strftime("%Y%m%dT%H%M%S") + "-pre-window-replacement"
    )
    audit_rows: list[dict] = []

    for ticker in sorted(ledger["ticker"].unique()):
        path = checkpoints / f"{ticker}.parquet"
        source_window = cross[cross["ticker"].eq(ticker)].sort_values("date").copy()
        if not path.exists() or source_window.empty:
            continue
        frame = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
        frame["date"] = pd.to_datetime(frame["date"])
        common = source_window[source_window["date"].isin(frame["date"])].copy()
        if common.empty:
            continue
        anchor = common.iloc[0]
        anchor_index = frame.index[frame["date"].eq(anchor["date"])][0]
        scale = float(frame.loc[anchor_index, "adjusted_close"]) / float(anchor["adjusted_close"])
        changed_indices: list[int] = []
        for item in common.itertuples(index=False):
            index = frame.index[frame["date"].eq(item.date)][0]
            for column in ["open", "high", "low", "close", "adjusted_close"]:
                frame.loc[index, column] = float(getattr(item, column)) * scale
            frame.loc[index, "volume"] = float(item.volume)
            frame.loc[index, "trading_value"] = float(item.volume) * float(frame.loc[index, "close"])
            changed_indices.append(index)
        last_index = max(changed_indices)
        original_last = float(pd.read_parquet(path).sort_values("date").reset_index(drop=True).loc[last_index, "adjusted_close"])
        continuation_factor = float(frame.loc[last_index, "adjusted_close"]) / original_last
        future = frame.index > last_index
        for column in ["open", "high", "low", "close", "adjusted_close", "trading_value"]:
            frame.loc[future, column] = pd.to_numeric(frame.loc[future, column]) * continuation_factor
        touched = frame.index >= min(changed_indices)
        frame.loc[touched, "source"] = "fdr_window_replaced_by_cafef_crosscheck"
        frame.loc[touched, "source_url"] = "https://cafef.vn/du-lieu/lich-su-giao-dich-hose/all-1.chn"
        frame.loc[touched, "parser_version"] = "fdr-cafef-window-replacement-v1"
        digest = hashlib.sha256(
            frame.drop(columns=["raw_checksum"], errors="ignore").to_csv(index=False).encode("utf-8")
        ).hexdigest()
        frame.loc[touched, "raw_checksum"] = digest
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / path.name)
        temporary = path.with_suffix(".parquet.window-replacing")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        audit_rows.append({
            "ticker": ticker, "window_start": str(common["date"].min().date()),
            "window_end": str(common["date"].max().date()),
            "common_rows_replaced": len(common), "anchor_scale": scale,
            "continuation_factor": continuation_factor,
            "evidence": "cafef_public_history",
        })

    result = {
        "status": "success" if len(audit_rows) == ledger["ticker"].nunique() else "partial",
        "tickers_corrected": len(audit_rows), "audit": audit_rows,
        "backup": str(backup), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (reports / "remaining_outlier_window_replacements.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
