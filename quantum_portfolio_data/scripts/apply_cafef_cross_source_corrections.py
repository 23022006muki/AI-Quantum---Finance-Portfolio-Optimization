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
    ledger["date"] = pd.to_datetime(ledger["date"])
    cross = pd.read_parquet(reports / "cafef_outlier_crosscheck.parquet").copy()
    cross["date"] = pd.to_datetime(cross["date"])
    cross = cross.sort_values(["ticker", "date"])
    cross["cross_return"] = cross.groupby("ticker")["adjusted_close"].pct_change()
    archive = ROOT / "outputs" / "archive" / (
        datetime.now().strftime("%Y%m%dT%H%M%S") + "-cafef-cross-source-correction"
    )
    correction_rows: list[dict] = []
    skipped: list[dict] = []
    changed_frames: dict[str, pd.DataFrame] = {}

    for item in ledger.itertuples(index=False):
        match = cross[(cross["ticker"] == item.ticker) & (cross["date"] == item.date)]
        if len(match) != 1 or pd.isna(match.iloc[0]["cross_return"]):
            skipped.append({"ticker": item.ticker, "date": str(item.date.date()), "reason": "cross_return_missing"})
            continue
        cross_row = match.iloc[0]
        cross_return = float(cross_row["cross_return"])
        if abs(cross_return) > 0.15:
            skipped.append({"ticker": item.ticker, "date": str(item.date.date()), "reason": "cross_return_not_stable"})
            continue
        path = checkpoints / f"{item.ticker}.parquet"
        if not path.exists():
            skipped.append({"ticker": item.ticker, "date": str(item.date.date()), "reason": "fdr_checkpoint_missing"})
            continue
        frame = changed_frames.setdefault(item.ticker, pd.read_parquet(path).sort_values("date").reset_index(drop=True))
        frame["date"] = pd.to_datetime(frame["date"])
        indices = frame.index[frame["date"].eq(item.date)].tolist()
        if len(indices) != 1 or indices[0] == 0:
            skipped.append({"ticker": item.ticker, "date": str(item.date.date()), "reason": "primary_row_missing"})
            continue
        index = indices[0]
        previous_adjusted = float(frame.loc[index - 1, "adjusted_close"])
        corrected_adjusted = previous_adjusted * (1.0 + cross_return)
        cross_adjusted = float(cross_row["adjusted_close"])
        if cross_adjusted <= 0:
            skipped.append({"ticker": item.ticker, "date": str(item.date.date()), "reason": "invalid_cross_price"})
            continue
        scale = corrected_adjusted / cross_adjusted
        old_adjusted = float(frame.loc[index, "adjusted_close"])
        for column in ["open", "high", "low", "close"]:
            frame.loc[index, column] = float(cross_row[column]) * scale
        frame.loc[index, "adjusted_close"] = corrected_adjusted
        frame.loc[index, "volume"] = float(cross_row["volume"])
        frame.loc[index, "trading_value"] = float(cross_row["volume"]) * float(frame.loc[index, "close"])
        frame.loc[index, "source"] = "fdr_corrected_by_cafef_public_history"
        frame.loc[index, "source_url"] = cross_row["source_url"]
        frame.loc[index, "parser_version"] = "fdr-cafef-cross-source-correction-v1"
        correction_rows.append({
            "ticker": item.ticker, "date": str(item.date.date()),
            "old_adjusted_close": old_adjusted,
            "corrected_adjusted_close": corrected_adjusted,
            "old_adjusted_return": float(item.adjusted_return),
            "cross_source_return": cross_return,
            "scale_to_primary_adjustment_basis": scale,
            "resolution": "verified_cross_source_correction",
            "source": "cafef_public_history",
            "source_url": cross_row["source_url"],
        })

    for ticker, frame in changed_frames.items():
        if not any(row["ticker"] == ticker for row in correction_rows):
            continue
        path = checkpoints / f"{ticker}.parquet"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / path.name)
        digest = hashlib.sha256(
            frame.drop(columns=["raw_checksum"], errors="ignore").to_csv(index=False).encode("utf-8")
        ).hexdigest()
        corrected_dates = {pd.Timestamp(row["date"]) for row in correction_rows if row["ticker"] == ticker}
        frame.loc[frame["date"].isin(corrected_dates), "raw_checksum"] = digest
        temporary = path.with_suffix(".parquet.correcting")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)

    result = {
        "status": "partial" if skipped else "success",
        "corrections": len(correction_rows),
        "skipped": skipped,
        "archive": str(archive) if correction_rows else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    pd.DataFrame(correction_rows).to_csv(reports / "cafef_cross_source_corrections.csv", index=False)
    (reports / "cafef_cross_source_corrections.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
