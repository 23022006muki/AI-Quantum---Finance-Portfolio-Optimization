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
    evidence = pd.read_csv(reports / "cafef_cross_source_corrections.csv")
    evidence["date"] = pd.to_datetime(evidence["date"])
    source_archives = sorted(
        (ROOT / "outputs" / "archive").glob("*-cafef-cross-source-correction")
    )
    if not source_archives:
        raise FileNotFoundError("Pre-correction checkpoint archive is missing")
    base_archive = source_archives[-1]
    checkpoint_dir = ROOT / "outputs" / "raw" / "fdr_ohlcv"
    backup = ROOT / "outputs" / "archive" / (
        datetime.now().strftime("%Y%m%dT%H%M%S") + "-pre-segment-scale-correction"
    )
    audit_rows: list[dict] = []

    for ticker, group in evidence.sort_values("date").groupby("ticker"):
        base_path = base_archive / f"{ticker}.parquet"
        current_path = checkpoint_dir / f"{ticker}.parquet"
        if not base_path.exists() or not current_path.exists():
            continue
        frame = pd.read_parquet(base_path).sort_values("date").reset_index(drop=True)
        frame["date"] = pd.to_datetime(frame["date"])
        touched = pd.Series(False, index=frame.index)
        for item in group.itertuples(index=False):
            indices = frame.index[frame["date"].eq(item.date)].tolist()
            if len(indices) != 1 or indices[0] == 0:
                continue
            index = indices[0]
            previous_adjusted = float(frame.loc[index - 1, "adjusted_close"])
            desired = previous_adjusted * (1.0 + float(item.cross_source_return))
            current = float(frame.loc[index, "adjusted_close"])
            if current <= 0:
                continue
            factor = desired / current
            affected = frame.index >= index
            for column in ["open", "high", "low", "close", "adjusted_close", "trading_value"]:
                frame.loc[affected, column] = pd.to_numeric(frame.loc[affected, column]) * factor
            touched |= affected
            audit_rows.append({
                "ticker": ticker, "event_date": str(item.date.date()),
                "original_outlier_return": float(item.old_adjusted_return),
                "cross_source_return": float(item.cross_source_return),
                "segment_scale_factor": factor,
                "affected_rows_from_event": int(affected.sum()),
                "source": "cafef_public_history",
                "source_url": item.source_url,
            })
        if not touched.any():
            continue
        frame.loc[touched, "source"] = "fdr_segment_scaled_by_cafef_crosscheck"
        frame.loc[touched, "source_url"] = "https://cafef.vn/du-lieu/lich-su-giao-dich-hose/all-1.chn"
        frame.loc[touched, "parser_version"] = "fdr-cafef-segment-scale-v1"
        digest = hashlib.sha256(
            frame.drop(columns=["raw_checksum"], errors="ignore").to_csv(index=False).encode("utf-8")
        ).hexdigest()
        frame.loc[touched, "raw_checksum"] = digest
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_path, backup / current_path.name)
        temporary = current_path.with_suffix(".parquet.segment-correcting")
        frame.to_parquet(temporary, index=False)
        temporary.replace(current_path)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(reports / "cafef_segment_scale_corrections.csv", index=False)
    result = {
        "status": "success" if len(audit) == len(evidence) else "partial",
        "events_corrected": len(audit),
        "evidence_events": len(evidence),
        "base_archive": str(base_archive),
        "backup": str(backup),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (reports / "cafef_segment_scale_corrections.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
