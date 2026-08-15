from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    directory = ROOT / "outputs" / "raw" / "fdr_ohlcv"
    archive = ROOT / "outputs" / "archive" / (
        datetime.now().strftime("%Y%m%dT%H%M%S") + "-availability-repair"
    )
    changed_files = 0
    changed_rows = 0
    for path in sorted(directory.glob("*.parquet")):
        frame = pd.read_parquet(path)
        dates = pd.to_datetime(frame["date"], errors="raise")
        available = pd.to_datetime(frame["available_at"], errors="coerce").dt.tz_localize(None)
        invalid = available.le(dates) | available.isna()
        if not invalid.any():
            continue
        candidate = frame.copy()
        candidate.loc[invalid, "available_at"] = dates.loc[invalid] + pd.Timedelta(days=1)
        digest = hashlib.sha256(
            candidate.drop(columns=["raw_checksum"], errors="ignore").to_csv(index=False).encode("utf-8")
        ).hexdigest()
        candidate.loc[invalid, "raw_checksum"] = digest
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / path.name)
        temporary = path.with_suffix(".parquet.availability-repairing")
        candidate.to_parquet(temporary, index=False)
        temporary.replace(path)
        changed_files += 1
        changed_rows += int(invalid.sum())
    print(f"changed_files={changed_files}")
    print(f"changed_rows={changed_rows}")
    print(f"archive={archive if changed_files else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
