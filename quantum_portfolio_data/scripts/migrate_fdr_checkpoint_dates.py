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
    checkpoint_dir = ROOT / "outputs" / "raw" / "fdr_ohlcv"
    archive = ROOT / "outputs" / "archive" / (
        datetime.now().strftime("%Y%m%dT%H%M%S") + "-fdr-date-migration"
    )
    changed_files = 0
    changed_rows = 0
    weekend_before = 0
    weekend_after = 0
    for path in sorted(checkpoint_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        weekend_before += int(dates.dt.dayofweek.ge(5).sum())
        # v1 stored raw_provider_date + one calendar day. Recover the provider
        # date, then apply the corrected next-business-day mapping.
        corrected = (dates - pd.Timedelta(days=1)) + pd.offsets.BDay(1)
        changed = dates.ne(corrected)
        if not changed.any():
            continue
        candidate = frame.copy()
        candidate["date"] = corrected
        if candidate.duplicated(["date", "ticker"], keep=False).any():
            raise RuntimeError(f"Date correction created duplicates for {path.name}")
        candidate["parser_version"] = "finance-datareader-yahoo-hose-v2-business-day"
        digest_frame = candidate.drop(columns=["raw_checksum"], errors="ignore")
        digest = hashlib.sha256(
            digest_frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
        ).hexdigest()
        candidate["raw_checksum"] = digest
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / path.name)
        temporary = path.with_suffix(".parquet.migrating")
        candidate.to_parquet(temporary, index=False)
        temporary.replace(path)
        changed_files += 1
        changed_rows += int(changed.sum())
        weekend_after += int(pd.to_datetime(candidate["date"]).dt.dayofweek.ge(5).sum())
    print(f"changed_files={changed_files}")
    print(f"changed_rows={changed_rows}")
    print(f"weekend_rows_before={weekend_before}")
    print(f"weekend_rows_after={weekend_after}")
    print(f"archive={archive if changed_files else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
