"""Create the versioned dataset bundle for the revised methodology."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "quantum_portfolio_data" / "colab_data" / "ai_quantum_complete_dataset.csv"
OUTPUT_DIR = ROOT / "data sau khi sửa method"
OUTPUT_CSV = OUTPUT_DIR / "data_sau_khi_sua_method.csv"
OUTPUT_ZIP = OUTPUT_DIR / "data_sau_khi_sua_method.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, low_memory=False)
    metadata = frame["record_type"].eq("METADATA")
    frame.loc[metadata, "dataset_id"] = "data_sau_khi_sua_method"
    frame.loc[metadata, "dataset_version"] = "2.0.0-aur-qaur"
    frame.loc[metadata, "created_at"] = "2026-08-28T00:00:00+07:00"
    frame.loc[metadata, "intended_config"] = "aur_qaur_shared_xyqaoa_standalone_full"
    frame.loc[metadata, "research_scope"] = (
        "point_in_time_HOSE_AUR_vs_QAUR_shared_cardinality_QUBO_XYQAOA"
    )
    frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(OUTPUT_CSV, OUTPUT_CSV.name)

    prices = frame[frame["record_type"].eq("PRICE")].copy()
    price_dates = pd.to_datetime(prices["date"], errors="coerce")
    counts = frame["record_type"].value_counts().to_dict()
    manifest = {
        "dataset_id": "data_sau_khi_sua_method",
        "dataset_version": "2.0.0-aur-qaur",
        "created_at": "2026-08-28T00:00:00+07:00",
        "methodology": "AUR_vs_QAUR_with_shared_downstream_XYQAOA",
        "source_dataset": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "csv_filename": OUTPUT_CSV.name,
        "csv_sha256": sha256(OUTPUT_CSV),
        "zip_filename": OUTPUT_ZIP.name,
        "zip_sha256": sha256(OUTPUT_ZIP),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "record_counts": {str(k): int(v) for k, v in counts.items()},
        "price_tickers": int(prices["ticker"].nunique()),
        "price_start": str(price_dates.min().date()),
        "price_end": str(price_dates.max().date()),
        "duplicate_price_ticker_dates": int(prices.duplicated(["ticker", "date"]).sum()),
        "missing_adjusted_close_price_rows": int(prices["adjusted_close"].isna().sum()),
        "zip_contract": "ZIP contains exactly one CSV for direct Colab upload",
        "data_change_scope": "metadata versioning only; market observations are unchanged",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    schema = {
        "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in frame.dtypes.items()],
        "required_by_colab": [
            "record_type", "date", "ticker", "adjusted_close", "volume", "trading_value"
        ],
        "record_types": sorted(frame["record_type"].dropna().unique().tolist()),
    }
    (OUTPUT_DIR / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
