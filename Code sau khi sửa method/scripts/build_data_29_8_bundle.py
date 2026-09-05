from __future__ import annotations

"""Build the single-file 29/8 research dataset used by the standalone Colab.

The historical dataset remains the audited source through 2025-12-31.  The
2026 CafeF extension is appended as a clearly labelled *provisional* price
panel.  This script never mutates either source file; it writes a new CSV, ZIP,
schema summary and SHA-256 manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd


CORE_PRICE_COLUMNS = [
    "date",
    "ticker",
    "security_id",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "trading_value",
    "source",
    "source_url",
    "fetched_at",
    "available_at",
    "raw_checksum",
    "parser_version",
    "data_class",
    "adjustment_policy",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    historical_path = args.historical.resolve()
    forward_path = args.forward.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    historical = pd.read_csv(historical_path, low_memory=False)
    forward = pd.read_parquet(forward_path).copy()
    for column in ("date", "ticker", "adjusted_close", "volume", "trading_value"):
        if column not in forward:
            raise ValueError(f"Forward panel is missing required column: {column}")

    # Align the provisional panel to the historical contract without inventing
    # audited corporate-action or benchmark information for 2026.
    aligned = pd.DataFrame(index=forward.index, columns=historical.columns)
    aligned["record_type"] = "PRICE"
    aligned["dataset_id"] = "data_29_8"
    aligned["dataset_version"] = "3.0.0-practical-research"
    aligned["created_at"] = "2026-08-29T00:00:00+07:00"
    aligned["intended_config"] = "AUR_vs_QAUR_practical_grid_and_paper_protocol"
    aligned["research_scope"] = "provisional_2026_extension_not_live_capital_certified"
    for column in CORE_PRICE_COLUMNS:
        if column in forward.columns and column in aligned.columns:
            aligned[column] = forward[column].to_numpy()
    aligned["source"] = "cafef_public_history_overlap_scaled_to_frozen_base"
    aligned["data_class"] = "PROVISIONAL_FORWARD_PRICE"
    aligned["adjustment_policy"] = "provisional_vendor_adjusted_overlap_scaled"
    aligned["verification_status"] = "PROVISIONAL_NOT_CROSS_SOURCE_CERTIFIED"
    aligned["verification_notes"] = (
        "2026 prices passed overlap continuity checks but corporate-action and "
        "total-return semantics are not independently certified"
    )

    historical_price = historical[historical["record_type"].eq("PRICE")].copy()
    historical_other = historical[~historical["record_type"].eq("PRICE")].copy()
    combined_price = pd.concat([historical_price, aligned], ignore_index=True)
    combined_price["date"] = pd.to_datetime(combined_price["date"], errors="coerce")
    combined_price = (
        combined_price.dropna(subset=["date", "ticker", "adjusted_close"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
    )
    combined_price["date"] = combined_price["date"].dt.strftime("%Y-%m-%d")
    combined = pd.concat([combined_price, historical_other], ignore_index=True)

    csv_path = output_dir / "data_29_8.csv"
    zip_path = output_dir / "data_29_8.zip"
    manifest_path = output_dir / "manifest_29_8.json"
    combined.to_csv(csv_path, index=False)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, csv_path.name)

    prices = combined[combined["record_type"].eq("PRICE")].copy()
    price_dates = pd.to_datetime(prices["date"], errors="coerce")
    provisional = prices[prices["data_class"].eq("PROVISIONAL_FORWARD_PRICE")]
    manifest = {
        "dataset_id": "data_29_8",
        "dataset_version": "3.0.0-practical-research",
        "created_at": "2026-08-29T00:00:00+07:00",
        "historical_source": str(historical_path),
        "historical_source_sha256": sha256_file(historical_path),
        "forward_source": str(forward_path),
        "forward_source_sha256": sha256_file(forward_path),
        "csv_filename": csv_path.name,
        "csv_sha256": sha256_file(csv_path),
        "zip_filename": zip_path.name,
        "zip_sha256": sha256_file(zip_path),
        "rows": int(len(combined)),
        "columns": int(len(combined.columns)),
        "record_counts": {
            str(key): int(value)
            for key, value in combined["record_type"].value_counts().sort_index().items()
        },
        "price_tickers": int(prices["ticker"].nunique()),
        "price_start": str(price_dates.min().date()),
        "price_end": str(price_dates.max().date()),
        "provisional_forward_rows": int(len(provisional)),
        "provisional_forward_tickers": int(provisional["ticker"].nunique()),
        "duplicate_price_ticker_dates": int(prices.duplicated(["ticker", "date"]).sum()),
        "missing_adjusted_close_price_rows": int(prices["adjusted_close"].isna().sum()),
        "research_evidence_label": {
            "through_2025_12_31": "historical_audited_research_panel",
            "2026_01_01_through_2026_08_28": "provisional_observed_extension",
            "future_after_2026_08_29": "prospective_only_if_protocol_remains_frozen",
        },
        "live_capital_authorized": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # ASCII-safe stdout keeps the builder portable on Windows consoles that
    # still default to a legacy code page; the saved manifest remains UTF-8.
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
