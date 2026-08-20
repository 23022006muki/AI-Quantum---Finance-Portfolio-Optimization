"""Build the single-file CSV package consumed by the Colab upload workflow.

The CSV uses ``record_type`` to store four normalized tables without losing their
native columns: PRICE, BENCHMARK, SECURITY and CORPORATE_ACTION.  A METADATA row
declares the package version and intended configuration.  The importer restores
the tables before invoking the unchanged research engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "outputs" / "Data 17_8"
DEFAULT_OUTPUT = ROOT / "colab_data" / "ai_quantum_complete_dataset.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def typed(frame: pd.DataFrame, record_type: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "record_type", record_type)
    return output


def build_dataset(workspace: Path, output: Path) -> dict:
    normalized = workspace / "outputs" / "normalized"
    required = {
        "prices": normalized / "prices.parquet",
        "benchmark": normalized / "benchmark.parquet",
        "security_master": normalized / "security_master.parquet",
        "security_master_full": normalized / "security_master_full.parquet",
        "corporate_actions": normalized / "corporate_actions.parquet",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Data 17/8 tables: {missing}")

    prices = pd.read_parquet(required["prices"])
    benchmark = pd.read_parquet(required["benchmark"])
    runtime_master = pd.read_parquet(required["security_master"])
    full_master = pd.read_parquet(required["security_master_full"])
    actions = pd.read_parquet(required["corporate_actions"])

    runtime_tickers = set(runtime_master["ticker"].astype(str))
    runtime_status = runtime_master.set_index("ticker").get(
        "research_eligibility_status", pd.Series(dtype=object)
    )
    full_master["runtime_eligible"] = full_master["ticker"].astype(str).isin(runtime_tickers)
    full_master["research_eligibility_status"] = (
        full_master["ticker"].map(runtime_status).fillna("excluded_from_complete_case")
    )

    metadata = pd.DataFrame([{
        "record_type": "METADATA",
        "dataset_id": "data_17_8_colab_complete_csv",
        "dataset_version": "2026-08-20",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intended_config": "configs/data_17_8.yaml",
        "research_scope": "exploratory_complete_case_not_confirmatory_full_hose",
        "price_rows": len(prices),
        "runtime_tickers": prices["ticker"].nunique(),
        "full_master_tickers": full_master["ticker"].nunique(),
        "benchmark_rows": len(benchmark),
        "corporate_action_rows": len(actions),
    }])
    tables = [
        metadata,
        typed(prices, "PRICE"),
        typed(benchmark, "BENCHMARK"),
        typed(full_master, "SECURITY"),
        typed(actions, "CORPORATE_ACTION"),
    ]
    combined = pd.concat(tables, ignore_index=True, sort=False)
    first = ["record_type", "dataset_id", "dataset_version", "created_at"]
    ordered = first + [column for column in combined.columns if column not in first]
    combined = combined[ordered]

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig", lineterminator="\n")
    digest = sha256_file(output)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(output, arcname=output.name)

    manifest = {
        "dataset": str(output),
        "archive": str(archive),
        "sha256": digest,
        "archive_sha256": sha256_file(archive),
        "bytes": output.stat().st_size,
        "archive_bytes": archive.stat().st_size,
        "rows": len(combined),
        "record_counts": {
            str(key): int(value)
            for key, value in combined["record_type"].value_counts().sort_index().items()
        },
        "columns": combined.columns.tolist(),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(
        build_dataset(args.workspace.resolve(), args.output.resolve()),
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
