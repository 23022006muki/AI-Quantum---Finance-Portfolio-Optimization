"""Validate one complete Colab CSV and restore the Data 17/8 runtime workspace."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import (
    PRICE_COLUMNS,
    Paths,
    build_universe,
    leakage_audit,
    sha256_file,
    validate_data,
)


DEFAULT_WORKSPACE = ROOT / "outputs" / "Data 17_8"

BENCHMARK_COLUMNS = [
    "date", "benchmark", "total_return_index", "index_type", "methodology_url",
    "available_at", "source", "source_url", "fetched_at", "data_class",
]
SECURITY_COLUMNS = [
    "security_id", "ticker", "company_name", "exchange", "isin", "figi",
    "hose_security_id", "listing_date", "delisting_date", "effective_from",
    "effective_to", "available_at", "source", "source_url", "fetched_at",
    "history_method", "data_class", "raw_checksum",
]
ACTION_COLUMNS = [
    "security_id", "ticker", "event_type", "announcement_date", "record_date",
    "ex_date", "effective_date", "payment_date", "cash_dividend_per_share",
    "stock_dividend_ratio", "bonus_share_ratio", "split_ratio",
    "reverse_split_ratio", "rights_ratio", "rights_subscription_price",
    "adjustment_factor", "currency", "source", "source_url",
    "corroboration_source", "corroboration_url", "fetched_at", "available_at",
    "raw_checksum", "parser_version", "verification_status", "verification_notes",
]


def require_columns(frame: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def parse_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output:
            output[column] = pd.to_datetime(output[column], errors="coerce", format="mixed")
    return output


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def import_dataset(csv_path: Path, workspace: Path, replace: bool = True) -> dict:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    if "record_type" not in frame:
        raise ValueError("CSV must contain record_type.")
    counts = frame["record_type"].value_counts().to_dict()
    required_types = {"METADATA", "PRICE", "BENCHMARK", "SECURITY", "CORPORATE_ACTION"}
    missing_types = sorted(required_types - set(counts))
    if missing_types:
        raise ValueError(f"CSV is missing record types: {missing_types}")

    prices = frame.loc[frame.record_type.eq("PRICE")].copy()
    benchmark = frame.loc[frame.record_type.eq("BENCHMARK")].copy()
    securities = frame.loc[frame.record_type.eq("SECURITY")].copy()
    actions = frame.loc[frame.record_type.eq("CORPORATE_ACTION")].copy()
    require_columns(prices, PRICE_COLUMNS, "PRICE")
    require_columns(benchmark, BENCHMARK_COLUMNS, "BENCHMARK")
    require_columns(securities, SECURITY_COLUMNS + ["runtime_eligible"], "SECURITY")
    require_columns(actions, ACTION_COLUMNS, "CORPORATE_ACTION")

    prices = parse_dates(prices[PRICE_COLUMNS], ["date", "available_at"])
    benchmark = parse_dates(
        benchmark[BENCHMARK_COLUMNS], ["date", "available_at"]
    )
    full_master = parse_dates(
        securities[SECURITY_COLUMNS],
        ["listing_date", "delisting_date", "effective_from", "effective_to", "available_at"],
    )
    runtime_mask = bool_series(securities["runtime_eligible"])
    runtime_master = full_master.loc[runtime_mask.to_numpy()].copy()
    if "research_eligibility_status" in securities:
        runtime_master["research_eligibility_status"] = securities.loc[
            runtime_mask, "research_eligibility_status"
        ].to_numpy()
        runtime_master["research_eligibility_as_of"] = pd.Timestamp("2026-08-17")
    actions = parse_dates(
        actions[ACTION_COLUMNS],
        ["announcement_date", "record_date", "ex_date", "effective_date", "payment_date", "available_at"],
    )

    if prices.empty or runtime_master.empty or benchmark.empty:
        raise ValueError("PRICE, runtime SECURITY and BENCHMARK records must be non-empty.")
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker-date rows in PRICE records.")
    if not set(prices["ticker"].astype(str)) <= set(runtime_master["ticker"].astype(str)):
        raise ValueError("Every PRICE ticker must exist in the runtime security master.")
    price_dates = set(prices["date"].dropna())
    benchmark_dates = set(benchmark["date"].dropna())
    if price_dates != benchmark_dates:
        raise ValueError("BENCHMARK dates must exactly match PRICE trading dates.")

    resolved = workspace.resolve()
    project_outputs = (ROOT / "outputs").resolve()
    if project_outputs not in resolved.parents:
        raise ValueError(f"Workspace must stay under {project_outputs}: {resolved}")
    if workspace.exists() and replace:
        shutil.rmtree(workspace)
    paths = Paths(workspace)
    for directory in [paths.normalized, paths.curated, paths.reports, paths.raw]:
        directory.mkdir(parents=True, exist_ok=True)

    prices.to_parquet(paths.normalized / "prices.parquet", index=False)
    benchmark.to_parquet(paths.normalized / "benchmark.parquet", index=False)
    runtime_master.to_parquet(paths.normalized / "security_master.parquet", index=False)
    full_master.to_parquet(paths.normalized / "security_master_full.parquet", index=False)
    actions.to_parquet(paths.normalized / "corporate_actions.parquet", index=False)

    price_hash = sha256_file(paths.normalized / "prices.parquet")
    contract = {
        "dataset": "Data 17/8 - complete CSV import",
        "adjustment_policy": "verified_vendor_total_return_adjusted",
        "source": "cafef_raw_kbs_adjusted_crosscheck_packaged_csv",
        "source_url": "local-upload://ai_quantum_complete_dataset.csv",
        "methodology": (
            "The CSV preserves the previously verified CafeF raw OHLC and KBS adjusted "
            "close observations with row-level provenance; no prices are recomputed on import."
        ),
        "certified_by": "colab-complete-csv-schema-and-hash-audit",
        "certified_at": datetime.now(timezone.utc).isoformat(),
        "input_csv_sha256": sha256_file(csv_path),
        "output_price_dataset_sha256": price_hash,
    }
    (paths.normalized / "price_adjustment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    crosscheck = {
        "dataset": contract["dataset"],
        "status": "packaged_verified_panel",
        "requested_tickers": int(prices["ticker"].nunique()),
        "cafef_series_collected": int(prices["ticker"].nunique()),
        "cross_source_verified_tickers": int(prices["ticker"].nunique()),
        "rows": len(prices),
        "dates": int(prices["date"].nunique()),
        "failures": [],
        "raw_price_source": "CafeF public PriceHistory endpoint (preserved package)",
        "adjusted_price_source": "KBS public endpoint through vnstock (preserved package)",
        "adjustment_policy": "verified_vendor_total_return_adjusted",
        "sha256": price_hash,
    }
    (paths.reports / "cafef_price_crosscheck_audit.json").write_text(
        json.dumps(crosscheck, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    quality, coverage = validate_data(paths)
    universe = build_universe(
        paths,
        rebalance="monthly",
        definition="hose_all_listed",
        max_assets=300,
        liquidity_lookback_days=60,
        minimum_observations=40,
    )
    leak = leakage_audit(paths)
    exploratory_permitted = bool(
        quality["status"] == "pass"
        and leak["status"] in {"pass", "pass_with_limitations"}
        and not prices.empty
        and not benchmark.empty
    )
    packaged_audit = {
        "dataset": "Data 17/8 - complete CSV import",
        "status": "blocked",
        "research_ready": False,
        "exploratory_run_permitted": exploratory_permitted,
        "checks": {
            "price_panel": {"passed": quality["status"] == "pass", "rows": len(prices)},
            "official_total_return_benchmark": {"passed": not benchmark.empty, "rows": len(benchmark)},
            "corporate_action_ledger": {"passed": not actions.empty, "rows": len(actions)},
            "company_document_repository": {"passed": False, "reason": "not_embedded_in_single_csv"},
        },
        "blockers": ["company_document_repository_not_embedded_in_single_csv"],
        "interpretation": (
            "The single CSV is complete for the exploratory model runtime. Raw disclosure "
            "binaries and optional PIT financial features are not embedded, so it is not a "
            "confirmatory full-HOSE research package."
        ),
    }
    (paths.reports / "DATA_17_8_AUDIT.json").write_text(
        json.dumps(packaged_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result = {
        "input_csv": str(csv_path),
        "input_csv_sha256": sha256_file(csv_path),
        "workspace": str(workspace),
        "record_counts": {str(key): int(value) for key, value in counts.items()},
        "price_rows": len(prices),
        "runtime_tickers": int(prices["ticker"].nunique()),
        "full_master_tickers": int(full_master["ticker"].nunique()),
        "benchmark_rows": len(benchmark),
        "corporate_action_rows": len(actions),
        "universe_rows": len(universe),
        "quality_status": quality["status"],
        "leakage_status": leak["status"],
        "exploratory_run_permitted": exploratory_permitted,
        "confirmatory_audit_status": packaged_audit["status"],
        "coverage_rows": len(coverage),
    }
    (paths.reports / "COLAB_CSV_IMPORT_REPORT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--no-replace", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        import_dataset(args.csv.resolve(), args.workspace.resolve(), not args.no_replace),
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
