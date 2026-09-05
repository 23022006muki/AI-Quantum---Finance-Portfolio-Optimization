from __future__ import annotations

"""Build a separate, provisional 2026 forward panel without mutating canonical data."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--workspace-name", default="forward_2026_cafef_120")
    parser.add_argument("--overlap-start", default="2025-11-01")
    parser.add_argument("--forward-start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-28")
    args = parser.parse_args()

    project = args.project.resolve()
    sys.path.insert(0, str(project))
    from src.data_pipeline import Paths
    from src.sources import crawl_cafef_standalone_workspace

    base = pd.read_csv(
        args.base_dataset,
        usecols=[
            "record_type", "date", "ticker", "adjusted_close", "close", "volume",
            "trading_value", "security_id",
        ],
        low_memory=False,
    )
    base = base[base["record_type"].eq("PRICE")].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["adjusted_close"] = pd.to_numeric(base["adjusted_close"], errors="coerce")
    tickers = sorted(base["ticker"].dropna().astype(str).str.upper().unique())
    if len(tickers) < 8:
        raise RuntimeError("Base dataset has fewer than eight price tickers.")

    workspace, source_manifest = crawl_cafef_standalone_workspace(
        Paths(project),
        start=args.overlap_start,
        end=args.end,
        tickers=tickers,
        max_workers=4,
        workspace_name=args.workspace_name,
    )
    workspace = Path(workspace)
    source_path = workspace / "outputs" / "normalized" / "prices.parquet"
    if not source_path.exists():
        raise RuntimeError(f"Forward source panel was not produced: {source_manifest}")
    cafef = pd.read_parquet(source_path)
    cafef["date"] = pd.to_datetime(cafef["date"], errors="coerce")
    cafef["adjusted_close"] = pd.to_numeric(cafef["adjusted_close"], errors="coerce")

    forward_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    forward_start = pd.Timestamp(args.forward_start)
    for ticker in tickers:
        historical = base[
            base["ticker"].eq(ticker)
            & base["date"].between(pd.Timestamp(args.overlap_start), forward_start - pd.Timedelta(days=1))
        ][["date", "adjusted_close"]].dropna().sort_values("date")
        source = cafef[cafef["ticker"].eq(ticker)].sort_values("date").copy()
        overlap = historical.merge(
            source[["date", "adjusted_close"]], on="date", suffixes=("_base", "_cafef"),
        )
        base_returns = overlap["adjusted_close_base"].pct_change(fill_method=None)
        cafef_returns = overlap["adjusted_close_cafef"].pct_change(fill_method=None)
        valid_returns = pd.concat([base_returns, cafef_returns], axis=1).dropna()
        return_corr = float(valid_returns.corr().iloc[0, 1]) if len(valid_returns) >= 3 else np.nan
        median_return_error = float((valid_returns.iloc[:, 0] - valid_returns.iloc[:, 1]).abs().median()) if len(valid_returns) else np.nan
        ratios = overlap["adjusted_close_base"] / overlap["adjusted_close_cafef"].replace(0, np.nan)
        scale = float(ratios.replace([np.inf, -np.inf], np.nan).median()) if ratios.notna().any() else np.nan
        provisional_ok = bool(
            len(overlap) >= 20
            and np.isfinite(scale)
            and np.isfinite(return_corr)
            and return_corr >= 0.98
            and median_return_error <= 0.005
        )
        forward = source[source["date"] >= forward_start].copy()
        if provisional_ok and not forward.empty:
            forward["adjusted_close"] *= scale
            forward["source"] = "cafef_public_history_overlap_scaled_to_frozen_base"
            forward["adjustment_policy"] = "provisional_vendor_adjusted_overlap_scaled"
            forward_parts.append(forward)
        audit_rows.append({
            "ticker": ticker,
            "overlap_rows": len(overlap),
            "return_correlation": return_corr,
            "median_absolute_return_error": median_return_error,
            "scale_to_frozen_base": scale,
            "forward_rows": len(forward),
            "forward_start": forward["date"].min() if len(forward) else pd.NaT,
            "forward_end": forward["date"].max() if len(forward) else pd.NaT,
            "provisional_eligible": provisional_ok and not forward.empty,
        })

    audit = pd.DataFrame(audit_rows)
    audit_path = workspace / "outputs" / "reports" / "forward_overlap_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    if not forward_parts:
        raise RuntimeError("No ticker passed provisional forward overlap checks.")
    forward_panel = (
        pd.concat(forward_parts, ignore_index=True)
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    output_path = workspace / "outputs" / "normalized" / "forward_prices_2026.parquet"
    forward_panel.to_parquet(output_path, index=False)
    eligible_count = int(audit["provisional_eligible"].sum())
    manifest = {
        "status": "provisional_forward_panel",
        "research_role": "untouched_forward_extension_for_frozen_strategy_rules",
        "base_dataset_sha256": sha256_file(args.base_dataset),
        "source_manifest": source_manifest,
        "requested_tickers": len(tickers),
        "eligible_tickers": eligible_count,
        "eligible_fraction": eligible_count / len(tickers),
        "rows": len(forward_panel),
        "start": str(forward_panel["date"].min().date()),
        "end": str(forward_panel["date"].max().date()),
        "forward_panel_sha256": sha256_file(output_path),
        "acceptance_gate": {
            "minimum_eligible_fraction": 0.80,
            "passed": eligible_count / len(tickers) >= 0.80,
        },
        "limitations": [
            "CafeF is an aggregated public source, not the official exchange feed",
            "2026 corporate-action adjustment semantics are provisional",
            "overlap scaling preserves continuity but does not independently certify total-return semantics",
            "not eligible for live capital until cross-source and corporate-action audit passes",
        ],
    }
    manifest_path = workspace / "outputs" / "reports" / "forward_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
