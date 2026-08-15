from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_pipeline import Paths, sha256_file


MATERIAL_EVENT_TYPES = {
    "CASH_DIVIDEND", "STOCK_DIVIDEND", "BONUS_SHARE", "STOCK_SPLIT",
    "REVERSE_SPLIT", "RIGHTS_ISSUE", "SHARE_CONVERSION", "MERGER",
}


def corporate_action_total_return(
    previous_close: float,
    current_close: float,
    *,
    cash_dividend_per_share: float = 0.0,
    stock_dividend_ratio: float = 0.0,
    bonus_share_ratio: float = 0.0,
    split_ratio: float = 1.0,
    reverse_split_ratio: float = 1.0,
    rights_ratio: float = 0.0,
    rights_subscription_price: float = 0.0,
) -> float:
    """Economic holding-period return for one pre-event share.

    Ratios follow the v2 contract: stock/bonus/rights ratios are new shares per old
    share, split_ratio is new shares per old share, and reverse_split_ratio is old
    shares per new share. Rights are assumed exercised and their subscription cash
    outflow is deducted. All values are per pre-event share.
    """
    if previous_close <= 0 or current_close <= 0:
        raise ValueError("Prices must be strictly positive.")
    if split_ratio <= 0 or reverse_split_ratio <= 0:
        raise ValueError("Split ratios must be strictly positive.")
    if min(
        cash_dividend_per_share, stock_dividend_ratio, bonus_share_ratio,
        rights_ratio, rights_subscription_price,
    ) < 0:
        raise ValueError("Corporate-action terms must be non-negative.")
    share_multiplier = (
        (1.0 + stock_dividend_ratio + bonus_share_ratio + rights_ratio)
        * split_ratio / reverse_split_ratio
    )
    terminal_wealth = (
        current_close * share_multiplier
        + cash_dividend_per_share
        - rights_ratio * rights_subscription_price
    )
    return terminal_wealth / previous_close - 1.0


def _numeric(row: pd.Series | dict[str, Any], name: str, default: float) -> float:
    value = row.get(name, default)
    return default if pd.isna(value) else float(value)


def _aggregate_events(events: pd.DataFrame) -> dict[str, float]:
    return {
        "cash_dividend_per_share": float(events["cash_dividend_per_share"].fillna(0).sum()),
        "stock_dividend_ratio": float(events["stock_dividend_ratio"].fillna(0).sum()),
        "bonus_share_ratio": float(events["bonus_share_ratio"].fillna(0).sum()),
        "split_ratio": float(events["split_ratio"].dropna().prod()) if events["split_ratio"].notna().any() else 1.0,
        "reverse_split_ratio": (
            float(events["reverse_split_ratio"].dropna().prod())
            if events["reverse_split_ratio"].notna().any() else 1.0
        ),
        "rights_ratio": float(events["rights_ratio"].fillna(0).sum()),
        "rights_subscription_price": (
            float(np.average(
                events.loc[events["rights_ratio"].fillna(0).gt(0), "rights_subscription_price"].fillna(0),
                weights=events.loc[events["rights_ratio"].fillna(0).gt(0), "rights_ratio"],
            )) if events["rights_ratio"].fillna(0).gt(0).any() else 0.0
        ),
    }


def _build_ticker_total_return(
    prices: pd.DataFrame, verified_events: pd.DataFrame,
) -> pd.DataFrame:
    frame = prices.sort_values("date").copy()
    frame["price_return"] = frame["raw_close"].pct_change()
    frame["source_adjusted_return"] = frame["source_adjusted_close"].pct_change()
    frame["total_return"] = frame["price_return"]
    frame["adjustment_known_at"] = pd.NaT
    frame["adjustment_source"] = "no_event_raw_close"
    event_groups = {
        pd.Timestamp(date): group for date, group in verified_events.groupby("effective_date")
    }
    dates = list(pd.to_datetime(frame["date"]))
    for position in range(1, len(frame)):
        date = dates[position]
        events = event_groups.get(date)
        if events is None or events.empty:
            continue
        terms = _aggregate_events(events)
        frame.iloc[position, frame.columns.get_loc("total_return")] = corporate_action_total_return(
            float(frame.iloc[position - 1]["raw_close"]),
            float(frame.iloc[position]["raw_close"]),
            **terms,
        )
        frame.iloc[position, frame.columns.get_loc("adjustment_known_at")] = events["available_at"].max()
        frame.iloc[position, frame.columns.get_loc("adjustment_source")] = "verified_corporate_action_ledger"
    frame["total_return"] = frame["total_return"].replace([np.inf, -np.inf], np.nan)
    initial_close = float(frame["raw_close"].iloc[0])
    total_return_index = (1.0 + frame["total_return"].fillna(0.0)).cumprod() * initial_close
    frame["research_adjusted_close"] = total_return_index
    scale = frame["research_adjusted_close"] / frame["raw_close"]
    for field in ["open", "high", "low"]:
        frame[f"research_adjusted_{field}"] = frame[f"raw_{field}"] * scale
    frame["adjustment_factor"] = scale
    return frame


def build_price_adjustment_v2(paths: Paths) -> dict:
    """Build an auditable total-return candidate and fail the research gate on gaps."""
    workspace = paths.root / "outputs" / "research_v2"
    normalized = workspace / "normalized"
    reports = workspace / "reports"
    normalized.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    price_path = paths.normalized / "prices.parquet"
    actions_path = normalized / "corporate_actions.parquet"
    if not price_path.exists() or not actions_path.exists():
        raise FileNotFoundError("Canonical prices and research_v2 corporate_actions are required.")
    prices = pd.read_parquet(price_path).copy()
    actions = pd.read_parquet(actions_path).copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    for column in ["effective_date", "available_at", "record_date", "announcement_date"]:
        actions[column] = pd.to_datetime(actions[column], errors="coerce")
    renamed = prices.rename(columns={
        "open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close",
        "adjusted_close": "source_adjusted_close",
    })
    verified = actions[
        actions["verification_status"].eq("verified_cross_source")
        & actions["event_type"].isin(MATERIAL_EVENT_TYPES)
        & actions["effective_date"].notna()
        & actions["available_at"].le(actions["effective_date"])
    ].copy()
    unresolved = actions[
        actions["event_type"].isin(MATERIAL_EVENT_TYPES)
        & ~actions.index.isin(verified.index)
        & actions["effective_date"].fillna(actions["record_date"]).between(
            prices["date"].min(), prices["date"].max()
        )
    ].copy()
    result_frames = []
    for ticker, ticker_prices in renamed.groupby("ticker", sort=True):
        result_frames.append(_build_ticker_total_return(
            ticker_prices, verified[verified["ticker"].eq(ticker)]
        ))
    output = pd.concat(result_frames, ignore_index=True).sort_values(["ticker", "date"])
    output["adjustment_version"] = "corporate-actions-v2"
    output["research_adjustment_status"] = np.where(
        output["ticker"].isin(set(unresolved["ticker"])),
        "blocked_unresolved_material_event", "candidate_no_known_unresolved_event",
    )
    output_path = normalized / "prices_total_return.parquet"
    output.to_parquet(output_path, index=False)

    changes = output[
        (output["source_adjusted_return"] - output["price_return"]).abs().gt(0.01)
    ].copy()
    verified_keys = set(zip(verified["ticker"], verified["effective_date"]))
    changes["matched_verified_event"] = [
        (ticker, pd.Timestamp(date)) in verified_keys
        for ticker, date in zip(changes["ticker"], changes["date"])
    ]
    unmatched = changes[~changes["matched_verified_event"]].copy()
    changes.to_csv(reports / "adjustment_reconciliation.csv", index=False)
    unmatched.to_csv(reports / "unmatched_adjustment_changes.csv", index=False)

    # HOSE's ordinary daily price band is approximately +/-7%. For non-adjacent
    # observations, compound the band over the number of business-day intervals
    # rather than misclassifying a long suspension or holiday gap as a one-day move.
    output["previous_observation_date"] = output.groupby("ticker")["date"].shift(1)
    valid_gap = output["previous_observation_date"].notna()
    business_gaps = np.ones(len(output), dtype=int)
    business_gaps[valid_gap] = [
        max(1, int(np.busday_count(previous.date(), current.date())))
        for previous, current in zip(
            output.loc[valid_gap, "previous_observation_date"],
            output.loc[valid_gap, "date"],
        )
    ]
    output["business_day_gap"] = business_gaps
    output["compounded_upper_price_band"] = np.power(1.07, business_gaps) - 1.0
    output["compounded_lower_price_band"] = np.power(0.93, business_gaps) - 1.0
    band_anomalies = output[
        output["price_return"].gt(output["compounded_upper_price_band"] + 1e-12)
        | output["price_return"].lt(output["compounded_lower_price_band"] - 1e-12)
    ].copy()
    all_event_keys = set(zip(
        actions["ticker"], actions["effective_date"].fillna(actions["record_date"])
    ))
    band_anomalies["matched_any_corporate_action"] = [
        (ticker, pd.Timestamp(date)) in all_event_keys
        for ticker, date in zip(band_anomalies["ticker"], band_anomalies["date"])
    ]
    unexplained_band_anomalies = band_anomalies[
        ~band_anomalies["matched_any_corporate_action"]
    ].copy()
    band_anomalies.to_csv(reports / "price_limit_anomalies.csv", index=False)

    windows = []
    for _, event in verified.iterrows():
        ticker_prices = output[output["ticker"].eq(event["ticker"])].reset_index(drop=True)
        matches = ticker_prices.index[ticker_prices["date"].eq(event["effective_date"])].tolist()
        if not matches:
            windows.append({
                "ticker": event["ticker"], "event_type": event["event_type"],
                "effective_date": event["effective_date"], "window_status": "missing_price_date",
            })
            continue
        center = matches[0]
        for relative, row in ticker_prices.iloc[max(0, center - 3):center + 4].iterrows():
            windows.append({
                "ticker": event["ticker"], "event_type": event["event_type"],
                "effective_date": event["effective_date"], "date": row["date"],
                "relative_observation": relative - center,
                "price_return": row["price_return"],
                "source_adjusted_return": row["source_adjusted_return"],
                "research_total_return": row["total_return"],
                "window_status": "observed",
            })
    pd.DataFrame(windows).to_csv(reports / "event_window_audit.csv", index=False)
    unresolved.to_csv(reports / "unresolved_material_events.csv", index=False)

    contract_path = paths.root / "docs" / "contracts" / "price_adjustment_contract.v2.json"
    contract_hash = sha256_file(contract_path) if contract_path.exists() else None
    blockers = []
    if not unresolved.empty:
        blockers.append(f"{len(unresolved)} material corporate-action rows are unresolved")
    if not unmatched.empty:
        blockers.append(f"{len(unmatched)} source-adjustment changes above 1 percentage point are unmatched")
    if not unexplained_band_anomalies.empty:
        blockers.append(
            f"{len(unexplained_band_anomalies)} raw price moves exceed the compounded "
            "HOSE band without a matched event"
        )
    if verified.empty:
        blockers.append("no cross-source verified corporate action was available")
    audit = {
        "status": "blocked" if blockers else "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_price_sha256": sha256_file(price_path),
        "corporate_action_sha256": sha256_file(actions_path),
        "contract_sha256": contract_hash,
        "output_price_sha256": sha256_file(output_path),
        "adjustment_version": "corporate-actions-v2",
        "price_rows": len(output), "tickers": int(output["ticker"].nunique()),
        "verified_events_applied": len(verified),
        "unresolved_material_events": len(unresolved),
        "source_adjustment_changes_over_1pp": len(changes),
        "unmatched_source_adjustment_changes_over_1pp": len(unmatched),
        "price_band_anomalies": len(band_anomalies),
        "unexplained_price_band_anomalies": len(unexplained_band_anomalies),
        "blockers": blockers,
        "research_eligible": not blockers,
        "note": (
            "The candidate file preserves raw and vendor-adjusted fields separately. "
            "A blocked candidate must not be used for confirmatory research."
        ),
    }
    (reports / "price_adjustment_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    methodology = [
        "# Price adjustment methodology v2", "",
        f"Generated: {audit['generated_at']}", "",
        "The implementation preserves raw OHLC and source-adjusted close. Research total return is constructed only from cross-source verified events using the versioned contract in `docs/contracts/price_adjustment_contract.v2.json`.",
        "",
        "For one share held before an event, terminal wealth equals current raw close multiplied by the post-event share count, plus cash dividend, less the subscription cash paid for exercised rights. Stock dividends, bonus shares, splits, reverse splits and rights therefore have distinct terms; rights are not treated as a split.",
        "",
        f"Gate status: **{audit['status']}**. Verified events applied: {len(verified)}. Unresolved material events: {len(unresolved)}. Unmatched source-adjustment changes: {len(unmatched)}.",
        "",
        "A blocked result is a diagnostic artifact only and cannot be promoted into research mode.",
    ]
    (reports / "PRICE_ADJUSTMENT_METHODOLOGY.md").write_text(
        "\n".join(methodology) + "\n", encoding="utf-8"
    )
    return audit


def build_return_only_adjustment_counterfactual(
    paths: Paths,
    baseline_experiment: Path | None = None,
) -> dict:
    """Reprice the frozen Data A holdings under three return definitions.

    This is a direct return-only counterfactual. It intentionally does not retrain the
    model, rebuild AUR, solve a new QUBO or change weights.
    """
    baseline = baseline_experiment or (
        paths.root / "outputs" / "Data A" / "outputs" / "experiments"
        / "20260813T164535-21c9b569ce"
    )
    workspace = paths.root / "outputs" / "research_v2"
    panel_path = workspace / "normalized" / "prices_total_return.parquet"
    audit_path = workspace / "reports" / "price_adjustment_audit.json"
    if not panel_path.exists() or not audit_path.exists():
        raise FileNotFoundError("Build price adjustment v2 before the counterfactual.")
    reports = workspace / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    weights = pd.read_csv(baseline / "weights.csv")
    weights = weights[weights["strategy"].eq("full_pipeline_xy_qaoa")]
    folds = pd.read_csv(baseline / "fold_manifest.csv")
    for column in ["test_start", "test_end"]:
        folds[column] = pd.to_datetime(folds[column])
    costs = pd.read_csv(baseline / "cost_ledger.csv")
    costs = costs[costs["strategy"].eq("full_pipeline_xy_qaoa")].set_index("fold")
    baseline_dates = pd.read_csv(baseline / "portfolio_returns.csv")
    baseline_dates = baseline_dates[baseline_dates["strategy"].eq("full_pipeline_xy_qaoa")]
    baseline_dates["date"] = pd.to_datetime(baseline_dates["date"])
    definitions = {
        "raw_close": "price_return",
        "source_adjusted": "source_adjusted_return",
        "research_total_return_candidate": "total_return",
    }
    daily_rows: list[dict] = []
    fold_rows: list[dict] = []
    ticker_rows: list[dict] = []
    for fold, fold_weights in weights.groupby("fold"):
        fold = int(fold)
        calendar = pd.DatetimeIndex(sorted(
            baseline_dates.loc[baseline_dates["fold"].eq(fold), "date"].unique()
        ))
        if calendar.empty:
            continue
        tickers = fold_weights["ticker"].astype(str).tolist()
        initial = fold_weights.set_index("ticker")["weight"].astype(float).reindex(tickers)
        cost = float(costs.at[fold, "transaction_cost"]) if fold in costs.index else 0.0
        for definition, return_column in definitions.items():
            matrix = panel[panel["ticker"].isin(tickers)].pivot(
                index="date", columns="ticker", values=return_column
            ).reindex(index=calendar, columns=tickers)
            # Missing observations on the common HOSE calendar are marked-to-market
            # at zero for this frozen-holdings diagnostic and counted explicitly.
            missing = int(matrix.isna().sum().sum())
            matrix = matrix.fillna(0.0)
            holdings = initial.to_numpy(dtype=float)
            holdings = holdings / holdings.sum()
            gross_fold: list[float] = []
            net_fold: list[float] = []
            for position, (date, returns) in enumerate(matrix.iterrows()):
                vector = returns.to_numpy(dtype=float)
                gross = float(holdings @ vector)
                net = (1.0 + gross) * (1.0 - cost) - 1.0 if position == 0 else gross
                gross_fold.append(gross)
                net_fold.append(net)
                daily_rows.append({
                    "fold": fold, "date": date, "return_definition": definition,
                    "gross_return": gross, "net_return": net,
                })
                holdings = holdings * (1.0 + vector)
                if holdings.sum() > 0:
                    holdings /= holdings.sum()
            fold_rows.append({
                "fold": fold, "return_definition": definition,
                "gross_cumulative_return": float(
                    np.prod(1.0 + np.asarray(gross_fold, dtype=float)) - 1.0
                ),
                "net_cumulative_return": float(
                    np.prod(1.0 + np.asarray(net_fold, dtype=float)) - 1.0
                ),
                "transaction_cost_fraction": cost,
                "missing_asset_day_returns_marked_zero": missing,
            })
        ticker_slice = panel[panel["ticker"].isin(tickers) & panel["date"].isin(calendar)]
        for ticker, ticker_data in ticker_slice.groupby("ticker"):
            ticker_rows.append({
                "fold": fold, "ticker": ticker,
                "initial_weight": float(initial.get(ticker, 0.0)),
                "source_minus_raw_return_sum": float(
                    (ticker_data["source_adjusted_return"] - ticker_data["price_return"]).sum()
                ),
                "research_minus_raw_return_sum": float(
                    (ticker_data["total_return"] - ticker_data["price_return"]).sum()
                ),
            })
    daily = pd.DataFrame(daily_rows)
    by_fold = pd.DataFrame(fold_rows)
    by_ticker = pd.DataFrame(ticker_rows)
    daily.to_csv(reports / "adjustment_return_only_daily.csv", index=False)
    by_fold.to_csv(reports / "adjustment_impact_by_fold.csv", index=False)
    by_ticker.to_csv(reports / "adjustment_impact_by_ticker.csv", index=False)
    totals = []
    for definition, frame in daily.groupby("return_definition"):
        totals.append({
            "return_definition": definition,
            "gross_cumulative_return": float((1.0 + frame.sort_values(["fold", "date"])["gross_return"]).prod() - 1.0),
            "net_cumulative_return": float((1.0 + frame.sort_values(["fold", "date"])["net_return"]).prod() - 1.0),
        })
    summary = pd.DataFrame(totals)
    summary.to_csv(reports / "adjustment_return_only_summary.csv", index=False)
    adjustment_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    lines = [
        "# Adjustment causal comparison", "",
        "## Scope", "",
        "This report holds Data A securities, target weights and rebalance dates fixed. It changes only the return definition. It is therefore a direct return-only counterfactual, not a full-pipeline causal result.",
        "",
        f"Price-adjustment gate: **{adjustment_audit['status']}**. The research total-return column is labelled a candidate while the gate is blocked.",
        "", "## Frozen-holdings results", "",
        "| Return definition | Gross cumulative return | Net cumulative return |",
        "|---|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['return_definition']} | {row['gross_cumulative_return']:.4%} | "
            f"{row['net_cumulative_return']:.4%} |"
        )
    lines.extend([
        "", "## Interpretation boundary", "",
        "Indirect effects on features, labels, covariance, AUR, QUBO, solver selections and weights require three independently rerun full pipelines. Those runs are not permitted until the total-return dataset and benchmark pass their confirmatory gates. No claim that corporate actions caused the baseline loss is made from this table.",
    ])
    (reports / "ADJUSTMENT_CAUSAL_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    hypothesis_impact = pd.DataFrame([{
        "hypothesis": item,
        "status": "not_testable_due_to_data",
        "reason": "full-pipeline adjustment counterfactual blocked by adjustment/benchmark gate",
    } for item in ["H1", "H2", "H3", "H4", "H5", "H6"]])
    hypothesis_impact.to_csv(reports / "adjustment_impact_on_hypotheses.csv", index=False)
    return {
        "status": "diagnostic_complete_research_blocked",
        "baseline_experiment": str(baseline),
        "return_definitions": summary.to_dict("records"),
        "adjustment_gate": adjustment_audit["status"],
    }


def prepare_research_v2_runtime(paths: Paths) -> Path:
    """Stage a new isolated project only after every confirmatory data gate passes."""
    workspace = paths.root / "outputs" / "research_v2"
    audit_path = workspace / "reports" / "price_adjustment_audit.json"
    panel_path = workspace / "normalized" / "prices_total_return.parquet"
    if not audit_path.exists() or not panel_path.exists():
        raise RuntimeError("Price-adjustment v2 artifacts are missing.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not audit.get("research_eligible"):
        raise RuntimeError("Price-adjustment v2 is blocked; runtime staging is prohibited.")
    benchmark_path = paths.normalized / "benchmark.parquet"
    if not benchmark_path.exists():
        raise RuntimeError("Verified total-return benchmark is missing; runtime staging is prohibited.")
    benchmark = pd.read_parquet(benchmark_path)
    if (
        benchmark.empty
        or not benchmark.get("index_type", pd.Series(dtype=str)).astype(str).str.lower().eq("total_return").all()
        or not benchmark.get("methodology_url", pd.Series(dtype=str)).astype(str).str.startswith(("http://", "https://")).all()
    ):
        raise RuntimeError("Benchmark does not satisfy the total-return provenance contract.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime_root = workspace / "runtime" / stamp
    runtime_paths = Paths(runtime_root)
    runtime_paths.ensure()
    panel = pd.read_parquet(panel_path)
    standard = pd.DataFrame({
        "date": panel["date"], "ticker": panel["ticker"],
        "security_id": panel["security_id"],
        "open": panel["research_adjusted_open"],
        "high": panel["research_adjusted_high"],
        "low": panel["research_adjusted_low"],
        "close": panel["research_adjusted_close"],
        "adjusted_close": panel["research_adjusted_close"],
        "volume": panel["volume"], "trading_value": panel["trading_value"],
        "source": "research_v2_verified_total_return",
        "source_url": "local://outputs/research_v2/normalized/prices_total_return.parquet",
        "fetched_at": panel["fetched_at"], "available_at": panel["available_at"],
        "raw_checksum": audit["output_price_sha256"],
        "parser_version": "research-total-return-runtime-v2",
        "data_class": "real", "adjustment_policy": "verified_corporate_action_adjusted",
    })
    standard.to_parquet(runtime_paths.normalized / "prices.parquet", index=False)
    shutil.copy2(paths.normalized / "security_master.parquet", runtime_paths.normalized / "security_master.parquet")
    shutil.copy2(workspace / "normalized" / "corporate_actions.parquet", runtime_paths.normalized / "corporate_actions.parquet")
    shutil.copy2(benchmark_path, runtime_paths.normalized / "benchmark.parquet")
    for optional in ["index_membership", "financial_statements", "macro", "foreign_flow"]:
        source = paths.normalized / f"{optional}.parquet"
        if source.exists():
            table = pd.read_parquet(source)
            fixture = (
                ("data_class" in table and table["data_class"].astype(str).eq("fixture").any())
                or ("source" in table and table["source"].astype(str).str.contains("fixture", case=False).any())
            )
            if not fixture:
                shutil.copy2(source, runtime_paths.normalized / source.name)
    contract = {
        "adjustment_policy": "verified_corporate_action_adjusted",
        "source": "VSDC official notices with independent ex-date corroboration",
        "source_url": "https://www.vsd.vn/",
        "methodology": "docs/contracts/price_adjustment_contract.v2.json",
        "certified_by": "automated fail-closed research-v2 gate",
        "certified_at": datetime.now(timezone.utc).isoformat(),
        "output_price_dataset_sha256": sha256_file(runtime_paths.normalized / "prices.parquet"),
        "source_panel_sha256": audit["input_price_sha256"],
        "corporate_action_sha256": audit["corporate_action_sha256"],
        "contract_sha256": audit["contract_sha256"],
        "adjustment_version": "corporate-actions-v2",
    }
    (runtime_paths.normalized / "price_adjustment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (runtime_root / "RUNTIME_MANIFEST.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_workspace": str(workspace),
        "price_adjustment_audit": audit,
        "runtime_price_sha256": contract["output_price_dataset_sha256"],
        "benchmark_sha256": sha256_file(benchmark_path),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return runtime_root
