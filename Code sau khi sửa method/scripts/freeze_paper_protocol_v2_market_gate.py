from __future__ import annotations

"""Freeze paper protocol v2 with a common 30-session market regime gate."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from run_constraint_strategy_search import build_features, load_market_data


CORE_COLUMNS = ["date", "ticker", "adjusted_close", "volume", "trading_value"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--forward-prices", type=Path, required=True)
    parser.add_argument("--v1-targets", type=Path, required=True)
    parser.add_argument("--strategy-script", type=Path, required=True)
    parser.add_argument("--overlay-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    base, _, _ = load_market_data(args.base_dataset)
    forward = pd.read_parquet(args.forward_prices).copy()
    forward["date"] = pd.to_datetime(forward["date"], errors="coerce")
    for column in ("adjusted_close", "volume", "trading_value"):
        forward[column] = pd.to_numeric(forward[column], errors="coerce")
    prices = (
        pd.concat([base[CORE_COLUMNS], forward[CORE_COLUMNS]], ignore_index=True)
        .dropna(subset=CORE_COLUMNS)
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
    )
    features = build_features(prices)
    market = features.pivot(index="date", columns="ticker", values="return_1d").mean(axis=1).sort_index()
    lookback = 30
    growth = float((1.0 + market.tail(lookback).fillna(0.0)).prod() - 1.0)
    risk_multiplier = float(growth > 0.0)

    targets = pd.read_csv(args.v1_targets)
    targets["shadow_weight"] = targets["weight"]
    targets["risk_multiplier"] = risk_multiplier
    targets["paper_target_weight"] = targets["shadow_weight"] * risk_multiplier
    targets["regime_state"] = "RISK_ON" if risk_multiplier else "CASH"
    targets.to_csv(output / "september_2026_shadow_and_executable_targets.csv", index=False)

    executable = targets[targets["paper_target_weight"] > 0].copy()
    if executable.empty:
        executable = pd.DataFrame([
            {"method": method, "ticker": "CASH", "paper_target_weight": 1.0, "regime_state": "CASH"}
            for method in ("AUR", "QAUR")
        ])
    executable.to_csv(output / "september_2026_executable_paper_portfolio.csv", index=False)

    protocol = {
        "status": "FROZEN_PROSPECTIVE_PAPER_ONLY_V2",
        "supersedes": "paper_protocol_202609_v1 before its prospective start",
        "frozen_at_local_date": "2026-08-29",
        "first_test_interval_start": "2026-09-02",
        "base_candidate": "C1_IV_X",
        "common_risk_overlay": {
            "name": "trailing_full_universe_market_gate",
            "lookback_sessions": lookback,
            "risk_on_rule": "compound equal-weight full-universe return over prior 30 sessions > 0",
            "risk_off_allocation": "100% cash with zero assumed return",
            "switching_cost_bps": 25.0,
            "applied_identically_to_AUR_and_QAUR": True,
        },
        "current_market_growth_30_sessions": growth,
        "current_regime": "RISK_ON" if risk_multiplier else "CASH",
        "current_risk_multiplier": risk_multiplier,
        "parameters_retuned_during_future_paper_window": False,
        "hashes": {
            "base_dataset_sha256": sha256(args.base_dataset.resolve()),
            "forward_prices_sha256": sha256(args.forward_prices.resolve()),
            "v1_targets_sha256": sha256(args.v1_targets.resolve()),
            "strategy_script_sha256": sha256(args.strategy_script.resolve()),
            "overlay_script_sha256": sha256(args.overlay_script.resolve()),
        },
        "evidence_labels": {
            "historical_and_2026_overlay_results": "posthoc method-design evidence",
            "september_2026_onward": "prospective paper evidence if protocol remains unchanged",
        },
        "live_capital_authorized": False,
        "quantum_advantage_claimed": False,
    }
    (output / "FROZEN_PROTOCOL_V2.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ledger = pd.DataFrame(columns=[
        "date", "method", "regime_state", "market_growth_30d", "ticker",
        "shadow_weight", "paper_target_weight", "executed_weight", "gross_return",
        "switching_cost", "commission", "tax", "slippage", "net_return",
        "primary_source", "secondary_source", "audit_status", "notes",
    ])
    ledger.to_csv(output / "paper_trading_ledger_v2.csv", index=False)
    report = rf"""# Prospective paper protocol v2 — common 30-session market gate

The methodology remains AUR versus QAUR with an identical downstream portfolio
pipeline. A common causal risk overlay is added after weight allocation:

\[
e_t=\mathbb{{1}}\left[\prod_{{s=t-30}}^{{t-1}}(1+r^{{EW}}_s)-1>0\right],
\qquad \tilde{{w}}_t=e_t w_t.
\]

The 30-session full-universe growth at the lock date is **{growth:.4%}**.
Therefore the September paper regime is **{'RISK_ON' if risk_multiplier else 'CASH'}**.

## Shadow and executable targets

{targets[["method", "ticker", "shadow_weight", "risk_multiplier", "paper_target_weight", "regime_state"]].to_markdown(index=False)}

The shadow portfolio is retained to measure asset-selection quality even when the
common risk gate is off. Executable paper capital remains in cash until a future
monthly decision observes a positive trailing 30-session market return.
"""
    (output / "PAPER_PROTOCOL_V2_README.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
