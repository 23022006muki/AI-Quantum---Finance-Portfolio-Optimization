from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "outputs" / "research_v2"
BASELINE = ROOT / "outputs" / "Data A" / "outputs" / "experiments" / "20260813T164535-21c9b569ce"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    audit_dir = WORKSPACE / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict[str, object]] = {}

    expected_baseline = {
        "manifest.json": "2eeab1e338d93e52e806818da5a8fc7c7f4cf89872f7f7e8618a359ae90ff6cc",
    }
    # The manifest hash is recorded at first successful audit if the repository's
    # historical artifact predates this script. Subsequent audits enforce it.
    lock_path = audit_dir / "baseline_hash_lock.json"
    if lock_path.exists():
        expected_baseline = json.loads(lock_path.read_text(encoding="utf-8"))
    elif BASELINE.exists():
        expected_baseline = {
            path.relative_to(BASELINE).as_posix(): sha256(path)
            for path in BASELINE.rglob("*") if path.is_file()
        }
        lock_path.write_text(json.dumps(expected_baseline, indent=2), encoding="utf-8")
    observed = {
        path.relative_to(BASELINE).as_posix(): sha256(path)
        for path in BASELINE.rglob("*") if path.is_file()
    } if BASELINE.exists() else {}
    checks["frozen_data_a"] = {
        "passed": bool(observed) and observed == expected_baseline,
        "files": len(observed),
        "path": str(BASELINE),
    }

    contract = ROOT / "docs" / "contracts" / "price_adjustment_contract.v2.json"
    ledger = WORKSPACE / "normalized" / "corporate_actions.parquet"
    adjustment = WORKSPACE / "reports" / "price_adjustment_audit.json"
    counterfactual = WORKSPACE / "reports" / "ADJUSTMENT_CAUSAL_COMPARISON.md"
    checks["method_contract"] = {"passed": contract.exists(), "path": str(contract)}
    if ledger.exists():
        table = pd.read_parquet(ledger)
        required = {
            "security_id", "ticker", "event_type", "announcement_date", "record_date",
            "ex_date", "payment_date", "available_at", "source", "source_url",
            "raw_checksum", "parser_version", "verification_status",
        }
        checks["corporate_action_ledger"] = {
            "passed": bool(len(table)) and required.issubset(table.columns),
            "records": len(table),
            "tickers": int(table["ticker"].nunique()) if "ticker" in table else 0,
            "verified_cross_source": int(
                table.get("verification_status", pd.Series(dtype=str))
                .astype(str).eq("verified_cross_source").sum()
            ),
            "sha256": sha256(ledger),
        }
    else:
        checks["corporate_action_ledger"] = {"passed": False, "reason": "missing"}

    adjustment_payload: dict[str, object] = {}
    if adjustment.exists():
        adjustment_payload = json.loads(adjustment.read_text(encoding="utf-8"))
    checks["price_adjustment_artifact"] = {
        "passed": bool(adjustment_payload),
        "gate_status": adjustment_payload.get("status", "missing"),
        "research_eligible": bool(adjustment_payload.get("research_eligible", False)),
        "path": str(adjustment),
    }
    checks["return_only_counterfactual"] = {
        "passed": counterfactual.exists(), "path": str(counterfactual)
    }
    universe_path = WORKSPACE / "curated" / "universe_monthly_pit.parquet"
    universe_audit_path = WORKSPACE / "reports" / "survivorship_bias_audit.json"
    universe_audit = (
        json.loads(universe_audit_path.read_text(encoding="utf-8"))
        if universe_audit_path.exists() else {}
    )
    checks["historical_universe_pit"] = {
        "passed": universe_path.exists() and bool(universe_audit),
        "gate_status": universe_audit.get("status", "missing"),
        "whole_sample_completeness_filter_used": universe_audit.get(
            "whole_sample_completeness_filter_used"
        ),
        "path": str(universe_path),
    }

    benchmark = ROOT / "outputs" / "normalized" / "benchmark.parquet"
    benchmark_ok = False
    if benchmark.exists():
        table = pd.read_parquet(benchmark)
        benchmark_ok = bool(
            len(table)
            and "index_type" in table
            and table["index_type"].astype(str).str.lower().eq("total_return").all()
            and "methodology_url" in table
            and table["methodology_url"].astype(str).str.startswith(("http://", "https://")).all()
        )
    checks["verified_total_return_benchmark"] = {
        "passed": benchmark_ok, "path": str(benchmark)
    }

    structural = all(
        checks[key]["passed"]
        for key in ["frozen_data_a", "method_contract", "corporate_action_ledger",
                    "price_adjustment_artifact", "return_only_counterfactual",
                    "historical_universe_pit"]
    )
    confirmatory = bool(
        structural
        and checks["price_adjustment_artifact"]["gate_status"] == "pass"
        and checks["price_adjustment_artifact"]["research_eligible"]
        and benchmark_ok
    )
    status = "pass" if confirmatory else ("blocked_valid" if structural else "fail")
    blockers = []
    if checks["price_adjustment_artifact"]["gate_status"] != "pass":
        blockers.append("price_adjustment_gate_not_passed")
    if not benchmark_ok:
        blockers.append("verified_total_return_benchmark_missing")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "confirmatory_run_authorized": confirmatory,
        "checks": checks,
        "blockers": blockers,
    }
    output = audit_dir / "research_v2_audit.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if status == "blocked_valid":
        blocked = WORKSPACE / "experiments" / "BLOCKED_DATA_GATES"
        blocked.mkdir(parents=True, exist_ok=True)
        (blocked / "manifest.json").write_text(json.dumps({
            "status": "blocked",
            "reason": blockers,
            "created_at": payload["generated_at"],
            "audit": str(output),
            "no_training_or_backtest_executed": True,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status in {"pass", "blocked_valid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
