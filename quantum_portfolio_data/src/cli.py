from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .data_pipeline import (
    Paths, apply_price_adjustment_contract, build_complete_case_workspace, build_universe,
    generate_fixture, import_csv, leakage_audit, quarantine_fixture_auxiliary, validate_data,
    sha256_file,
)
from .research import ResearchRunBlocked, build_features, load_config, run_experiment
from .sources import (
    audit_available_data_sources,
    crawl_historical_hose_price_gaps,
    crawl_ssi_stage1,
    crawl_hose_official_security_master,
    crawl_vietstock_stage1,
    crawl_fdr_hose,
    crawl_trading_economics_crosscheck,
    crawl_vnstock_hose,
    import_point_in_time_table,
    merge_historical_hose_checkpoints,
    merge_hose_checkpoints,
    crawl_world_bank_vietnam_snapshot,
    crawl_cafef_standalone_workspace,
)
from .corporate_actions import crawl_corporate_actions
from .price_adjustment import (
    build_price_adjustment_v2, build_return_only_adjustment_counterfactual,
    prepare_research_v2_runtime,
)
from .universe_pit import build_historical_universe_pit


ROOT = Path(__file__).resolve().parents[1]


def configure_utf8_console() -> None:
    """Make Vietnamese output reliable on Windows and redirected shells."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Point-in-time AI–Quantum portfolio pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    crawl = sub.add_parser("crawl", help="Collect/import raw data through an explicit source adapter")
    crawl.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    crawl.add_argument(
        "--source",
        choices=["fixture", "csv", "ssi", "vietstock", "vnstock", "fdr", "tradingeconomics"],
        required=True,
    )
    crawl.add_argument("--from", dest="start", default="2020-01-01")
    crawl.add_argument("--to", dest="end", default="2025-12-31")
    crawl.add_argument("--tickers", default="AAA,BBB,CCC,DDD,EEE,FFF,GGG,HHH")
    crawl.add_argument("--input", type=Path)
    crawl.add_argument("--source-name", default="user_authorized_csv")
    crawl.add_argument("--source-url", default="local://user-authorized")
    crawl.add_argument("--dry-run", action="store_true")
    crawl.add_argument(
        "--max-tickers", type=int, default=300,
        help="Maximum current HOSE equities for --source vnstock when --tickers=auto.",
    )
    crawl.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification only for a diagnosed local certificate-chain problem.",
    )
    pit = sub.add_parser("import-pit-table", help="Import a point-in-time Stage 1/2/3 table")
    pit.add_argument("--table", required=True, choices=[
        "index_membership", "corporate_actions", "financial_statements", "macro",
        "foreign_flow", "benchmark", "security_master"
    ])
    pit.add_argument("--input", type=Path, required=True)
    adjustment = sub.add_parser(
        "apply-adjustment-contract",
        help="Certify the exact normalized price panel using a hash-bound JSON contract",
    )
    adjustment.add_argument("--input", type=Path, required=True)
    sub.add_parser("normalize", help="Normalization is performed by the selected adapter")
    merge = sub.add_parser(
        "merge-market-sources",
        help="Merge FinanceDataReader primary and vnstock fallback checkpoints.",
    )
    merge.add_argument("--target-tickers", type=int, default=300)
    val = sub.add_parser("validate", help="Validate schemas and data quality")
    val.add_argument("--stage", type=int, default=1)
    uni = sub.add_parser("build-universe", help="Build point-in-time rebalance universe")
    uni.add_argument("--rebalance", default="monthly")
    uni.add_argument(
        "--definition", choices=["hose_all_listed", "index_membership"],
        default="hose_all_listed",
    )
    uni.add_argument("--index-code")
    uni.add_argument("--max-assets", type=int)
    uni.add_argument("--liquidity-lookback-days", type=int, default=60)
    uni.add_argument("--minimum-observations", type=int, default=40)
    sub.add_parser("report-coverage", help="Generate coverage report")
    sub.add_parser("leakage-audit", help="Audit point-in-time contracts")
    feat = sub.add_parser("build-features", help="Build leakage-aware features")
    feat.add_argument("--stage", type=int, default=1)
    for name in ("make-folds", "train-ranker", "build-instances", "run-solvers",
                 "optimize-weights", "backtest", "evaluate"):
        sp = sub.add_parser(name, help=f"Stage command; use run-experiment for orchestrated execution")
        sp.add_argument("--config", type=Path, default=ROOT / "configs" / "quick.yaml")
    run = sub.add_parser("run-experiment", help="Run end-to-end reproducible experiment")
    run.add_argument("--config", type=Path, default=ROOT / "configs" / "quick.yaml")
    full = sub.add_parser(
        "run-full",
        help="Validate/build/run the complete fixture or pre-imported research pipeline",
    )
    full.add_argument("--config", type=Path, default=ROOT / "configs" / "full_demo.yaml")
    complete_case = sub.add_parser(
        "run-complete-case",
        help=(
            "Build an isolated complete-case real-data workspace and run the "
            "explicitly exploratory pipeline"
        ),
    )
    complete_case.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "hose300_complete_case_exploratory.yaml",
    )
    complete_case.add_argument("--from", dest="start", default="2020-01-01")
    complete_case.add_argument("--to", dest="end", default="2025-12-31")
    complete_case.add_argument("--minimum-total-observations", type=int, default=40)
    complete_case.add_argument("--maximum-calendar-gap-days", type=int, default=5)
    data_b = sub.add_parser(
        "run-data-b",
        help=(
            "Reuse the immutable Data A cleaned panel, create a separate Data B workspace, "
            "and run the optimized leakage-aware exploratory pipeline"
        ),
    )
    data_b.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "data_b.yaml",
    )
    data_b.add_argument(
        "--base-workspace", type=Path, default=ROOT / "outputs" / "Data A",
    )
    data_b.add_argument(
        "--output-workspace", type=Path, default=ROOT / "outputs" / "Data B",
    )
    cafef = sub.add_parser(
        "run-cafef",
        help="Crawl a separate CafeF-only panel, quality-gate it, and run only if accepted",
    )
    cafef.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "cafef_standalone_exploratory.yaml",
    )
    cafef.add_argument("--from", dest="start", default="2020-01-01")
    cafef.add_argument("--to", dest="end", default="2025-12-31")
    cafef.add_argument(
        "--tickers",
        default="VCB,BID,CTG,MBB,HPG,FPT,VNM,VIC,GAS,MSN,MWG,SSI",
    )
    cafef.add_argument("--max-workers", type=int, default=3)
    cafef.add_argument(
        "--workspace-name",
        help="Stable output folder name under outputs, for example 'data CafeF'",
    )
    cafef.add_argument("--minimum-total-observations", type=int, default=40)
    cafef.add_argument("--maximum-calendar-gap-days", type=int, default=5)
    cafef.add_argument(
        "--existing-workspace", type=Path,
        help="Resume quality-gating an already collected CafeF workspace without recrawling",
    )
    prepare = sub.add_parser(
        "prepare-research-data",
        help="Inspect data contracts and build the PIT universe without fabricating missing tables",
    )
    prepare.add_argument("--config", type=Path, default=ROOT / "configs" / "hose300_real.yaml")
    hose_master = sub.add_parser(
        "crawl-hose-security-master",
        help="Collect official HOSE current listings and historical delisting events",
    )
    hose_master.add_argument("--from-year", type=int, default=2015)
    hose_master.add_argument("--to-year", type=int, default=2025)
    hose_master.add_argument("--pause-seconds", type=float, default=0.05)
    historical_prices = sub.add_parser(
        "crawl-historical-price-gaps",
        help="Checkpoint missing historical HOSE symbols using public adapters without promotion",
    )
    historical_prices.add_argument("--from", dest="start", default="2020-01-01")
    historical_prices.add_argument("--to", dest="end", default="2025-12-31")
    historical_prices.add_argument("--no-vnstock-fallback", action="store_true")
    historical_prices.add_argument("--pause-seconds", type=float, default=0.35)
    historical_merge = sub.add_parser(
        "merge-historical-price-checkpoints",
        help="Promote all available historical HOSE checkpoints using official security identities",
    )
    historical_merge.add_argument("--from", dest="start", default="2020-01-01")
    historical_merge.add_argument("--to", dest="end", default="2025-12-31")
    world_bank = sub.add_parser(
        "crawl-world-bank",
        help="Collect a keyless official World Bank Vietnam macro snapshot (non-PIT)",
    )
    world_bank.add_argument("--from-year", type=int, default=2015)
    world_bank.add_argument("--to-year", type=int, default=2025)
    actions = sub.add_parser(
        "crawl-corporate-actions",
        help="Collect VSDC official notices and CafeF ex-date corroboration into research_v2",
    )
    actions.add_argument("--from", dest="start", default="2020-01-01")
    actions.add_argument("--to", dest="end", default="2025-12-31")
    actions.add_argument("--tickers", default="auto")
    actions.add_argument("--max-workers", type=int, default=3)
    actions.add_argument("--pause-seconds", type=float, default=0.20)
    sub.add_parser(
        "build-price-adjustment-v2",
        help="Build raw/source-adjusted/research-total-return candidate and its fail-closed audit",
    )
    universe_v2 = sub.add_parser(
        "build-universe-pit-v2",
        help="Build the isolated monthly historical HOSE universe from prior-only data",
    )
    universe_v2.add_argument("--from", dest="start", default="2020-01-01")
    universe_v2.add_argument("--to", dest="end", default="2025-12-31")
    universe_v2.add_argument("--lookback-days", type=int, default=90)
    universe_v2.add_argument("--minimum-observations", type=int, default=40)
    sub.add_parser(
        "adjustment-counterfactual",
        help="Reprice frozen Data A holdings using raw, source-adjusted and research returns",
    )
    research_v2 = sub.add_parser(
        "run-research-v2",
        help=(
            "Stage and run the isolated confirmatory pipeline only after the "
            "adjustment and total-return benchmark gates pass"
        ),
    )
    research_v2.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "hose_research_v2.yaml"
    )
    sub.add_parser("audit-data-sources", help="Write the current source and research-data gap inventory")
    return p


def _fmt(value, percent=False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "NA"
    return f"{number * 100:.2f}%" if percent else f"{number:.6f}"


def print_experiment_summary(out: Path) -> None:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((out / "data_quality.json").read_text(encoding="utf-8"))
    leakage = json.loads((out / "leakage_audit.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(out / "metrics_long.csv")
    comparisons = pd.read_csv(out / "comparisons.csv")
    statistics = pd.read_csv(out / "statistical_tests.csv")
    ablations = pd.read_csv(out / "ablation_results.csv")
    sensitivity = pd.read_csv(out / "sensitivity_results.csv")
    rankings = pd.read_csv(out / "rankings.csv")
    trades = pd.read_csv(out / "trades.csv")
    weights = pd.read_csv(out / "weights.csv")
    latest_path = out / "latest_selected_portfolio.csv"
    latest = pd.read_csv(latest_path) if latest_path.exists() else pd.DataFrame()
    print("\n" + "=" * 100)
    print("BÁO CÁO CHẠY TOÀN BỘ HỆ THỐNG AI–QUANTUM PORTFOLIO")
    print("=" * 100)
    print(f"Experiment ID       : {manifest['experiment_id']}")
    print(f"Trạng thái          : {manifest['status']}")
    print(f"Nhãn dữ liệu         : {manifest['label']}")
    print(f"Config hash          : {manifest['config_hash']}")
    print(f"Dataset hash         : {manifest['dataset_hash']}")
    print(f"Walk-forward folds   : {manifest['folds_completed']}/{manifest['folds_requested']}")
    print(f"Thư mục kết quả      : {out}")
    print("\n[1] KIỂM TRA DỮ LIỆU VÀ POINT-IN-TIME")
    print("-" * 100)
    print(f"Data class           : {', '.join(quality['data_class'])}")
    print(f"Số bản ghi           : {quality['records']:,}")
    print(f"Số mã                : {quality['tickers']}")
    print(f"Khoảng thời gian     : {quality['start']} → {quality['end']}")
    print(f"Data quality         : {quality['status']}")
    print(f"Leakage audit        : {leakage['status']}")
    for check, passed in leakage["checks"].items():
        print(f"  - {check:<50}: {'PASS' if passed else 'FAIL'}")
    print("\n[2] CHẤT LƯỢNG XẾP HẠNG XGBOOST")
    print("-" * 100)
    fold_ic = rankings.groupby("fold")["fold_rank_ic"].first()
    print(f"Số ranking rows      : {len(rankings):,}")
    print(f"Mean rank IC         : {fold_ic.mean():.6f}")
    print(f"Median rank IC       : {fold_ic.median():.6f}")
    print(f"Min/Max rank IC      : {fold_ic.min():.6f} / {fold_ic.max():.6f}")
    print("\n[3] SO SÁNH SOLVER")
    print("-" * 100)
    solver_view = comparisons[[
        "method", "energy_mean", "feasibility_rate", "optimality_gap_mean",
        "runtime_seconds", "runs",
    ]].copy()
    solver_view["feasibility_rate"] = solver_view["feasibility_rate"].map(
        lambda x: _fmt(x, percent=True)
    )
    print(solver_view.to_string(index=False))
    print("\n[4] KẾT QUẢ DANH MỤC VÀ BACKTEST")
    print("-" * 100)
    metric_view = metrics[[
        "strategy", "cumulative_return", "annualized_return", "annualized_volatility",
        "sharpe", "sortino", "max_drawdown", "observations",
    ]].copy()
    for column in ["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown"]:
        metric_view[column] = metric_view[column].map(lambda x: _fmt(x, percent=True))
    for column in ["sharpe", "sortino"]:
        metric_view[column] = metric_view[column].map(_fmt)
    print(metric_view.to_string(index=False))
    print(f"\nSố trade rows        : {len(trades):,}")
    print(f"Số weight rows       : {len(weights):,}")
    print("\nRổ cổ phiếu của fold cuối cùng:")
    if latest.empty:
        print("  Không có fold nghiên cứu hoàn tất.")
    else:
        columns = [column for column in [
            "ticker", "company_name", "sector", "target_weight", "signal",
            "trade_weight", "estimated_cost", "adv_participation",
        ] if column in latest.columns]
        print(latest[columns].to_string(index=False))
    print("\n[5] ABLATION STUDY")
    print("-" * 100)
    ablation_view = ablations.groupby(["configuration", "selector", "solver"]).agg(
        folds=("fold", "nunique"),
        objective_mean=("objective", "mean"),
        feasibility_rate=("feasibility_rate", "mean"),
        optimality_gap=("optimality_gap", "mean"),
    ).reset_index()
    ablation_view["feasibility_rate"] = ablation_view["feasibility_rate"].map(
        lambda x: _fmt(x, percent=True)
    )
    print(ablation_view.to_string(index=False))
    print("\n[6] SENSITIVITY / ROBUSTNESS")
    print("-" * 100)
    print(f"Số sensitivity cases: {len(sensitivity):,}")
    sensitivity_view = sensitivity.groupby(
        ["depth_p", "shots", "cardinality", "uniform_probability_noise_proxy"]
    ).agg(
        feasibility_rate=("feasibility_rate", "mean"),
        optimality_gap=("optimality_gap", "mean"),
        runtime_seconds=("runtime_seconds", "mean"),
    ).reset_index()
    sensitivity_view["feasibility_rate"] = sensitivity_view["feasibility_rate"].map(
        lambda x: _fmt(x, percent=True)
    )
    print(sensitivity_view.to_string(index=False))
    print("\n[7] KIỂM ĐỊNH THỐNG KÊ — BLOCK BOOTSTRAP + HOLM")
    print("-" * 100)
    stats_view = statistics[[
        "test", "mean_difference", "ci_low", "ci_high",
        "p_value", "p_value_holm", "conclusion",
    ]]
    print(stats_view.to_string(index=False))
    print("\n[8] CÁC FILE OUTPUT")
    print("-" * 100)
    for name in manifest["artifacts"]:
        print(f"  - {out / name}")
    print("\nKẾT LUẬN THỰC THI")
    print("-" * 100)
    if manifest.get("mode") == "exploratory":
        print("Toàn bộ pipeline đã chạy thành công trên panel giá thật dạng complete-case.")
        print("Kết quả chỉ mang tính khám phá, không phải kiểm định confirmatory toàn HOSE,")
        print("khuyến nghị đầu tư hoặc bằng chứng quantum advantage.")
    elif "fixture" in str(manifest.get("data_class", "")).lower():
        print("Toàn bộ code path đã chạy thành công. Kết quả fixture chỉ dùng kiểm thử phần mềm;")
        print("không được diễn giải là kết quả nghiên cứu HOSE hoặc quantum advantage.")
    else:
        print("Toàn bộ pipeline nghiên cứu đã chạy thành công trên panel giá thị trường thực.")
        print("Kết quả phụ thuộc cấu hình và mẫu nghiên cứu; không phải khuyến nghị đầu tư hay bằng chứng quantum advantage.")
    print("=" * 100 + "\n")


def main(argv=None) -> int:
    configure_utf8_console()
    args = parser().parse_args(argv)
    paths = Paths(ROOT)
    if args.command == "crawl":
        if args.dry_run:
            print(json.dumps(vars(args), default=str, indent=2))
            return 0
        if args.stage != 1 and args.source != "csv":
            raise SystemExit("Stage 2/3 use import-pit-table or an explicitly configured adapter.")
        if args.source == "fixture":
            result = generate_fixture(paths, args.start, args.end, args.tickers.split(","))
        elif args.source == "csv":
            if not args.input:
                raise SystemExit("--input is required for --source csv")
            result = import_csv(paths, args.input, args.source_name, args.source_url)
        elif args.source == "ssi":
            result = crawl_ssi_stage1(paths, args.tickers.split(","), args.start, args.end)
        elif args.source == "vietstock":
            result = crawl_vietstock_stage1(
                paths,
                args.tickers.split(","),
                args.start,
                args.end,
                verify_tls=not args.insecure,
            )
        elif args.source == "vnstock":
            requested = None if args.tickers.strip().lower() == "auto" else args.tickers.split(",")
            result = crawl_vnstock_hose(
                paths, args.start, args.end,
                max_tickers=args.max_tickers, tickers=requested,
            )
        elif args.source == "fdr":
            requested = None if args.tickers.strip().lower() == "auto" else args.tickers.split(",")
            result = crawl_fdr_hose(
                paths, args.start, args.end,
                max_tickers=args.max_tickers, tickers=requested,
            )
        else:
            if args.tickers.strip().lower() == "auto":
                raise SystemExit(
                    "Trading Economics cross-check requires explicit tickers; its current-list API "
                    "must not define the historical HOSE universe."
                )
            result = crawl_trading_economics_crosscheck(
                paths, args.tickers.split(","), args.start, args.end
            )
        if args.source != "fixture":
            result["quarantined_fixture_auxiliary"] = quarantine_fixture_auxiliary(paths)
        print(json.dumps(result, indent=2))
    elif args.command == "import-pit-table":
        contracts = {
            "index_membership": {"ticker", "index_code", "effective_from", "effective_to", "available_at", "source", "source_url", "history_method"},
            "corporate_actions": {"ticker", "security_id", "event_type", "announcement_date", "effective_date", "available_at", "source", "source_url"},
            "financial_statements": {"ticker", "fiscal_period_end", "publication_date", "available_at", "source", "source_url"},
            "macro": {"series_id", "observation_date", "release_date", "available_at", "value", "source", "source_url"},
            "foreign_flow": {"date", "ticker", "available_at", "foreign_net_value", "source", "source_url"},
            "benchmark": {"date", "benchmark", "total_return_index", "index_type", "methodology_url", "available_at", "source", "source_url"},
            "security_master": {
                "security_id", "ticker", "exchange", "listing_date", "delisting_date", "effective_from",
                "effective_to", "available_at", "history_method", "source", "source_url",
            },
        }
        output = paths.normalized / f"{args.table}.parquet"
        result = import_point_in_time_table(args.input, output, contracts[args.table], args.table)
        print(json.dumps(result, indent=2))
    elif args.command == "apply-adjustment-contract":
        print(json.dumps(
            apply_price_adjustment_contract(paths, args.input),
            indent=2, ensure_ascii=False,
        ))
    elif args.command == "normalize":
        print("Normalization is idempotently performed during crawl/import.")
    elif args.command == "merge-market-sources":
        result = merge_hose_checkpoints(paths, args.target_tickers)
        result["quarantined_fixture_auxiliary"] = quarantine_fixture_auxiliary(paths)
        print(json.dumps(
            result,
            indent=2, ensure_ascii=False,
        ))
    elif args.command in {"validate", "report-coverage"}:
        report, coverage = validate_data(paths)
        print(json.dumps(report, indent=2))
        if args.command == "report-coverage":
            print(coverage.to_string(index=False))
    elif args.command == "build-universe":
        universe = build_universe(
            paths, args.rebalance, args.definition, args.index_code,
            args.max_assets, args.liquidity_lookback_days, args.minimum_observations,
        )
        print(f"universe_rows={len(universe)}")
    elif args.command == "leakage-audit":
        print(json.dumps(leakage_audit(paths), indent=2))
    elif args.command == "build-features":
        prices = pd.read_parquet(paths.normalized / "prices.parquet")
        from .research import attach_point_in_time_features
        features = attach_point_in_time_features(build_features(prices), paths)
        paths.curated.mkdir(parents=True, exist_ok=True)
        features.to_parquet(paths.curated / "features.parquet", index=False)
        print(f"feature_rows={len(features)}")
    elif args.command == "prepare-research-data":
        cfg = load_config(args.config.resolve())
        universe_cfg = cfg.get("universe", {})
        credential_status = {
            name: bool(os.getenv(name)) for name in [
                "SSI_CONSUMER_ID", "SSI_CONSUMER_SECRET", "VIETSTOCK_COOKIE_FILE",
                "VIETSTOCK_AUTH_HEADER_FILE", "TRADING_ECONOMICS_API_KEY", "FRED_API_KEY",
            ]
        }
        quality, _ = validate_data(paths)
        build_error = None
        try:
            universe = build_universe(
                paths, cfg["data"].get("rebalance", "monthly"),
                universe_cfg.get("definition", "hose_all_listed"),
                universe_cfg.get("index_code"), universe_cfg.get("max_assets"),
                universe_cfg.get("liquidity_lookback_days", 60),
                universe_cfg.get("minimum_observations", 40),
            )
            universe_rows = len(universe)
        except (ValueError, KeyError) as exc:
            build_error = str(exc)
            universe_rows = 0
        result = {
            "config": str(args.config.resolve()), "credentials_configured": credential_status,
            "data_quality": quality, "universe_rows": universe_rows,
            "universe_build_error": build_error, "leakage_audit": leakage_audit(paths),
            "note": "Credential booleans only; no secret values are printed or persisted.",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "crawl-hose-security-master":
        result = crawl_hose_official_security_master(
            paths, args.from_year, args.to_year, args.pause_seconds
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "crawl-historical-price-gaps":
        result = crawl_historical_hose_price_gaps(
            paths, args.start, args.end,
            try_vnstock_fallback=not args.no_vnstock_fallback,
            pause_seconds=args.pause_seconds,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "merge-historical-price-checkpoints":
        result = merge_historical_hose_checkpoints(paths, args.start, args.end)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "crawl-world-bank":
        result = crawl_world_bank_vietnam_snapshot(paths, args.from_year, args.to_year)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "crawl-corporate-actions":
        requested = None if args.tickers.strip().lower() == "auto" else args.tickers.split(",")
        result = crawl_corporate_actions(
            paths, args.start, args.end, requested, args.max_workers, args.pause_seconds,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "build-price-adjustment-v2":
        result = build_price_adjustment_v2(paths)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["status"] == "blocked":
            raise SystemExit(2)
    elif args.command == "build-universe-pit-v2":
        result = build_historical_universe_pit(
            paths, args.start, args.end, args.lookback_days, args.minimum_observations,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "adjustment-counterfactual":
        result = build_return_only_adjustment_counterfactual(paths)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "run-research-v2":
        try:
            runtime_root = prepare_research_v2_runtime(paths)
        except RuntimeError as exc:
            raise SystemExit(f"RESEARCH_V2_BLOCKED: {exc}") from exc
        temporary_out = run_experiment(runtime_root, args.config.resolve())
        experiments = ROOT / "outputs" / "research_v2" / "experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        out = experiments / temporary_out.name
        if out.exists():
            raise RuntimeError(f"Research V2 artifact already exists: {out}")
        shutil.copytree(temporary_out, out)
        print_experiment_summary(out)
    elif args.command == "audit-data-sources":
        print(json.dumps(audit_available_data_sources(paths), indent=2, ensure_ascii=False))
    elif args.command == "run-experiment":
        out = run_experiment(ROOT, args.config.resolve())
        print_experiment_summary(out)
    elif args.command == "run-full":
        cfg = load_config(args.config.resolve())
        if cfg["data"]["source"] == "fixture":
            # A demo must never replace, certify, or otherwise mutate the real-data
            # workspace. Build and execute it in an isolated temporary project, then
            # copy only the immutable experiment artifact back to outputs/experiments.
            with tempfile.TemporaryDirectory(prefix="ai-quantum-fixture-") as temporary:
                demo_root = Path(temporary)
                demo_paths = Paths(demo_root)
                manifest = generate_fixture(
                    demo_paths, cfg["data"]["start"], cfg["data"]["end"],
                    cfg["data"]["tickers"], cfg["seed"],
                )
                print(json.dumps(manifest, indent=2, ensure_ascii=False))
                quality, _ = validate_data(demo_paths)
                print(json.dumps(quality, indent=2, ensure_ascii=False))
                universe_cfg = cfg.get("universe", {})
                universe = build_universe(
                    demo_paths, cfg["data"].get("rebalance", "monthly"),
                    universe_cfg.get("definition", "hose_all_listed"),
                    universe_cfg.get("index_code"), universe_cfg.get("max_assets"),
                    universe_cfg.get("liquidity_lookback_days", 60),
                    universe_cfg.get("minimum_observations", 40),
                )
                print(f"universe_rows={len(universe):,}")
                print(json.dumps(leakage_audit(demo_paths), indent=2, ensure_ascii=False))
                temporary_out = run_experiment(demo_root, args.config.resolve())
                experiments = ROOT / "outputs" / "experiments"
                experiments.mkdir(parents=True, exist_ok=True)
                out = experiments / temporary_out.name
                if out.exists():
                    raise RuntimeError(f"Experiment artifact already exists: {out}")
                shutil.copytree(temporary_out, out)
            print_experiment_summary(out)
            return 0
        if not (paths.normalized / "prices.parquet").exists():
            raise SystemExit(
                "Research run-full requires an existing normalized price panel. "
                "Run the authorized crawl/import command first."
            )
        quality, coverage = validate_data(paths)
        print(json.dumps(quality, indent=2, ensure_ascii=False))
        universe_cfg = cfg.get("universe", {})
        universe = build_universe(
            paths, cfg["data"].get("rebalance", "monthly"),
            universe_cfg.get("definition", "hose_all_listed"),
            universe_cfg.get("index_code"),
            universe_cfg.get("max_assets"),
            universe_cfg.get("liquidity_lookback_days", 60),
            universe_cfg.get("minimum_observations", 40),
        )
        print(f"universe_rows={len(universe):,}")
        print(json.dumps(leakage_audit(paths), indent=2, ensure_ascii=False))
        out = run_experiment(ROOT, args.config.resolve())
        print_experiment_summary(out)
    elif args.command == "run-complete-case":
        workspace, dataset_manifest = build_complete_case_workspace(
            paths, args.start, args.end, args.minimum_total_observations,
            args.maximum_calendar_gap_days,
        )
        print(json.dumps(dataset_manifest, indent=2, ensure_ascii=False))
        complete_paths = Paths(workspace)
        quality, _ = validate_data(complete_paths)
        print(json.dumps(quality, indent=2, ensure_ascii=False))
        temporary_out = run_experiment(workspace, args.config.resolve())
        (temporary_out / "complete_case_dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_manifest_path = temporary_out / "manifest.json"
        temporary_manifest = json.loads(temporary_manifest_path.read_text(encoding="utf-8"))
        temporary_manifest["artifacts"] = sorted(
            path.relative_to(temporary_out).as_posix()
            for path in temporary_out.rglob("*") if path.is_file()
        )
        temporary_manifest["artifact_sha256"] = {
            path.relative_to(temporary_out).as_posix(): sha256_file(path)
            for path in temporary_out.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        temporary_manifest_path.write_text(
            json.dumps(temporary_manifest, indent=2), encoding="utf-8"
        )
        experiments = ROOT / "outputs" / "experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        out = experiments / temporary_out.name
        if out.exists():
            raise RuntimeError(f"Experiment artifact already exists: {out}")
        shutil.copytree(temporary_out, out)
        print(f"Complete-case workspace: {workspace}")
        print_experiment_summary(out)
    elif args.command == "run-data-b":
        base_workspace = args.base_workspace.resolve()
        output_workspace = args.output_workspace.resolve()
        base_outputs = base_workspace / "outputs"
        required = [
            base_outputs / "normalized" / "prices.parquet",
            base_outputs / "normalized" / "security_master.parquet",
            base_outputs / "raw" / "manifest.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit("Data B requires the existing Data A package: " + ", ".join(missing))
        if not output_workspace.exists():
            (output_workspace / "outputs").mkdir(parents=True, exist_ok=False)
            for folder in ["raw", "normalized", "reports"]:
                source = base_outputs / folder
                if source.exists():
                    shutil.copytree(source, output_workspace / "outputs" / folder)
        else:
            output_prices = output_workspace / "outputs" / "normalized" / "prices.parquet"
            if not output_prices.exists():
                raise RuntimeError(
                    f"Existing Data B folder is incomplete and was not overwritten: {output_workspace}"
                )
            if sha256_file(output_prices) != sha256_file(required[0]):
                raise RuntimeError(
                    "Existing Data B price panel no longer matches Data A; refusing implicit overwrite."
                )
        data_b_paths = Paths(output_workspace)
        quality, _ = validate_data(data_b_paths)
        print(json.dumps(quality, indent=2, ensure_ascii=False))
        out = run_experiment(output_workspace, args.config.resolve())
        base_manifest = base_workspace / "DATA_A_PACKAGE.json"
        package = {
            "package": "Data B",
            "status": "completed_exploratory" if (out / "strategy_metrics_summary.csv").exists()
            else "incomplete",
            "base_workspace": str(base_workspace),
            "base_data_a_manifest": str(base_manifest) if base_manifest.exists() else None,
            "base_price_sha256": sha256_file(required[0]),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config.resolve()),
            "experiment_id": out.name,
            "experiment_path": str(out),
            "method": (
                "Data A reuse + purged-validation XGBoost/technical blend + adaptive "
                "universe reduction + best-observed XY-QAOA + constrained weights + "
                "market-regime exposure"
            ),
            "interpretation": "exploratory_only_not_confirmatory_research",
        }
        (output_workspace / "DATA_B_PACKAGE.json").write_text(
            json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Data B workspace: {output_workspace}")
        print_experiment_summary(out)
    elif args.command == "run-cafef":
        if args.existing_workspace:
            collection_root = args.existing_workspace.resolve()
            collection_manifest_path = collection_root / "outputs" / "raw" / "manifest.json"
            if not collection_manifest_path.exists():
                raise SystemExit(
                    f"Existing CafeF workspace has no collection manifest: "
                    f"{collection_manifest_path}"
                )
            collection_manifest = json.loads(
                collection_manifest_path.read_text(encoding="utf-8")
            )
        else:
            requested_tickers = (
                None if args.tickers.strip().lower() == "auto" else args.tickers.split(",")
            )
            collection_root, collection_manifest = crawl_cafef_standalone_workspace(
                paths, args.start, args.end, requested_tickers, args.max_workers,
                args.workspace_name,
            )
        print(json.dumps(collection_manifest, indent=2, ensure_ascii=False))
        if collection_manifest.get("status") == "rejected":
            raise SystemExit(
                f"CafeF dataset rejected before quality gate. Audit: "
                f"{collection_root / 'outputs' / 'raw' / 'manifest.json'}"
            )
        complete_root, complete_manifest = build_complete_case_workspace(
            Paths(collection_root), args.start, args.end,
            args.minimum_total_observations, args.maximum_calendar_gap_days,
        )
        cfg = load_config(args.config.resolve())
        retained = int(complete_manifest["tickers_retained"])
        required = int(cfg.get("reduction", {}).get("candidate_size", 8))
        complete_paths = Paths(complete_root)
        quality, _ = validate_data(complete_paths)
        initial_quality = quality
        quality_excluded_tickers: list[str] = []
        if quality["status"] != "pass":
            review_path = complete_paths.reports / "return_outlier_review.csv"
            if review_path.exists():
                review = pd.read_csv(review_path)
                quality_excluded_tickers = sorted(
                    review.loc[
                        review.get("resolution", pd.Series(dtype=str)).astype(str).eq("unresolved"),
                        "ticker",
                    ].dropna().astype(str).unique().tolist()
                )
            non_outlier_errors = [
                issue for issue in quality.get("issues", [])
                if issue.get("severity") == "error"
                and issue.get("check") != "unresolved_return_outlier"
            ]
            if (
                quality_excluded_tickers
                and not non_outlier_errors
                and retained - len(quality_excluded_tickers) >= required
            ):
                complete_root, complete_manifest = build_complete_case_workspace(
                    Paths(collection_root), args.start, args.end,
                    args.minimum_total_observations, args.maximum_calendar_gap_days,
                    forced_excluded_tickers=quality_excluded_tickers,
                )
                complete_paths = Paths(complete_root)
                quality, _ = validate_data(complete_paths)
                retained = int(complete_manifest["tickers_retained"])
        acceptance = {
            "status": "accepted" if quality["status"] == "pass" and retained >= required else "rejected",
            "quality": quality,
            "initial_quality": initial_quality,
            "quality_excluded_tickers": quality_excluded_tickers,
            "quality_exclusion_policy": (
                "drop_entire_ticker_with_unresolved_return_outlier; never alter source price"
            ),
            "retained_tickers": retained,
            "minimum_required_tickers": required,
            "collection_workspace": str(collection_root),
            "complete_case_workspace": str(complete_root),
            "created_at": pd.Timestamp.utcnow().isoformat(),
        }
        (complete_paths.reports / "cafef_acceptance_gate.json").write_text(
            json.dumps(acceptance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(acceptance, indent=2, ensure_ascii=False))
        if acceptance["status"] != "accepted":
            raise SystemExit(
                "CafeF panel did not meet the declared system quality gate; no training or "
                f"backtest was run. Audit: {complete_paths.reports / 'cafef_acceptance_gate.json'}"
            )
        temporary_out = run_experiment(complete_root, args.config.resolve())
        for name, payload in (
            ("cafef_collection_manifest.json", collection_manifest),
            ("complete_case_dataset_manifest.json", complete_manifest),
            ("cafef_acceptance_gate.json", acceptance),
        ):
            (temporary_out / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        temporary_manifest_path = temporary_out / "manifest.json"
        temporary_manifest = json.loads(temporary_manifest_path.read_text(encoding="utf-8"))
        temporary_manifest["artifacts"] = sorted(
            path.relative_to(temporary_out).as_posix()
            for path in temporary_out.rglob("*") if path.is_file()
        )
        temporary_manifest["artifact_sha256"] = {
            path.relative_to(temporary_out).as_posix(): sha256_file(path)
            for path in temporary_out.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        temporary_manifest_path.write_text(
            json.dumps(temporary_manifest, indent=2), encoding="utf-8"
        )
        experiments = ROOT / "outputs" / "experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        out = experiments / temporary_out.name
        if out.exists():
            raise RuntimeError(f"Experiment artifact already exists: {out}")
        shutil.copytree(temporary_out, out)
        print(f"CafeF collection workspace: {collection_root}")
        print(f"CafeF accepted workspace: {complete_root}")
        print_experiment_summary(out)
    else:
        out = run_experiment(ROOT, args.config.resolve())
        artifact_map = {
            "make-folds": "fold_manifest.csv",
            "train-ranker": "rankings.csv",
            "build-instances": "optimization_instances.json",
            "run-solvers": "solver_runs.csv",
            "optimize-weights": "weights.csv",
            "backtest": "portfolio_returns.csv",
            "evaluate": "RESEARCH_REPORT.md",
        }
        print(out.relative_to(ROOT) / artifact_map[args.command])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResearchRunBlocked as exc:
        print(f"RESEARCH RUN BLOCKED: {exc}", file=sys.stderr)
        print(f"Audit artifact: {exc.output_dir}", file=sys.stderr)
        raise SystemExit(2)
