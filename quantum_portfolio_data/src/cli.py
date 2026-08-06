from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data_pipeline import (
    Paths, apply_price_adjustment_contract, build_universe, generate_fixture, import_csv, leakage_audit,
    quarantine_fixture_auxiliary, validate_data,
)
from .research import ResearchRunBlocked, build_features, load_config, run_experiment
from .sources import (
    crawl_ssi_stage1,
    crawl_vietstock_stage1,
    crawl_fdr_hose,
    crawl_vnstock_hose,
    import_point_in_time_table,
    merge_hose_checkpoints,
)


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
        choices=["fixture", "csv", "ssi", "vietstock", "vnstock", "fdr"],
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
    if "fixture" in str(manifest.get("data_class", "")).lower():
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
        else:
            requested = None if args.tickers.strip().lower() == "auto" else args.tickers.split(",")
            result = crawl_fdr_hose(
                paths, args.start, args.end,
                max_tickers=args.max_tickers, tickers=requested,
            )
        if args.source != "fixture":
            result["quarantined_fixture_auxiliary"] = quarantine_fixture_auxiliary(paths)
        print(json.dumps(result, indent=2))
    elif args.command == "import-pit-table":
        contracts = {
            "index_membership": {"ticker", "index_code", "effective_from", "effective_to", "available_at", "source", "source_url", "history_method"},
            "corporate_actions": {"ticker", "event_type", "announcement_date", "effective_date", "available_at", "source", "source_url"},
            "financial_statements": {"ticker", "fiscal_period_end", "publication_date", "available_at", "source", "source_url"},
            "macro": {"series_id", "observation_date", "release_date", "available_at", "value", "source", "source_url"},
            "foreign_flow": {"date", "ticker", "available_at", "foreign_net_value", "source", "source_url"},
            "benchmark": {"date", "benchmark", "total_return_index", "available_at", "source", "source_url"},
            "security_master": {
                "ticker", "exchange", "listing_date", "delisting_date", "effective_from",
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
        universe = build_universe(paths, args.rebalance, args.definition, args.index_code)
        print(f"universe_rows={len(universe)}")
    elif args.command == "leakage-audit":
        print(json.dumps(leakage_audit(paths), indent=2))
    elif args.command == "build-features":
        import pandas as pd
        prices = pd.read_parquet(paths.normalized / "prices.parquet")
        from .research import attach_point_in_time_features
        features = attach_point_in_time_features(build_features(prices), paths)
        paths.curated.mkdir(parents=True, exist_ok=True)
        features.to_parquet(paths.curated / "features.parquet", index=False)
        print(f"feature_rows={len(features)}")
    elif args.command == "run-experiment":
        out = run_experiment(ROOT, args.config.resolve())
        print_experiment_summary(out)
    elif args.command == "run-full":
        cfg = load_config(args.config.resolve())
        if cfg["data"]["source"] == "fixture":
            manifest = generate_fixture(
                paths, cfg["data"]["start"], cfg["data"]["end"],
                cfg["data"]["tickers"], cfg["seed"],
            )
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
        elif not (paths.normalized / "prices.parquet").exists():
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
        )
        print(f"universe_rows={len(universe):,}")
        print(json.dumps(leakage_audit(paths), indent=2, ensure_ascii=False))
        out = run_experiment(ROOT, args.config.resolve())
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
