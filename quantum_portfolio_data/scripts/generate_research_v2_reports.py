from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "research_v2"
REPORTS = V2 / "reports"
BASELINE = ROOT / "outputs" / "Data A" / "outputs" / "experiments" / "20260813T164535-21c9b569ce"


def _pct(value: float) -> str:
    return f"{float(value):.2%}"


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit = json.loads((V2 / "audits" / "research_v2_audit.json").read_text(encoding="utf-8"))
    adjustment = json.loads((REPORTS / "price_adjustment_audit.json").read_text(encoding="utf-8"))
    source_audit_path = ROOT / "outputs" / "reports" / "data_source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    ca_audit = json.loads((REPORTS / "corporate_action_source_audit.json").read_text(encoding="utf-8"))
    survivorship = json.loads((REPORTS / "survivorship_bias_audit.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(REPORTS / "adjustment_return_only_summary.csv").set_index("return_definition")
    manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(BASELINE / "strategy_metrics_summary.csv").set_index("strategy")
    baseline_metric = metrics.loc["full_pipeline_xy_qaoa"]
    solver = pd.read_csv(BASELINE / "solver_runs.csv")
    xy = solver[solver["method"].eq("xy_qaoa_dicke_ideal_statevector")]
    hypotheses = pd.read_csv(BASELINE / "hypothesis_results.csv")
    latest = pd.read_csv(BASELINE / "latest_selected_portfolio.csv")
    rankings = pd.read_csv(BASELINE / "rankings.csv")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    created_at = datetime.now(timezone.utc).isoformat()

    header = [
        "| Trường | Giá trị |", "|---|---|",
        "| Experiment ID | `BLOCKED_DATA_GATES` |",
        f"| Dataset hash (candidate) | `{adjustment['output_price_sha256']}` |",
        f"| Adjustment version | `{adjustment['adjustment_version']}` |",
        "| Config hash | `not_created_no_confirmatory_run` |",
        f"| Git commit tại thời điểm báo cáo | `{git_commit}` |",
        f"| Created at | `{created_at}` |",
        "| Mode | `research_v2_fail_closed` |",
        "| Label | `BLOCKED — DIAGNOSTIC ARTIFACT, NOT CONFIRMATORY RESEARCH` |",
        "| Folds | `0 (không huấn luyện/backtest khi gate chưa đạt)` |",
        "| OOS start/end | `not_created` |",
        f"| Audit status | `{audit['status']}` |",
    ]
    source_lines = []
    for item in source_audit["sources"]:
        source_lines.append(
            f"- **{item['source']}**: usable_now=`{item['usable_now']}`; "
            f"vai trò: {item['role']}; giới hạn: {item['limitation'] or 'không ghi nhận'}."
        )
    raw = comparison.loc["raw_close"]
    source = comparison.loc["source_adjusted"]
    candidate = comparison.loc["research_total_return_candidate"]
    basket = ", ".join(latest["ticker"].astype(str).tolist())
    hypothesis_lines = [
        f"- **{row.hypothesis}**: `{row.status}` (Data A). Research V2: "
        "`not_testable_due_to_data`." for row in hypotheses.itertuples()
    ]
    report = [
        "# BÁO CÁO VẬN HÀNH HỆ THỐNG RESEARCH V2", "", *header, "",
        "## 1. Phạm vi và nguyên tắc diễn giải", "",
        "Hệ thống đã hoàn tất các phần có thể thực hiện độc lập gồm thu thập và hòa giải sự kiện doanh nghiệp, xây dựng bộ giá ứng viên, kiểm toán điều chỉnh giá, xây dựng universe point-in-time và đối chứng return-only. Pipeline xác nhận không được chạy vì cổng dữ liệu chưa đạt. Đây là hành vi fail-closed theo thiết kế, không phải một kết quả backtest thất bại.", "",
        "## 2. Nguồn dữ liệu", "", *source_lines, "",
        "## 3. Corporate actions", "",
        f"VSDC cung cấp {ca_audit['official_rows']:,} bản ghi sự kiện chính thức và CafeF cung cấp {ca_audit['reference_rows']:,} bản ghi đối chiếu. Ledger sau hòa giải có {ca_audit['ledger_rows']:,} dòng, trong đó {ca_audit['status_counts']['verified_cross_source']:,} dòng được xác minh chéo; {ca_audit['status_counts']['unresolved_ex_date']:,} dòng chưa xác minh ex-date; {ca_audit['status_counts']['conflict']:,} dòng xung đột và {ca_audit['status_counts']['reference_only_unverified']:,} dòng chỉ có nguồn tham khảo.", "",
        "## 4. Kiểm toán điều chỉnh giá", "",
        f"Price-adjustment gate có trạng thái **{adjustment['status']}**. Hệ thống đã áp dụng {adjustment['verified_events_applied']:,} sự kiện đã xác minh vào bộ dữ liệu ứng viên, nhưng vẫn còn {adjustment['unresolved_material_events']:,} sự kiện trọng yếu chưa giải quyết, {adjustment['unmatched_source_adjustment_changes_over_1pp']:,} thay đổi source-adjusted trên 1 điểm phần trăm chưa ghép được với sự kiện và {adjustment.get('unexplained_price_band_anomalies', 0):,} biến động raw vượt biên độ HOSE đã ghép theo khoảng cách phiên mà chưa có sự kiện giải thích. Vì vậy `research_eligible=false`.", "",
        "## 5. Đối chứng trực tiếp trên danh mục Data A đã đóng băng", "",
        "| Định nghĩa lợi nhuận | Gross cumulative return | Net cumulative return |",
        "|---|---:|---:|",
        f"| Raw close | {_pct(raw['gross_cumulative_return'])} | {_pct(raw['net_cumulative_return'])} |",
        f"| Source-adjusted close | {_pct(source['gross_cumulative_return'])} | {_pct(source['net_cumulative_return'])} |",
        f"| Research total-return candidate | {_pct(candidate['gross_cumulative_return'])} | {_pct(candidate['net_cumulative_return'])} |",
        "", "Phép so sánh này giữ nguyên mã, tỷ trọng và ngày tái cân bằng; do đó chỉ định lượng tác động trực tiếp của định nghĩa lợi nhuận. Cột research total-return vẫn là candidate vì gate bị chặn. Chênh lệch này không chứng minh corporate actions là nguyên nhân của lợi nhuận âm.", "",
        "## 6. Historical universe point-in-time", "",
        f"Security master PIT chứa {survivorship['security_master_rows']:,} mã trên {survivorship['decision_dates']} ngày quyết định tháng, giữ lại {survivorship['delisted_securities_retained_in_master']} mã đã hủy niêm yết và không dùng bộ lọc độ đầy đủ toàn giai đoạn tương lai. Trạng thái audit là `{survivorship['status']}` vì chưa có lịch sử đình chỉ đầy đủ và chưa xác minh toàn bộ đổi mã/sáp nhập pháp nhân.", "",
        "## 7. Baseline Data A được bảo toàn", "",
        f"Data A có dataset hash `{manifest['dataset_hash']}`, {manifest['folds_completed']}/{manifest['folds_requested']} folds, Rank IC trung bình {rankings.groupby('fold')['fold_rank_ic'].first().mean():.6f}, cumulative net return {_pct(baseline_metric['cumulative_return'])}, Sharpe {baseline_metric['sharpe']:.6f}, maximum drawdown {_pct(baseline_metric['max_drawdown'])}. XY-QAOA có feasibility {xy['feasibility_rate'].mean():.2%}, primary gap trung bình {xy['optimality_gap'].mean():.2%} và best-observed gap trung bình {xy['best_observed_gap'].mean():.2%}. Danh mục fold cuối: {basket}.", "",
        "## 8. H1–H6", "", *hypothesis_lines, "",
        "Research V2 không kế thừa trạng thái giả thuyết từ Data A. Khi chưa có experiment xác nhận, cả H1–H6 đều mang trạng thái `not_testable_due_to_data`.", "",
        "## 9. Kết luận", "",
        "Code path, dữ liệu chẩn đoán và blocked artifact đã được tạo có kiểm toán. Không có experiment ID, config hash, danh mục fold cuối hay kết quả H1–H6 mới vì hệ thống đã chủ động không huấn luyện trên dữ liệu chưa đạt hợp đồng xác nhận.", "",
    ]
    (REPORTS / "SYSTEM_RUN_REPORT_VI.md").write_text("\n".join(report), encoding="utf-8")

    trace = [
        "# REPORT SYSTEM TRACEABILITY", "", *header, "",
        "## Artifact map", "",
        "| Nội dung | Artifact nguồn | Trạng thái |", "|---|---|---|",
        "| Baseline Data A | `outputs/research_v2/audits/BASELINE_AUDIT.md` | frozen/pass |",
        "| Raw corporate actions | `outputs/research_v2/raw/corporate_actions/` | collected |",
        "| Ledger chuẩn hóa | `outputs/research_v2/normalized/corporate_actions.parquet` | diagnostic |",
        "| Corporate-action source audit | `outputs/research_v2/reports/corporate_action_source_audit.json` | unresolved |",
        "| Adjustment contract | `docs/contracts/price_adjustment_contract.v2.json` | versioned |",
        "| Total-return candidate | `outputs/research_v2/normalized/prices_total_return.parquet` | blocked candidate |",
        "| Adjustment audit | `outputs/research_v2/reports/price_adjustment_audit.json` | blocked |",
        "| Return-only comparison | `outputs/research_v2/reports/adjustment_return_only_summary.csv` | diagnostic complete |",
        "| Historical security master | `outputs/research_v2/normalized/security_master_pit.parquet` | partial |",
        "| Monthly PIT universe | `outputs/research_v2/curated/universe_monthly_pit.parquet` | partial |",
        "| Survivorship audit | `outputs/research_v2/reports/survivorship_bias_audit.json` | partial_blocked |",
        "| Verified total-return benchmark | `outputs/normalized/benchmark.parquet` | missing/blocked |",
        "| Research V2 experiment | `outputs/research_v2/experiments/BLOCKED_DATA_GATES/manifest.json` | intentionally not run |",
        "| Overall audit | `outputs/research_v2/audits/research_v2_audit.json` | blocked_valid |",
        "", "## Reproducibility commands", "",
        "```powershell", ".\\scripts\\run_research_v2.ps1", ".\\scripts\\audit_research_v2.ps1", ".\\scripts\\reproduce_research_v2.ps1", "```", "",
    ]
    (REPORTS / "REPORT_SYSTEM_TRACEABILITY.md").write_text("\n".join(trace), encoding="utf-8")
    print(json.dumps({
        "status": "generated", "system_report": str(REPORTS / "SYSTEM_RUN_REPORT_VI.md"),
        "traceability_report": str(REPORTS / "REPORT_SYSTEM_TRACEABILITY.md"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
