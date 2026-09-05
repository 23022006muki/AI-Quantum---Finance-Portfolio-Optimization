from __future__ import annotations

"""Build a Colab release whose verified results remain visible after reload.

The full standalone implementation and embedded dataset are preserved.  This
builder adds a report-first section made from the audited local release so the
research tables, hypotheses, portfolio basket, and figure are visible before a
runtime is connected.  Running all cells still recomputes every artifact.
"""

import base64
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "colab" / "AUR_QAUR_Standalone_Full_Web_Colab_30_8.ipynb"
OUTPUT = ROOT / "colab" / "AUR_QAUR_RESULTS_INCLUDED_FULL_WEB_31_8.ipynb"
RESULTS = ROOT / "outputs" / "release_29_8_final"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def table(path: str, *, columns: list[str] | None = None) -> str:
    frame = pd.read_csv(RESULTS / path)
    if columns is not None:
        frame = frame[columns]
    return frame.to_markdown(index=False, floatfmt=".6f")


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    practical = manifest["practical_best_config"]

    diagnostics = pd.read_csv(RESULTS / "forecast_diagnostics.csv")
    forecast_summary = pd.DataFrame(
        [
            {
                "walk_forward_folds": len(diagnostics),
                "mean_validation_rank_ic": diagnostics["validation_rank_ic"].mean(),
                "median_validation_rank_ic": diagnostics["validation_rank_ic"].median(),
                "positive_rank_ic_folds": int((diagnostics["validation_rank_ic"] > 0).sum()),
                "mean_validation_rmse": diagnostics["validation_rmse"].mean(),
                "mean_universe_size": diagnostics["universe_size"].mean(),
            }
        ]
    )

    inventory = pd.DataFrame(
        [
            {"file": path.name, "bytes": path.stat().st_size}
            for path in sorted(RESULTS.iterdir())
            if path.is_file()
        ]
    )

    results_cells = [
        markdown(
            f"""
# Kết quả đã xác minh — hiển thị sẵn, không cần kết nối runtime

Các bảng dưới đây được nhúng từ lần chạy end-to-end đã vượt qua
`RESEARCH_AUDIT_29_8_OK`. Phần mã nguồn đầy đủ và dữ liệu standalone nằm phía
dưới; chọn **Runtime → Run all** để tái tạo độc lập toàn bộ artifacts.

- Dữ liệu quan sát đến: **{manifest['observed_price_end']}**.
- SHA-256 dữ liệu: `{manifest['dataset_sha256']}`.
- Confirmatory split: development **0–28**, untouched holdout **29–43**.
- Cấu hình thực tiễn: **{practical['config_id']}**, `K={practical['candidate_size']}`,
  `k_p={practical['portfolio_cardinality']}`, inverse-volatility,
  weight bounds **{practical['weight_lower']:.0%}–{practical['weight_upper']:.0%}**,
  transaction cost **{practical['transaction_cost_bps']:.0f} bps**.
- Common market gate: **{manifest['practical_best_market_gate_lookback']} phiên**.
- XY-QAOA audit: **{manifest['xy_qaoa_holdout_audit_instances']} instances**,
  feasibility trung bình **{manifest['xy_qaoa_mean_feasibility_rate']:.0%}**.
- Không cho phép vốn thật; không tuyên bố quantum advantage.
"""
        ),
        markdown(
            "## H1–H5 — kiểm định xác nhận trên historical holdout\n\n"
            + table("confirmatory_hypothesis_tests.csv")
            + "\n\n**Kết luận:** H1, H2 và H3 được ủng hộ sau Holm 5%; H4 và H5 chưa được ủng hộ."
        ),
        markdown(
            "## Hiệu quả đầu tư của cấu hình thực tiễn được chọn\n\n"
            + table("selected_practical_period_results.csv")
        ),
        markdown(
            "## Lợi nhuận dương và bằng chứng thống kê\n\n"
            + table("selected_practical_positive_return_evidence.csv")
            + "\n\nLợi nhuận tích lũy dương ở 6/6 ô period × reducer, nhưng mean daily return "
            "không đạt ý nghĩa sau hiệu chỉnh Holm; đây là hiệu quả kinh tế hậu nghiệm, "
            "chưa phải prospective proof."
        ),
        markdown(
            "## H4 theo từng giai đoạn\n\n"
            + table("selected_practical_h4_by_period.csv")
        ),
        markdown(
            "## Rổ cổ phiếu tháng 9/2026 — shadow và executable\n\n"
            + table("september_2026_shadow_and_executable_basket.csv")
            + "\n\nMarket gate 30 phiên đang âm nên `executable_weight = 0` và cash = 100%; "
            "shadow basket chỉ dùng cho paper trading."
        ),
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {"tags": ["precomputed-output"]},
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "image/png": base64.b64encode(
                            (RESULTS / "selected_practical_results_and_basket.png").read_bytes()
                        ).decode("ascii"),
                        "text/plain": ["<Precomputed AUR–QAUR results figure>"],
                    },
                    "metadata": {},
                }
            ],
            "source": "# Precomputed figure from the audited web-equivalent release.\n",
        },
        markdown(
            "## Chẩn đoán dự báo XGBoost/EWMA\n\n### Tổng hợp\n\n"
            + forecast_summary.to_markdown(index=False, floatfmt=".6f")
            + "\n\n### Toàn bộ 54 folds\n\n"
            + table("forecast_diagnostics.csv")
        ),
        markdown(
            "## QAUR seed robustness\n\n### Confirmatory\n\n"
            + table("confirmatory_seed_robustness.csv")
            + "\n\n### Practical\n\n"
            + table("selected_practical_seed_robustness.csv")
        ),
        markdown(
            "## Toàn bộ 30 XY-QAOA holdout audit instances\n\n"
            + table("confirmatory_xy_qaoa_holdout_audit.csv")
        ),
        markdown(
            "## Toàn bộ 172 kết quả confirmatory (43 cấu hình × 2 mẫu × 2 reducers)\n\n"
            + table("confirmatory_configuration_results.csv")
        ),
        markdown(
            "## Toàn bộ xếp hạng 96 phương án practical (24 cấu hình × 4 gates)\n\n"
            + table("practical_robust_ranking.csv")
        ),
        markdown(
            "## Danh mục artifacts đầy đủ\n\n"
            + inventory.to_markdown(index=False)
        ),
        markdown((RESULTS / "FINAL_RESULTS_29_8_VI.md").read_text(encoding="utf-8")),
        markdown("---\n\n# Mã nguồn standalone và quy trình tái lập đầy đủ"),
    ]

    notebook["cells"] = [notebook["cells"][0], *results_cells, *notebook["cells"][1:]]
    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
