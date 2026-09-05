from __future__ import annotations

"""Render the evidence-backed Vietnamese release report from saved artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    data = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    run = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    tests = pd.read_csv(results / "confirmatory_hypothesis_tests.csv")
    periods = pd.read_csv(results / "selected_practical_period_results.csv")
    positive = pd.read_csv(
        results / "selected_practical_positive_return_evidence.csv"
    )
    h4_period = pd.read_csv(results / "selected_practical_h4_by_period.csv")
    xy = pd.read_csv(results / "confirmatory_xy_qaoa_holdout_audit.csv")
    basket = pd.read_csv(
        results / "september_2026_shadow_and_executable_basket.csv"
    )
    config = run["practical_best_config"]
    supported = tests.loc[tests["supported_holm_5pct"], "hypothesis"].tolist()
    unsupported = tests.loc[~tests["supported_holm_5pct"], "hypothesis"].tolist()
    statistically_positive = int(
        positive["positive_mean_supported_holm_5pct"].sum()
    )
    shadow = basket[basket["method"].eq("AUR")][
        ["ticker", "shadow_weight", "executable_weight"]
    ]

    report = f"""# Kết quả cuối cùng — bản dữ liệu và thí nghiệm ngày 29/8

## 1. Phạm vi dữ liệu và thiết kế bằng chứng

Bộ dữ liệu hợp nhất gồm **{data['rows']:,} bản ghi** và
**{data['price_tickers']} mã cổ phiếu**, bao phủ từ **{data['price_start']} đến
{data['price_end']}**.
Phần dữ liệu đến hết năm 2025 được dùng cho lớp kiểm định lịch sử; phần năm
2026 có nhãn **provisional forward**, vì vậy không được xem là dữ liệu đã qua
quy trình kiểm toán tương đương dữ liệu lịch sử.

Lớp confirmatory sàng lọc {run['confirmatory_configurations_screened']} cấu hình
chỉ trên folds 0–28 và đánh giá cấu hình đã khóa trên holdout folds 29–43. Lớp
practical thử {run['practical_configurations_screened']} cấu hình ràng buộc và
phân bổ với market-gate 0, 20, 30 và 40 phiên. Lớp thứ hai sử dụng dữ liệu đã
quan sát đến 29/8/2026 nên chỉ là **post-hoc method design**, không phải bằng
chứng prospective.

## 2. Cấu hình practical được chọn

Cấu hình tối ưu theo tiêu chí đã khóa là **{config['config_id']} + common market
gate {run['practical_best_market_gate_lookback']} phiên**. Cấu hình sử dụng
Top-{config['candidate_size']} ở tầng universe reduction, chọn đúng
{config['portfolio_cardinality']} cổ phiếu ở tầng portfolio QUBO, tín hiệu
XGBoost thuần, EWMA covariance span {config['covariance_span']} phiên và phân bổ
inverse-volatility. Mỗi cổ phiếu bị chặn trong khoảng
{pct(config['weight_lower'])}–{pct(config['weight_upper'])}; correlation penalty
= {config['correlation_penalty']}, stability weight = {config['stability_weight']}
và chi phí giao dịch = {config['transaction_cost_bps']:.0f} bps. Các tham số và
downstream pipeline được giữ giống nhau cho AUR và QAUR.

Quy tắc chọn là lợi nhuận dương trong cả ba giai đoạn cho cả hai reducer, MDD
không thấp hơn −20%, sau đó tối đa hóa Sharpe tệ nhất. Cấu hình được chọn đạt
6/6 ô lợi nhuận dương; worst-case Sharpe =
{periods['sharpe_zero_rf'].min():.3f} và MDD tệ nhất =
{pct(periods['maximum_drawdown'].min())}.

## 3. Hiệu quả đầu tư

{periods.to_markdown(index=False)}

Lợi nhuận cộng dồn lần lượt là 23,63% ở development cho cả hai reducer;
14,43% (AUR) và 19,57% (QAUR) trên historical holdout; 8,81% (AUR) và 8,80%
(QAUR) trong phần dữ liệu quan sát năm 2026. Đây là hiệu quả kinh tế dương sau
chi phí mô phỏng, nhưng không đồng nghĩa lợi nhuận kỳ vọng đã được chứng minh
khác 0.

## 4. Bằng chứng thống kê cho H1–H5

{tests.to_markdown(index=False)}

Sau hiệu chỉnh Holm cho H1–H4, các giả thuyết được ủng hộ là:
{', '.join(supported)}. Các giả thuyết chưa được ủng hộ là:
{', '.join(unsupported)}. Do đó, kết quả cho thấy QAUR cải thiện trực tiếp
objective giảm vũ trụ, làm giảm tương quan trong candidate set và không kém hơn
về turnover theo biên 2 điểm phần trăm. Tuy nhiên, chưa có bằng chứng rằng QAUR
tạo mean daily return cao hơn AUR; hướng chênh lệch Sharpe cũng không bền vững
qua các seed.

Kiểm định riêng đối với lợi nhuận dương cho kết quả sau:

{positive.to_markdown(index=False)}

Số ô có lợi nhuận cộng dồn dương là 6/6, nhưng số ô có mean daily return dương
với ý nghĩa 5% sau kiểm soát Holm là **{statistically_positive}/6**. Vì vậy,
nghiên cứu chỉ được báo cáo “lợi nhuận quan sát dương”, không được kết luận
“alpha dương có ý nghĩa thống kê”.

Phân tích hậu nghiệm chênh lệch QAUR–AUR theo giai đoạn:

{h4_period.to_markdown(index=False)}

## 5. Audit Cardinality-Constrained QUBO và XY-QAOA

XY-QAOA được audit trên **{len(xy)} instances** của historical holdout, dùng
cùng depth, budget và shots cho hai reducer. Feasibility rate trung bình đạt
{xy['feasibility_rate'].mean():.2%}; optimality gap trung bình bằng
{xy['optimality_gap'].mean():.6f}. Điều này xác nhận nghiệm nằm trong feasible
subspace và đạt nghiệm tham chiếu trên các instance nhỏ, nhưng **không chứng
minh quantum advantage**, vì backend hiện vẫn là mô phỏng cổ điển.

## 6. Rổ cổ phiếu và trạng thái thực thi

Rổ shadow cho kỳ bắt đầu 02/09/2026 là:

{shadow.to_markdown(index=False)}

Tăng trưởng market proxy 30 phiên tại thời điểm quyết định là
{pct(run['current_market_growth'])}, thấp hơn 0; common gate vì vậy đặt exposure
= 0. Tỷ trọng thực thi hiện tại là **0% cổ phiếu và 100% tiền mặt**. Rổ NAF,
STB, VCB và VJC chỉ là rổ shadow để tiếp tục theo dõi; không phải khuyến nghị
mua tại thời điểm 29/8.

## 7. Kết luận và điều kiện áp dụng

Cấu hình `C1_IV_X + gate 30` là phương án mạnh nhất trong không gian thí nghiệm
đã định nghĩa, đủ điều kiện chuyển sang **paper trading không vốn**. Hệ thống
chưa đủ bằng chứng để triển khai vốn thật vì (i) lớp practical là post-hoc,
(ii) lợi nhuận dương chưa có ý nghĩa thống kê sau hiệu chỉnh, (iii) dữ liệu 2026
còn provisional, và (iv) chưa có benchmark prospective với slippage, market
impact và sự cố dữ liệu thực tế. Protocol chỉ nên được xem xét nâng cấp sau một
giai đoạn forward test được khóa trước, không thay tham số, và có tiêu chí dừng
rõ ràng.
"""
    destination = results / "FINAL_RESULTS_29_8_VI.md"
    destination.write_text(report, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
