# Kết quả cuối cùng — bản dữ liệu và thí nghiệm ngày 29/8

## 1. Phạm vi dữ liệu và thiết kế bằng chứng

Bộ dữ liệu hợp nhất gồm **179,173 bản ghi** và
**120 mã cổ phiếu**, bao phủ từ **2020-01-02 đến
2026-08-28**.
Phần dữ liệu đến hết năm 2025 được dùng cho lớp kiểm định lịch sử; phần năm
2026 có nhãn **provisional forward**, vì vậy không được xem là dữ liệu đã qua
quy trình kiểm toán tương đương dữ liệu lịch sử.

Lớp confirmatory sàng lọc 43 cấu hình
chỉ trên folds 0–28 và đánh giá cấu hình đã khóa trên holdout folds 29–43. Lớp
practical thử 24 cấu hình ràng buộc và
phân bổ với market-gate 0, 20, 30 và 40 phiên. Lớp thứ hai sử dụng dữ liệu đã
quan sát đến 29/8/2026 nên chỉ là **post-hoc method design**, không phải bằng
chứng prospective.

## 2. Cấu hình practical được chọn

Cấu hình tối ưu theo tiêu chí đã khóa là **C1_IV_X + common market
gate 30 phiên**. Cấu hình sử dụng
Top-8 ở tầng universe reduction, chọn đúng
4 cổ phiếu ở tầng portfolio QUBO, tín hiệu
XGBoost thuần, EWMA covariance span 60 phiên và phân bổ
inverse-volatility. Mỗi cổ phiếu bị chặn trong khoảng
5.00%–30.00%; correlation penalty
= 0.1, stability weight = 0.15
và chi phí giao dịch = 25 bps. Các tham số và
downstream pipeline được giữ giống nhau cho AUR và QAUR.

Quy tắc chọn là lợi nhuận dương trong cả ba giai đoạn cho cả hai reducer, MDD
không thấp hơn −20%, sau đó tối đa hóa Sharpe tệ nhất. Cấu hình được chọn đạt
6/6 ô lợi nhuận dương; worst-case Sharpe =
0.871 và MDD tệ nhất =
-15.77%.

## 3. Hiệu quả đầu tư

| config_id   |   market_gate_lookback | sample                       | method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:------------|-----------------------:|:-----------------------------|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| C1_IV_X     |                     30 | development_2022_2024        | AUR      |            604 |           0.236253  |           0.0925187 |               0.0942706 |         0.981416 |          0.965711 |          -0.157677 |
| C1_IV_X     |                     30 | development_2022_2024        | QAUR     |            604 |           0.236253  |           0.0925187 |               0.0942706 |         0.981416 |          0.965711 |          -0.157677 |
| C1_IV_X     |                     30 | historical_holdout_2024_2025 | AUR      |            312 |           0.1443    |           0.115019  |               0.131991  |         0.871415 |          0.984565 |          -0.130961 |
| C1_IV_X     |                     30 | historical_holdout_2024_2025 | QAUR     |            312 |           0.195703  |           0.155302  |               0.13636   |         1.13892  |          1.32701  |          -0.130961 |
| C1_IV_X     |                     30 | observed_2026                | AUR      |            160 |           0.0880689 |           0.142178  |               0.123103  |         1.15495  |          1.40085  |          -0.077845 |
| C1_IV_X     |                     30 | observed_2026                | QAUR     |            160 |           0.0880052 |           0.142073  |               0.123112  |         1.15401  |          1.39963  |          -0.077845 |

Lợi nhuận cộng dồn lần lượt là 23,63% ở development cho cả hai reducer;
14,43% (AUR) và 19,57% (QAUR) trên historical holdout; 8,81% (AUR) và 8,80%
(QAUR) trong phần dữ liệu quan sát năm 2026. Đây là hiệu quả kinh tế dương sau
chi phí mô phỏng, nhưng không đồng nghĩa lợi nhuận kỳ vọng đã được chứng minh
khác 0.

## 4. Bằng chứng thống kê cho H1–H5

| hypothesis                                      |     estimate |   statistic |   pvalue_one_sided | supported_5pct   | evidence_label                            |   holm_adjusted_pvalue | supported_holm_5pct   |
|:------------------------------------------------|-------------:|------------:|-------------------:|:-----------------|:------------------------------------------|-----------------------:|:----------------------|
| H1_QAUR_higher_QUR_objective                    |  0.0602777   |    5.08878  |        8.25358e-05 | True             | confirmatory_untouched_historical_holdout |            0.000247607 | True                  |
| H2_QAUR_lower_candidate_correlation             | -0.0123637   |   -6.90641  |        3.62481e-06 | True             | confirmatory_untouched_historical_holdout |            1.44993e-05 | True                  |
| H3_QAUR_turnover_noninferior_margin_2pp         | -0.06        |   -2.3864   |        0.0158414   | True             | confirmatory_untouched_historical_holdout |            0.0316829   | True                  |
| H4_QAUR_higher_mean_daily_return                | -3.57576e-05 |   -0.170742 |        0.567731    | False            | confirmatory_untouched_historical_holdout |            0.567731    | False                 |
| H5_QAUR_financial_direction_robust_across_seeds |  0           |  nan        |      nan           | False            | confirmatory_untouched_historical_holdout |          nan           | False                 |

Sau hiệu chỉnh Holm cho H1–H4, các giả thuyết được ủng hộ là:
H1_QAUR_higher_QUR_objective, H2_QAUR_lower_candidate_correlation, H3_QAUR_turnover_noninferior_margin_2pp. Các giả thuyết chưa được ủng hộ là:
H4_QAUR_higher_mean_daily_return, H5_QAUR_financial_direction_robust_across_seeds. Do đó, kết quả cho thấy QAUR cải thiện trực tiếp
objective giảm vũ trụ, làm giảm tương quan trong candidate set và không kém hơn
về turnover theo biên 2 điểm phần trăm. Tuy nhiên, chưa có bằng chứng rằng QAUR
tạo mean daily return cao hơn AUR; hướng chênh lệch Sharpe cũng không bền vững
qua các seed.

Kiểm định riêng đối với lợi nhuận dương cho kết quả sau:

| sample                       | method   |   observations |   mean_daily_return |   cumulative_return |   one_sample_t_pvalue |   block_bootstrap_ci_low |   block_bootstrap_ci_high |   block_bootstrap_pvalue |   combined_conservative_pvalue |   holm_adjusted_pvalue | positive_economically   | positive_mean_supported_holm_5pct   | evidence_label                         |
|:-----------------------------|:---------|---------------:|--------------------:|--------------------:|----------------------:|-------------------------:|--------------------------:|-------------------------:|-------------------------------:|-----------------------:|:------------------------|:------------------------------------|:---------------------------------------|
| development_2022_2024        | AUR      |            604 |         0.000368843 |           0.236253  |             0.0637103 |             -5.40572e-05 |                0.00106504 |                0.0369926 |                      0.0637103 |               0.382262 | True                    | False                               | posthoc_method_design_not_confirmatory |
| development_2022_2024        | QAUR     |            604 |         0.000368843 |           0.236253  |             0.0637103 |             -5.40572e-05 |                0.00106504 |                0.0369926 |                      0.0637103 |               0.382262 | True                    | False                               | posthoc_method_design_not_confirmatory |
| historical_holdout_2024_2025 | AUR      |            312 |         0.000466568 |           0.1443    |             0.161187  |             -0.000391468 |                0.00141801 |                0.132573  |                      0.161187  |               0.483562 | True                    | False                               | posthoc_method_design_not_confirmatory |
| historical_holdout_2024_2025 | QAUR     |            312 |         0.000609756 |           0.195703  |             0.105419  |             -0.000292824 |                0.00166765 |                0.085183  |                      0.105419  |               0.421675 | True                    | False                               | posthoc_method_design_not_confirmatory |
| observed_2026                | AUR      |            160 |         0.000557516 |           0.0880689 |             0.182261  |             -0.000572626 |                0.00117406 |                0.192961  |                      0.192961  |               0.483562 | True                    | False                               | posthoc_method_design_not_confirmatory |
| observed_2026                | QAUR     |            160 |         0.000557154 |           0.0880052 |             0.182433  |             -0.000572626 |                0.00117406 |                0.192961  |                      0.192961  |               0.483562 | True                    | False                               | posthoc_method_design_not_confirmatory |

Số ô có lợi nhuận cộng dồn dương là 6/6, nhưng số ô có mean daily return dương
với ý nghĩa 5% sau kiểm soát Holm là **0/6**. Vì vậy,
nghiên cứu chỉ được báo cáo “lợi nhuận quan sát dương”, không được kết luận
“alpha dương có ý nghĩa thống kê”.

Phân tích hậu nghiệm chênh lệch QAUR–AUR theo giai đoạn:

| sample                       |   observations |   mean_daily_difference |   paired_t_statistic |   paired_t_pvalue_one_sided |   block_bootstrap_ci_low |   block_bootstrap_ci_high |   block_bootstrap_pvalue_one_sided | supported_5pct   | evidence_label                         |
|:-----------------------------|---------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|--------------------------:|-----------------------------------:|:-----------------|:---------------------------------------|
| development_2022_2024        |            604 |             0           |           nan        |                  nan        |              0           |                0          |                          1         | False            | posthoc_method_design_not_confirmatory |
| historical_holdout_2024_2025 |            312 |             0.000143187 |             0.962903 |                    0.168172 |             -6.70964e-05 |                0.00038355 |                          0.0859828 | False            | posthoc_method_design_not_confirmatory |
| observed_2026                |            160 |            -3.61945e-07 |            -1        |                    0.840585 |             -3.61945e-07 |                0          |                          1         | False            | posthoc_method_design_not_confirmatory |

## 5. Audit Cardinality-Constrained QUBO và XY-QAOA

XY-QAOA được audit trên **30 instances** của historical holdout, dùng
cùng depth, budget và shots cho hai reducer. Feasibility rate trung bình đạt
100.00%; optimality gap trung bình bằng
0.000000. Điều này xác nhận nghiệm nằm trong feasible
subspace và đạt nghiệm tham chiếu trên các instance nhỏ, nhưng **không chứng
minh quantum advantage**, vì backend hiện vẫn là mô phỏng cổ điển.

## 6. Rổ cổ phiếu và trạng thái thực thi

Rổ shadow cho kỳ bắt đầu 02/09/2026 là:

| ticker   |   shadow_weight |   executable_weight |
|:---------|----------------:|--------------------:|
| NAF      |        0.3      |                   0 |
| STB      |        0.210668 |                   0 |
| VCB      |        0.24689  |                   0 |
| VJC      |        0.242442 |                   0 |

Tăng trưởng market proxy 30 phiên tại thời điểm quyết định là
-1.14%, thấp hơn 0; common gate vì vậy đặt exposure
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
