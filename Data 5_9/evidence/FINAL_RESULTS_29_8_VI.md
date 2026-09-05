# Kết quả tối ưu hóa thực tiễn — bản dữ liệu 29/8

## Lớp xác nhận lịch sử

- Cấu hình chọn chỉ từ development: `P_K10P6_CP30_NONE`.
- Holdout: folds 29--43.

| hypothesis                                      |     estimate |   statistic |   pvalue_one_sided | supported_5pct   |   holm_adjusted_pvalue | supported_holm_5pct   | evidence_label                            |
|:------------------------------------------------|-------------:|------------:|-------------------:|:-----------------|-----------------------:|:----------------------|:------------------------------------------|
| H1_QAUR_higher_QUR_objective                    |  0.0602777   |    5.08878  |        8.25358e-05 | True             |            0.000247607 | True                  | confirmatory_untouched_historical_holdout |
| H2_QAUR_lower_candidate_correlation             | -0.0123637   |   -6.90641  |        3.62481e-06 | True             |            1.44993e-05 | True                  | confirmatory_untouched_historical_holdout |
| H3_QAUR_turnover_noninferior_margin_2pp         | -0.06        |   -2.3864   |        0.0158414   | True             |            0.0316829   | True                  | confirmatory_untouched_historical_holdout |
| H4_QAUR_higher_mean_daily_return                | -3.57576e-05 |   -0.170742 |        0.567731    | False            |            0.567731    | False                 | confirmatory_untouched_historical_holdout |
| H5_QAUR_financial_direction_robust_across_seeds |  0           |  nan        |      nan           | False            |          nan           | False                 | confirmatory_untouched_historical_holdout |

XY-QAOA holdout audit (30 instances): mean feasibility
rate = 1.0000, mean optimality gap
= 0.000000.

## Lớp thiết kế phương pháp thực tiễn

- Cấu hình: `C1_IV_X`.
- Common market gate: 30 phiên.
- Nhãn bằng chứng: **post-hoc method design**, chưa phải prospective proof.

| config_id   |   market_gate_lookback | sample                       | method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:------------|-----------------------:|:-----------------------------|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| C1_IV_X     |                     30 | development_2022_2024        | AUR      |            604 |           0.236253  |           0.0925187 |               0.0942706 |         0.981416 |          0.965711 |          -0.157677 |
| C1_IV_X     |                     30 | development_2022_2024        | QAUR     |            604 |           0.236253  |           0.0925187 |               0.0942706 |         0.981416 |          0.965711 |          -0.157677 |
| C1_IV_X     |                     30 | historical_holdout_2024_2025 | AUR      |            312 |           0.1443    |           0.115019  |               0.131991  |         0.871415 |          0.984565 |          -0.130961 |
| C1_IV_X     |                     30 | historical_holdout_2024_2025 | QAUR     |            312 |           0.195703  |           0.155302  |               0.13636   |         1.13892  |          1.32701  |          -0.130961 |
| C1_IV_X     |                     30 | observed_2026                | AUR      |            160 |           0.0880689 |           0.142178  |               0.123103  |         1.15495  |          1.40085  |          -0.077845 |
| C1_IV_X     |                     30 | observed_2026                | QAUR     |            160 |           0.0880052 |           0.142073  |               0.123112  |         1.15401  |          1.39963  |          -0.077845 |

## H4 theo từng giai đoạn

| sample                       |   observations |   mean_daily_difference |   paired_t_statistic |   paired_t_pvalue_one_sided |   block_bootstrap_ci_low |   block_bootstrap_ci_high |   block_bootstrap_pvalue_one_sided | supported_5pct   | evidence_label                         |
|:-----------------------------|---------------:|------------------------:|---------------------:|----------------------------:|-------------------------:|--------------------------:|-----------------------------------:|:-----------------|:---------------------------------------|
| development_2022_2024        |            604 |             0           |           nan        |                  nan        |              0           |                0          |                          1         | False            | posthoc_method_design_not_confirmatory |
| historical_holdout_2024_2025 |            312 |             0.000143187 |             0.962903 |                    0.168172 |             -6.70964e-05 |                0.00038355 |                          0.0859828 | False            | posthoc_method_design_not_confirmatory |
| observed_2026                |            160 |            -3.61945e-07 |            -1        |                    0.840585 |             -3.61945e-07 |                0          |                          1         | False            | posthoc_method_design_not_confirmatory |

## Lợi nhuận dương: hiệu quả kinh tế và ý nghĩa thống kê

| sample                       | method   |   observations |   mean_daily_return |   cumulative_return |   one_sample_t_pvalue |   block_bootstrap_ci_low |   block_bootstrap_ci_high |   block_bootstrap_pvalue |   combined_conservative_pvalue |   holm_adjusted_pvalue | positive_economically   | positive_mean_supported_holm_5pct   | evidence_label                         |
|:-----------------------------|:---------|---------------:|--------------------:|--------------------:|----------------------:|-------------------------:|--------------------------:|-------------------------:|-------------------------------:|-----------------------:|:------------------------|:------------------------------------|:---------------------------------------|
| development_2022_2024        | AUR      |            604 |         0.000368843 |           0.236253  |             0.0637103 |             -5.40572e-05 |                0.00106504 |                0.0369926 |                      0.0637103 |               0.382262 | True                    | False                               | posthoc_method_design_not_confirmatory |
| development_2022_2024        | QAUR     |            604 |         0.000368843 |           0.236253  |             0.0637103 |             -5.40572e-05 |                0.00106504 |                0.0369926 |                      0.0637103 |               0.382262 | True                    | False                               | posthoc_method_design_not_confirmatory |
| historical_holdout_2024_2025 | AUR      |            312 |         0.000466568 |           0.1443    |             0.161187  |             -0.000391468 |                0.00141801 |                0.132573  |                      0.161187  |               0.483562 | True                    | False                               | posthoc_method_design_not_confirmatory |
| historical_holdout_2024_2025 | QAUR     |            312 |         0.000609756 |           0.195703  |             0.105419  |             -0.000292824 |                0.00166765 |                0.085183  |                      0.105419  |               0.421675 | True                    | False                               | posthoc_method_design_not_confirmatory |
| observed_2026                | AUR      |            160 |         0.000557516 |           0.0880689 |             0.182261  |             -0.000572626 |                0.00117406 |                0.192961  |                      0.192961  |               0.483562 | True                    | False                               | posthoc_method_design_not_confirmatory |
| observed_2026                | QAUR     |            160 |         0.000557154 |           0.0880052 |             0.182433  |             -0.000572626 |                0.00117406 |                0.192961  |                      0.192961  |               0.483562 | True                    | False                               | posthoc_method_design_not_confirmatory |

## Rổ tháng 9/2026

| method   | ticker   |   shadow_weight |   executable_weight |   cash_weight |   market_growth |
|:---------|:---------|----------------:|--------------------:|--------------:|----------------:|
| AUR      | NAF      |        0.3      |                   0 |             1 |      -0.0113525 |
| AUR      | STB      |        0.210668 |                   0 |             1 |      -0.0113525 |
| AUR      | VCB      |        0.24689  |                   0 |             1 |      -0.0113525 |
| AUR      | VJC      |        0.242442 |                   0 |             1 |      -0.0113525 |
| QAUR     | NAF      |        0.3      |                   0 |             1 |      -0.0113525 |
| QAUR     | STB      |        0.210668 |                   0 |             1 |      -0.0113525 |
| QAUR     | VCB      |        0.24689  |                   0 |             1 |      -0.0113525 |
| QAUR     | VJC      |        0.242442 |                   0 |             1 |      -0.0113525 |

## Kết luận hợp lệ

Phương pháp được phép chuyển sang paper trading không vốn từ 02/09/2026 nếu giữ
nguyên tham số. Không có kết quả nào trong lớp practical được diễn giải thành
quantum advantage hoặc cho phép triển khai vốn thật.
