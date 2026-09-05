# Frozen 2026 forward evaluation

| method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| AUR      |            140 |          -0.0680122 |           -0.119076 |                0.13622  |        -0.874143 |          -1.15567 |          -0.114952 |
| QAUR     |            140 |          -0.129303  |           -0.220599 |                0.147253 |        -1.49809  |          -1.96508 |          -0.173761 |

## Full-universe baseline

| method           |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:-----------------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| FULL_UNIVERSE_EW |            140 |          -0.0556872 |          -0.0979958 |                0.117846 |        -0.831555 |         -0.903883 |          -0.108587 |

## Deflated Sharpe

| method   |   observed_annual_sharpe_arithmetic |   expected_max_annual_sharpe_under_trials |   deflated_sharpe_probability |   number_of_trials |   skewness |   pearson_kurtosis |
|:---------|------------------------------------:|------------------------------------------:|------------------------------:|-------------------:|-----------:|-------------------:|
| AUR      |                           -0.862598 |                                 0.0896365 |                     0.236413  |                 44 |  -0.598851 |            5.10005 |
| QAUR     |                           -1.61824  |                                 0.0896365 |                     0.0965345 |                 44 |  -0.584063 |            4.5718  |

## Bootstrap versus Full-Universe EW

| method   | comparator       |   observations |   mean_daily_difference |   annualized_arithmetic_difference |    ci_2_5 |   ci_97_5 |   pvalue_one_sided_positive |
|:---------|:-----------------|---------------:|------------------------:|-----------------------------------:|----------:|----------:|----------------------------:|
| AUR      | FULL_UNIVERSE_EW |            140 |            -8.47371e-05 |                         -0.0213537 | -0.29351  |  0.221925 |                      0.5812 |
| QAUR     | FULL_UNIVERSE_EW |            140 |            -0.000564055 |                         -0.142142  | -0.556097 |  0.173734 |                      0.7966 |

## Gates

| gate                                        | passed   |
|:--------------------------------------------|:---------|
| forward_return_positive_both                | False    |
| forward_sharpe_above_1_both                 | False    |
| forward_drawdown_below_20pct                | True     |
| forward_DSR_probability_above_95pct         | False    |
| significant_forward_excess_vs_full_EW       | False    |
| minimum_24_month_forward_record             | False    |
| cross_source_and_corporate_action_certified | False    |

No parameter was retuned on the 2026 period. The source panel remains provisional and cannot authorize live capital by itself.
