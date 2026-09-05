# Recovery candidate robustness audit

The candidate below was selected after inspecting 2026 and therefore defines a
future protocol; it is not evidence from an untouched test.

## Candidate ranking on observed 2026

| run          |   worst_method_sharpe |   worst_method_return |   worst_method_drawdown |
|:-------------|----------------------:|----------------------:|------------------------:|
| C1_IV_X      |              1.89703  |             0.156707  |              -0.107788  |
| C1_IV_X_QAWS |              1.89703  |             0.156707  |              -0.107788  |
| C2_IV_X      |              0.853523 |             0.0633075 |              -0.0662521 |
| C2_IV_X_QAWS |              0.853523 |             0.0633075 |              -0.0662521 |
| C3_IV_X      |              0.243607 |             0.0187428 |              -0.0891847 |
| C3_IV_X_QAWS |              0.243607 |             0.0187428 |              -0.0891847 |

## Selected candidate across time regimes

| sample                       |   cumulative_return_AUR |   cumulative_return_QAUR |   sharpe_zero_rf_AUR |   sharpe_zero_rf_QAUR |   maximum_drawdown_AUR |   maximum_drawdown_QAUR |
|:-----------------------------|------------------------:|-------------------------:|---------------------:|----------------------:|-----------------------:|------------------------:|
| bridge_december_2025         |              0.00165707 |              -0.00426101 |             0.140395 |             -0.270901 |             -0.0573997 |              -0.0505494 |
| development_2022_2024        |             -0.157747   |              -0.157747   |            -0.425673 |             -0.425673 |             -0.487182  |              -0.487182  |
| historical_holdout_2024_2025 |              0.185274   |               0.248558   |             1.06527  |              1.38729  |             -0.130961  |              -0.130961  |
| observed_forward_2026        |              0.156775   |               0.156707   |             1.898    |              1.89703  |             -0.107788  |              -0.107788  |

## Transaction-cost stress

| run             | sample                | method   |   cost_bps |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:----------------|:----------------------|:---------|-----------:|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| C1_IV_X_COST0   | observed_forward_2026 | AUR      |          0 |            140 |            0.168005 |            0.322515 |                0.158194 |          2.03873 |           3.36397 |          -0.106098 |
| C1_IV_X_COST0   | observed_forward_2026 | QAUR     |          0 |            140 |            0.168005 |            0.322515 |                0.158194 |          2.03873 |           3.36397 |          -0.106098 |
| C1_IV_X_COST25  | observed_forward_2026 | AUR      |         25 |            140 |            0.156775 |            0.299714 |                0.157911 |          1.898   |           3.10581 |          -0.107788 |
| C1_IV_X_COST25  | observed_forward_2026 | QAUR     |         25 |            140 |            0.156707 |            0.299577 |                0.157919 |          1.89703 |           3.10407 |          -0.107788 |
| C1_IV_X_COST50  | observed_forward_2026 | AUR      |         50 |            140 |            0.145635 |            0.277272 |                0.157789 |          1.75723 |           2.87412 |          -0.109476 |
| C1_IV_X_COST50  | observed_forward_2026 | QAUR     |         50 |            140 |            0.145501 |            0.277002 |                0.157807 |          1.75532 |           2.87056 |          -0.109476 |
| C1_IV_X_COST75  | observed_forward_2026 | AUR      |         75 |            140 |            0.134586 |            0.255184 |                0.15783  |          1.61683 |           2.64639 |          -0.111163 |
| C1_IV_X_COST75  | observed_forward_2026 | QAUR     |         75 |            140 |            0.134386 |            0.254786 |                0.157859 |          1.61401 |           2.64098 |          -0.111163 |
| C1_IV_X_COST100 | observed_forward_2026 | AUR      |        100 |            140 |            0.123627 |            0.233445 |                0.158033 |          1.47719 |           2.42343 |          -0.112849 |
| C1_IV_X_COST100 | observed_forward_2026 | QAUR     |        100 |            140 |            0.123363 |            0.232923 |                0.158074 |          1.47351 |           2.41619 |          -0.112849 |

## QAUR seed stress on observed 2026

|           seed |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|---------------:|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
|    7           |            140 |            0.156707 |            0.299577 |                0.157919 |          1.89703 |           3.10407 |          -0.107788 |
|   29           |            140 |            0.156707 |            0.299577 |                0.157919 |          1.89703 |           3.10407 |          -0.107788 |
|  101           |            140 |            0.156707 |            0.299577 |                0.157919 |          1.89703 |           3.10407 |          -0.107788 |
| 1009           |            140 |            0.156707 |            0.299577 |                0.157919 |          1.89703 |           3.10407 |          -0.107788 |
|    2.02608e+07 |            140 |            0.156707 |            0.299577 |                0.157919 |          1.89703 |           3.10407 |          -0.107788 |

## Latest component portfolio

| config_id   |   fold | method   | ticker   | selected_downstream   |   weight |
|:------------|-------:|:---------|:---------|:----------------------|---------:|
| C1_IV_X     |     51 | AUR      | BWE      | True                  | 0.3      |
| C1_IV_X     |     51 | AUR      | DCL      | False                 | 0        |
| C1_IV_X     |     51 | AUR      | DCM      | True                  | 0.172543 |
| C1_IV_X     |     51 | AUR      | PAN      | False                 | 0        |
| C1_IV_X     |     51 | AUR      | STB      | False                 | 0        |
| C1_IV_X     |     51 | AUR      | VCB      | True                  | 0.277695 |
| C1_IV_X     |     51 | AUR      | VPI      | True                  | 0.249761 |
| C1_IV_X     |     51 | AUR      | VPL      | False                 | 0        |
| C1_IV_X     |     51 | QAUR     | BWE      | True                  | 0.3      |
| C1_IV_X     |     51 | QAUR     | DCL      | False                 | 0        |
| C1_IV_X     |     51 | QAUR     | DCM      | True                  | 0.172543 |
| C1_IV_X     |     51 | QAUR     | PAN      | False                 | 0        |
| C1_IV_X     |     51 | QAUR     | STB      | False                 | 0        |
| C1_IV_X     |     51 | QAUR     | VCB      | True                  | 0.277695 |
| C1_IV_X     |     51 | QAUR     | VPI      | True                  | 0.249761 |
| C1_IV_X     |     51 | QAUR     | VPL      | False                 | 0        |

Because selection used observed 2026, the only valid next confirmation is a new
untouched paper-trading period with frozen code, data policy and parameters.
