# August 2026 quasi-holdout

The candidate was selected using complete folds only through 2026-07-31. August
returns were not used to select or tune it. Because the raw August data already
existed before this run, this remains a quasi-holdout rather than prospective
paper trading.

## Candidate

| method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| AUR      |             20 |           0.0205352 |            0.291912 |                0.176112 |          1.65753 |            1.8477 |         -0.0516629 |
| QAUR     |             20 |           0.0205352 |            0.291912 |                0.176112 |          1.65753 |            1.8477 |         -0.0516629 |

## Full-universe equal-weight baseline

| method           |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:-----------------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| FULL_UNIVERSE_EW |             20 |           0.0053418 |           0.0694318 |               0.0801371 |         0.866414 |           2.04454 |         -0.0156887 |

## Portfolio decided before the August test interval

| config_id   |   fold | method   | ticker   | selected_downstream   |   weight |
|:------------|-------:|:---------|:---------|:----------------------|---------:|
| C1_IV_X     |     52 | AUR      | BMP      | False                 | 0        |
| C1_IV_X     |     52 | AUR      | BWE      | True                  | 0.3      |
| C1_IV_X     |     52 | AUR      | DCL      | True                  | 0.272805 |
| C1_IV_X     |     52 | AUR      | DCM      | False                 | 0        |
| C1_IV_X     |     52 | AUR      | GVR      | False                 | 0        |
| C1_IV_X     |     52 | AUR      | STB      | True                  | 0.18988  |
| C1_IV_X     |     52 | AUR      | VPI      | True                  | 0.237315 |
| C1_IV_X     |     52 | AUR      | VPL      | False                 | 0        |
| C1_IV_X     |     52 | QAUR     | BMP      | False                 | 0        |
| C1_IV_X     |     52 | QAUR     | BWE      | True                  | 0.3      |
| C1_IV_X     |     52 | QAUR     | DCL      | True                  | 0.272805 |
| C1_IV_X     |     52 | QAUR     | DCM      | False                 | 0        |
| C1_IV_X     |     52 | QAUR     | GVR      | False                 | 0        |
| C1_IV_X     |     52 | QAUR     | STB      | True                  | 0.18988  |
| C1_IV_X     |     52 | QAUR     | VPI      | True                  | 0.237315 |
| C1_IV_X     |     52 | QAUR     | VPL      | False                 | 0        |
