# 2026 pre-existing grid audit (post-hoc diagnostic)

This audit evaluates specifications that were already encoded before the 2026
result was inspected. Choosing a winner from this table is nevertheless a
post-selection decision and is **not** a new forward test.

## Most robust across AUR and QAUR

| config_id   |   cumulative_return_AUR |   cumulative_return_QAUR |   maximum_drawdown_AUR |   maximum_drawdown_QAUR |   sharpe_zero_rf_AUR |   sharpe_zero_rf_QAUR |   worst_method_sharpe |   worst_method_return |
|:------------|------------------------:|-------------------------:|-----------------------:|------------------------:|---------------------:|----------------------:|----------------------:|----------------------:|
| C1_IV_X     |               0.111619  |                0.113244  |             -0.105758  |              -0.107788  |             1.32485  |              1.36265  |              1.32485  |             0.111619  |
| C1_EW_X     |               0.0974021 |                0.0955809 |             -0.109647  |              -0.110574  |             1.09784  |              1.118    |              1.09784  |             0.0955809 |
| C2_IV_X     |               0.063991  |                0.0775024 |             -0.0662521 |              -0.0662521 |             0.863414 |              1.01962  |              0.863414 |             0.063991  |
| C3_NMVT_M   |               0.0663012 |                0.0663012 |             -0.0970524 |              -0.0970524 |             0.820881 |              0.820881 |              0.820881 |             0.0663012 |
| C3_IV_M     |               0.0576159 |                0.0576159 |             -0.0852812 |              -0.0852812 |             0.745398 |              0.745398 |              0.745398 |             0.0576159 |
| C3_NMV_M    |               0.0561848 |                0.0561848 |             -0.0972689 |              -0.0972689 |             0.700224 |              0.700224 |              0.700224 |             0.0561848 |
| C3_EW_M     |               0.0541539 |                0.0541539 |             -0.0877434 |              -0.0877434 |             0.672697 |              0.672697 |              0.672697 |             0.0541539 |
| C2_NMVT_X   |               0.0485265 |                0.0516238 |             -0.0730078 |              -0.0727028 |             0.602903 |              0.630678 |              0.602903 |             0.0485265 |
| C3_IV_X     |               0.0451985 |                0.0430022 |             -0.0891847 |              -0.0891847 |             0.592311 |              0.564087 |              0.564087 |             0.0430022 |
| C2_NMV_X    |               0.0611129 |                0.0473732 |             -0.0819567 |              -0.0819567 |             0.751532 |              0.560737 |              0.560737 |             0.0473732 |
| C2_EW_X     |               0.0426109 |                0.0591321 |             -0.0743231 |              -0.0743231 |             0.551994 |              0.735584 |              0.551994 |             0.0426109 |
| C2_EW_M     |               0.0357448 |                0.0357448 |             -0.0897054 |              -0.0897054 |             0.461418 |              0.461418 |              0.461418 |             0.0357448 |
| C2_IV_M     |               0.0300401 |                0.0300401 |             -0.0904704 |              -0.0904704 |             0.406293 |              0.406293 |              0.406293 |             0.0300401 |
| C1_NMVT_X   |               0.0330264 |                0.0383237 |             -0.0998904 |              -0.0959883 |             0.358124 |              0.446976 |              0.358124 |             0.0330264 |
| C3_EW_X     |               0.0298268 |                0.0233148 |             -0.0984307 |              -0.0984307 |             0.372137 |              0.289947 |              0.289947 |             0.0233148 |

## Signal rank IC by forward fold

|   fold | test_start          | test_end            |   xgb_signal_rank_ic |   xgb_signal_pvalue |   momentum_signal_rank_ic |   momentum_signal_pvalue |   blend_signal_rank_ic |   blend_signal_pvalue |
|-------:|:--------------------|:--------------------|---------------------:|--------------------:|--------------------------:|-------------------------:|-----------------------:|----------------------:|
|     45 | 2026-01-02 00:00:00 | 2026-02-02 00:00:00 |             0.132767 |          0.202076   |                 0.020851  |              0.841891    |              0.149399  |            0.150669   |
|     46 | 2026-02-02 00:00:00 | 2026-03-02 00:00:00 |             0.190483 |          0.0576563  |                 0.179833  |              0.0733982   |              0.240361  |            0.0160041  |
|     47 | 2026-03-02 00:00:00 | 2026-04-02 00:00:00 |             0.249865 |          0.0121713  |                 0.0471973 |              0.641002    |              0.248     |            0.0128529  |
|     48 | 2026-04-02 00:00:00 | 2026-05-02 00:00:00 |             0.178248 |          0.0730667  |                -0.348489  |              0.000331451 |              0.0819574 |            0.41284    |
|     49 | 2026-05-02 00:00:00 | 2026-06-02 00:00:00 |             0.209512 |          0.0327988  |                 0.0133369 |              0.893108    |              0.186088  |            0.0585771  |
|     50 | 2026-06-02 00:00:00 | 2026-07-02 00:00:00 |             0.122341 |          0.216009   |                -0.097905  |              0.322786    |              0.0889205 |            0.36938    |
|     51 | 2026-07-02 00:00:00 | 2026-08-02 00:00:00 |             0.257486 |          0.00800809 |                 0.180482  |              0.0654145   |              0.288701  |            0.00281943 |

## Top configurations by method

| config_id   | family                    | method   |   candidate_size |   portfolio_cardinality |   weight_upper | weight_mode              |   cumulative_return |   sharpe_zero_rf |   maximum_drawdown |   mean_max_weight |   mean_effective_names |
|:------------|:--------------------------|:---------|-----------------:|------------------------:|---------------:|:-------------------------|--------------------:|-----------------:|-------------------:|------------------:|-----------------------:|
| C1_IV_X     | constraint_and_allocation | AUR      |                8 |                       4 |           0.3  | inverse_volatility       |           0.111619  |         1.32485  |         -0.105758  |          0.299146 |                3.87206 |
| C1_EW_X     | constraint_and_allocation | AUR      |                8 |                       4 |           0.3  | equal                    |           0.0974021 |         1.09784  |         -0.109647  |          0.25     |                4       |
| C2_IV_X     | constraint_and_allocation | AUR      |               10 |                       6 |           0.25 | inverse_volatility       |           0.063991  |         0.863414 |         -0.0662521 |          0.235656 |                5.53894 |
| C3_NMVT_M   | constraint_and_allocation | AUR      |               10 |                       8 |           0.15 | normalized_mean_variance |           0.0663012 |         0.820881 |         -0.0970524 |          0.15     |                7.09878 |
| C2_NMV_X    | constraint_and_allocation | AUR      |               10 |                       6 |           0.25 | normalized_mean_variance |           0.0611129 |         0.751532 |         -0.0819567 |          0.25     |                4.31566 |
| C3_IV_M     | constraint_and_allocation | AUR      |               10 |                       8 |           0.15 | inverse_volatility       |           0.0576159 |         0.745398 |         -0.0852812 |          0.149195 |                7.80307 |
| C3_NMV_M    | constraint_and_allocation | AUR      |               10 |                       8 |           0.15 | normalized_mean_variance |           0.0561848 |         0.700224 |         -0.0972689 |          0.15     |                7.05944 |
| C3_EW_M     | constraint_and_allocation | AUR      |               10 |                       8 |           0.15 | equal                    |           0.0541539 |         0.672697 |         -0.0877434 |          0.125    |                8       |
| C2_NMVT_X   | constraint_and_allocation | AUR      |               10 |                       6 |           0.25 | normalized_mean_variance |           0.0485265 |         0.602903 |         -0.0730078 |          0.25     |                4.57277 |
| C3_IV_X     | constraint_and_allocation | AUR      |               10 |                       8 |           0.15 | inverse_volatility       |           0.0451985 |         0.592311 |         -0.0891847 |          0.15     |                7.70399 |
| C1_IV_X     | constraint_and_allocation | QAUR     |                8 |                       4 |           0.3  | inverse_volatility       |           0.113244  |         1.36265  |         -0.107788  |          0.3      |                3.87764 |
| C1_EW_X     | constraint_and_allocation | QAUR     |                8 |                       4 |           0.3  | equal                    |           0.0955809 |         1.118    |         -0.110574  |          0.25     |                4       |
| C2_IV_X     | constraint_and_allocation | QAUR     |               10 |                       6 |           0.25 | inverse_volatility       |           0.0775024 |         1.01962  |         -0.0662521 |          0.232759 |                5.5913  |
| C3_NMVT_M   | constraint_and_allocation | QAUR     |               10 |                       8 |           0.15 | normalized_mean_variance |           0.0663012 |         0.820881 |         -0.0970524 |          0.15     |                7.09878 |
| C3_IV_M     | constraint_and_allocation | QAUR     |               10 |                       8 |           0.15 | inverse_volatility       |           0.0576159 |         0.745398 |         -0.0852812 |          0.149195 |                7.80307 |
| C2_EW_X     | constraint_and_allocation | QAUR     |               10 |                       6 |           0.25 | equal                    |           0.0591321 |         0.735584 |         -0.0743231 |          0.166667 |                6       |
| C3_NMV_M    | constraint_and_allocation | QAUR     |               10 |                       8 |           0.15 | normalized_mean_variance |           0.0561848 |         0.700224 |         -0.0972689 |          0.15     |                7.05944 |
| C3_EW_M     | constraint_and_allocation | QAUR     |               10 |                       8 |           0.15 | equal                    |           0.0541539 |         0.672697 |         -0.0877434 |          0.125    |                8       |
| C2_NMVT_X   | constraint_and_allocation | QAUR     |               10 |                       6 |           0.25 | normalized_mean_variance |           0.0516238 |         0.630678 |         -0.0727028 |          0.25     |                4.53857 |
| C3_IV_X     | constraint_and_allocation | QAUR     |               10 |                       8 |           0.15 | inverse_volatility       |           0.0430022 |         0.564087 |         -0.0891847 |          0.15     |                7.69083 |

The next strategy specification must be frozen before collecting a new, unseen
paper-trading period. These tables can diagnose and design that specification,
but cannot establish live readiness or quantum advantage.
