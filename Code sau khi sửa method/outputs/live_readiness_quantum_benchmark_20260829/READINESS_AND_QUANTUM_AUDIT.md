# Live-capital readiness and quantum-advantage audit

## Outcome

- Live-capital gates passed: **4/9**.
- Quantum-advantage gates passed: **0/7**.
- Config audited: `W_K10P6_CP30`.
- Dataset ends at 2025-12-31; this is not a current live recommendation.

## Live-capital readiness gates

| gate                                                       | passed   | evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:-----------------------------------------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| positive_holdout_return_both_methods                       | True     | {'AUR': 0.24812569348228752, 'QAUR': 0.23509557001243464}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| holdout_sharpe_at_least_1_both_methods                     | True     | {'AUR': 1.3735020667133202, 'QAUR': 1.3381896934560211}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| holdout_max_drawdown_no_worse_than_minus_20pct             | True     | {'AUR': -0.12444393373487939, 'QAUR': -0.12822776073773023}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| positive_sharpe_at_75bps                                   | True     | [{'method': 'AUR', 'sharpe_zero_rf': 1.1347037723855122}, {'method': 'QAUR', 'sharpe_zero_rf': 1.0834637589517089}]                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| significant_excess_return_vs_full_EW                       | False    | [{'method': 'AUR', 'comparator': 'FULL_UNIVERSE_EW', 'observations': 312, 'mean_daily_difference': 1.1888406604930734e-05, 'annualized_arithmetic_difference': 0.0029958784644425447, 'ci_2_5': -0.20646383530215687, 'ci_97_5': 0.20391761230298233, 'pvalue_one_sided_positive': 0.5138}, {'method': 'QAUR', 'comparator': 'FULL_UNIVERSE_EW', 'observations': 312, 'mean_daily_difference': -2.3869163150833868e-05, 'annualized_arithmetic_difference': -0.0060150291140101345, 'ci_2_5': -0.22525532351211505, 'ci_97_5': 0.1809286486030434, 'pvalue_one_sided_positive': 0.5772}] |
| significant_excess_return_vs_VNAllshare                    | False    | [{'method': 'AUR', 'comparator': 'VNALLSHARE_TRI', 'observations': 312, 'mean_daily_difference': -0.0004074655013331395, 'annualized_arithmetic_difference': -0.10268130633595114, 'ci_2_5': -0.3587094040378448, 'ci_97_5': 0.12571599819520798, 'pvalue_one_sided_positive': 0.7912}, {'method': 'QAUR', 'comparator': 'VNALLSHARE_TRI', 'observations': 312, 'mean_daily_difference': -0.000443223071088904, 'annualized_arithmetic_difference': -0.1116922139144038, 'ci_2_5': -0.3929663443294058, 'ci_97_5': 0.11140847313591232, 'pvalue_one_sided_positive': 0.8294}]            |
| deflated_sharpe_probability_at_least_95pct                 | False    | [{'method': 'AUR', 'deflated_sharpe_probability': 0.9139855722512756}, {'method': 'QAUR', 'deflated_sharpe_probability': 0.9086317954566215}]                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| at_least_24_months_untouched_forward_or_paper_track_record | False    | Current temporal holdout is about 15 months and is now observed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| market_data_current_with_operational_live_pipeline         | False    | Dataset ends 2025-12-31; no audited 2026 paper-trading feed/order pipeline.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

## Transaction-cost stress

|   cost_bps | sample   | method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|-----------:|:---------|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
|          0 | holdout  | AUR      |            312 |            0.270476 |            0.213313 |                0.142831 |         1.49346  |           1.91537 |          -0.123926 |
|          0 | holdout  | QAUR     |            312 |            0.258209 |            0.203842 |                0.139022 |         1.46625  |           1.89971 |          -0.127214 |
|         25 | holdout  | AUR      |            312 |            0.248126 |            0.196044 |                0.142733 |         1.3735   |           1.76114 |          -0.124444 |
|         25 | holdout  | QAUR     |            312 |            0.235096 |            0.185949 |                0.138955 |         1.33819  |           1.73438 |          -0.128228 |
|         50 | holdout  | AUR      |            312 |            0.22614  |            0.178998 |                0.142763 |         1.25381  |           1.60855 |          -0.124962 |
|         50 | holdout  | QAUR     |            312 |            0.212376 |            0.168297 |                0.139035 |         1.21047  |           1.5697  |          -0.129242 |
|         75 | holdout  | AUR      |            312 |            0.204513 |            0.162173 |                0.142921 |         1.1347   |           1.4588  |          -0.125481 |
|         75 | holdout  | QAUR     |            312 |            0.190042 |            0.150883 |                0.13926  |         1.08346  |           1.40582 |          -0.130256 |
|        100 | holdout  | AUR      |            312 |            0.18324  |            0.145567 |                0.143207 |         1.01647  |           1.3065  |          -0.125999 |
|        100 | holdout  | QAUR     |            312 |            0.16809  |            0.133705 |                0.139631 |         0.957562 |           1.24259 |          -0.13127  |

## Deflated Sharpe after 43 tested configurations

| method   |   observed_annual_sharpe_arithmetic |   expected_max_annual_sharpe_under_trials |   deflated_sharpe_probability |   number_of_trials |   skewness |   pearson_kurtosis |
|:---------|------------------------------------:|------------------------------------------:|------------------------------:|-------------------:|-----------:|-------------------:|
| AUR      |                             1.32574 |                                 0.0872853 |                      0.913986 |                 43 |   0.132567 |            15.8679 |
| QAUR     |                             1.29694 |                                 0.0872853 |                      0.908632 |                 43 |   0.124729 |            17.4422 |

## Paired moving-block bootstrap versus baselines

| method   | comparator       |   observations |   mean_daily_difference |   annualized_arithmetic_difference |    ci_2_5 |   ci_97_5 |   pvalue_one_sided_positive |
|:---------|:-----------------|---------------:|------------------------:|-----------------------------------:|----------:|----------:|----------------------------:|
| AUR      | FULL_UNIVERSE_EW |            312 |             1.18884e-05 |                         0.00299588 | -0.206464 |  0.203918 |                      0.5138 |
| AUR      | VNALLSHARE_TRI   |            312 |            -0.000407466 |                        -0.102681   | -0.358709 |  0.125716 |                      0.7912 |
| QAUR     | FULL_UNIVERSE_EW |            312 |            -2.38692e-05 |                        -0.00601503 | -0.225255 |  0.180929 |                      0.5772 |
| QAUR     | VNALLSHARE_TRI   |            312 |            -0.000443223 |                        -0.111692   | -0.392966 |  0.111408 |                      0.8294 |

## Prequential configuration selection

| method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| AUR      |            667 |            0.57597  |            0.187506 |                0.120153 |          1.56055 |           2.10041 |          -0.131403 |
| QAUR     |            667 |            0.760968 |            0.238362 |                0.119232 |          1.99915 |           2.81861 |          -0.131724 |

## Prequential lookback sensitivity

|   lookback_folds | sample         | method   |   configuration_changes |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|-----------------:|:---------------|:---------|------------------------:|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
|                6 | available_path | AUR      |                      16 |            791 |            0.295072 |           0.0858632 |                0.149909 |         0.572767 |          0.691851 |          -0.3196   |
|                6 | available_path | QAUR     |                      16 |            791 |            0.544764 |           0.148599  |                0.138863 |         1.07011  |          1.44081  |          -0.268238 |
|                6 | common_holdout | AUR      |                      16 |            312 |            0.239387 |           0.189276  |                0.154812 |         1.22261  |          1.58789  |          -0.127984 |
|                6 | common_holdout | QAUR     |                      16 |            312 |            0.360886 |           0.282587  |                0.155288 |         1.81975  |          2.55669  |          -0.131316 |
|                9 | available_path | AUR      |                       8 |            726 |            0.742416 |           0.212566  |                0.124255 |         1.71073  |          2.3316   |          -0.124444 |
|                9 | available_path | QAUR     |                       8 |            726 |            0.975112 |           0.266491  |                0.122298 |         2.17903  |          3.21644  |          -0.128228 |
|                9 | common_holdout | AUR      |                       8 |            312 |            0.208949 |           0.165629  |                0.15156  |         1.09283  |          1.43635  |          -0.124444 |
|                9 | common_holdout | QAUR     |                       8 |            312 |            0.331277 |           0.26      |                0.148471 |         1.75119  |          2.45919  |          -0.128228 |
|               12 | available_path | AUR      |                       6 |            667 |            0.57597  |           0.187506  |                0.120153 |         1.56055  |          2.10041  |          -0.131403 |
|               12 | available_path | QAUR     |                       6 |            667 |            0.760968 |           0.238362  |                0.119232 |         1.99915  |          2.81861  |          -0.131724 |
|               12 | common_holdout | AUR      |                       6 |            312 |            0.176636 |           0.1404    |                0.145794 |         0.963005 |          1.28492  |          -0.131403 |
|               12 | common_holdout | QAUR     |                       6 |            312 |            0.251232 |           0.198447  |                0.142482 |         1.39279  |          1.91034  |          -0.131724 |
|               18 | available_path | AUR      |                       4 |            542 |            0.358448 |           0.153075  |                0.127531 |         1.2003   |          1.61127  |          -0.120136 |
|               18 | available_path | QAUR     |                       4 |            542 |            0.447555 |           0.187644  |                0.123481 |         1.51962  |          2.09695  |          -0.120461 |
|               18 | common_holdout | AUR      |                       4 |            312 |            0.181501 |           0.144207  |                0.150695 |         0.956947 |          1.30569  |          -0.120136 |
|               18 | common_holdout | QAUR     |                       4 |            312 |            0.244893 |           0.193541  |                0.144594 |         1.33852  |          1.84139  |          -0.120461 |
|               24 | available_path | AUR      |                       3 |            417 |            0.330673 |           0.188443  |                0.137603 |         1.36947  |          1.85807  |          -0.119843 |
|               24 | available_path | QAUR     |                       3 |            417 |            0.389678 |           0.220016  |                0.137718 |         1.59758  |          2.1808   |          -0.120609 |
|               24 | common_holdout | AUR      |                       3 |            312 |            0.242903 |           0.192     |                0.15252  |         1.25886  |          1.69577  |          -0.119843 |
|               24 | common_holdout | QAUR     |                       3 |            312 |            0.2864   |           0.225581  |                0.152309 |         1.48108  |          2.00796  |          -0.120609 |

## Multi-lookback prequential ensemble

| sample                | method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:----------------------|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| common_available_path | AUR      |            417 |            0.327436 |            0.186695 |                0.133231 |          1.40128 |           1.81392 |          -0.123637 |
| common_available_path | QAUR     |            417 |            0.426433 |            0.239415 |                0.128955 |          1.85658 |           2.50431 |          -0.125316 |
| common_holdout        | AUR      |            312 |            0.2105   |            0.166837 |                0.146859 |          1.13604 |           1.46788 |          -0.123637 |
| common_holdout        | QAUR     |            312 |            0.29587  |            0.232864 |                0.141284 |          1.6482  |           2.20341 |          -0.125316 |

| method   | comparator       |   observations |   mean_daily_difference |   annualized_arithmetic_difference |    ci_2_5 |   ci_97_5 |   pvalue_one_sided_positive |
|:---------|:-----------------|---------------:|------------------------:|-----------------------------------:|----------:|----------:|----------------------------:|
| AUR      | FULL_UNIVERSE_EW |            312 |            -8.39355e-05 |                         -0.0211518 | -0.239592 | 0.201809  |                      0.6126 |
| AUR      | VNALLSHARE_TRI   |            312 |            -0.000503289 |                         -0.126829  | -0.380343 | 0.0988498 |                      0.867  |
| QAUR     | FULL_UNIVERSE_EW |            312 |             0.000131462 |                          0.0331284 | -0.190151 | 0.209393  |                      0.4714 |
| QAUR     | VNALLSHARE_TRI   |            312 |            -0.000287892 |                         -0.0725488 | -0.349128 | 0.141281  |                      0.779  |

| method   |   observed_annual_sharpe_arithmetic |   expected_max_annual_sharpe_under_trials |   deflated_sharpe_probability |   number_of_trials |   skewness |   pearson_kurtosis |
|:---------|------------------------------------:|------------------------------------------:|------------------------------:|-------------------:|-----------:|-------------------:|
| AUR      |                             1.12407 |                                 0.0896365 |                      0.87434  |                 44 |   0.188829 |            14.4781 |
| QAUR     |                             1.55262 |                                 0.0896365 |                      0.946799 |                 44 |   0.215622 |            15.3604 |

## Quantum-advantage gates

| gate                                            | passed   | evidence                                                                                                                       |
|:------------------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------|
| executed_on_physical_QPU                        | False    | All XY-QAOA results are ideal statevector simulations.                                                                         |
| matched_best_classical_baselines                | False    | Exact enumeration and simulated annealing included; commercial/state-of-the-art solvers and tuned HPC baselines still missing. |
| better_solution_quality_than_best_classical     | False    | Exact classical gap is zero by definition; mean statevector observed gap=0.                                                    |
| lower_end_to_end_wall_clock_than_best_classical | False    | Mean XY-QAOA simulator runtime=0.152054s; mean exact/heuristic runtime=0.085816s.                                              |
| scaling_crossover_demonstrated                  | False    | Statevector tested only through n=12; no QPU scaling crossover.                                                                |
| hardware_noise_and_repeated_run_statistics      | False    | No hardware calibration, queue, shot-noise or error-mitigation study.                                                          |
| independent_reproducibility                     | False    | No independent QPU reproduction.                                                                                               |

## Recommendation

The current evidence supports continued research and a shadow/paper portfolio, not unrestricted live capital. A staged pilot can only be considered after a fresh forward period, current data and operational controls. The present simulator cannot support a quantum-advantage statement. A valid claim requires physical-QPU runs, matched wall-clock accounting and statistically superior quality/time scaling against the strongest tuned classical solvers.
