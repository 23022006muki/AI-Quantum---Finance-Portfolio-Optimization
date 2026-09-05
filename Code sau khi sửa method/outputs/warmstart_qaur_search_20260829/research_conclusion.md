# Kết luận strategy search có temporal holdout

## Thiết kế

- Dữ liệu: `data_sau_khi_sua_method.csv`; SHA-256 `16fde8609090d0bdf342908d8539a901fc3aec1d66bc51aa452ba33dead03ee3`.
- 44 walk-forward folds; development folds 0–28, untouched holdout folds 29–43.
- Đã sàng lọc 6 cấu hình. Cấu hình được chọn bằng Sharpe trung bình của AUR và QAUR trên development, không nhìn holdout.
- Grid dùng exact feasible-subspace reference; cấu hình thắng được audit lại bằng shared fixed-Hamming-weight XY-QAOA statevector trên holdout.

## Cấu hình đề xuất

`W_K10P6_CP30`

```json
{
  "config_id": "W_K10P6_CP30",
  "family": "warm_started_qaur",
  "candidate_size": 10,
  "portfolio_cardinality": 6,
  "weight_upper": 0.25,
  "weight_lower": 0.02,
  "weight_mode": "normalized_mean_variance",
  "signal_blend": 0.7,
  "correlation_penalty": 0.3,
  "stability_weight": 0.0,
  "covariance_span": 60,
  "covariance_shrinkage": 0.2,
  "risk_aversion_qubo": 0.55,
  "risk_aversion_weights": 2.0,
  "turnover_penalty": 0.15,
  "volatility_target": 0.0,
  "transaction_cost_bps": 25.0,
  "qa_warm_start": true,
  "market_regime_lookback": 0,
  "minimum_validation_ic": -1.0
}
```

## Kết quả holdout

| config_id    | sample   | method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |   selection_score |
|:-------------|:---------|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|------------------:|
| W_K10P6_CP30 | holdout  | AUR      |            312 |            0.248126 |            0.196044 |                0.142733 |          1.3735  |           1.76114 |          -0.124444 |         0.0156843 |
| W_K10P6_CP30 | holdout  | QAUR     |            312 |            0.235096 |            0.185949 |                0.138955 |          1.33819 |           1.73438 |          -0.128228 |         0.0156843 |

## Baseline holdout

| method           |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:-----------------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| FULL_UNIVERSE_EW |            312 |            0.244105 |            0.192931 |                0.139445 |          1.38356 |           1.3899  |          -0.174571 |
| VNALLSHARE_TRI   |            312 |            0.400115 |            0.312367 |                0.199313 |          1.56722 |           1.74739 |          -0.183418 |

## Giả thuyết

| hypothesis                              |     estimate |   statistic |   pvalue_one_sided | supported_5pct   |
|:----------------------------------------|-------------:|------------:|-------------------:|:-----------------|
| H1_QAUR_higher_QUR_objective            |  0.0602777   |    5.08878  |        8.25358e-05 | True             |
| H2_QAUR_lower_candidate_correlation     | -0.0123637   |   -6.90641  |        3.62481e-06 | True             |
| H3_QAUR_turnover_noninferior_margin_2pp | -0.06        |   -2.3864   |        0.0158414   | True             |
| H4_QAUR_higher_mean_daily_return        | -3.57576e-05 |   -0.170742 |        0.567731    | False            |
| H5_direction_robust_across_QAUR_seeds   |  0           |  nan        |      nan           | False            |

Có 3/5 giả thuyết đạt tiêu chí đã định trước. H5 là robustness direction check, không phải kiểm định quantum advantage.

## Diễn giải hợp lệ

Kết quả dương trên development không được xem là bằng chứng nếu không lặp lại trên temporal holdout. QAUR vẫn là classical surrogate cho quantum-ready QUBO; XY-QAOA là ideal statevector simulation. Không có tuyên bố quantum advantage.
