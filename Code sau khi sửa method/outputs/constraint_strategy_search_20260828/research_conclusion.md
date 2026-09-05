# Kết luận strategy search có temporal holdout

## Thiết kế

- Dữ liệu: `data_sau_khi_sua_method.csv`; SHA-256 `16fde8609090d0bdf342908d8539a901fc3aec1d66bc51aa452ba33dead03ee3`.
- 44 walk-forward folds; development folds 0–28, untouched holdout folds 29–43.
- Đã sàng lọc 29 cấu hình. Cấu hình được chọn bằng Sharpe trung bình của AUR và QAUR trên development, không nhìn holdout.
- Grid dùng exact feasible-subspace reference; cấu hình thắng được audit lại bằng shared fixed-Hamming-weight XY-QAOA statevector trên holdout.

## Cấu hình đề xuất

`C1_NMVT_M`

```json
{
  "config_id": "C1_NMVT_M",
  "family": "constraint_and_allocation",
  "candidate_size": 8,
  "portfolio_cardinality": 4,
  "weight_upper": 0.3,
  "weight_lower": 0.05,
  "weight_mode": "normalized_mean_variance",
  "signal_blend": 0.7,
  "correlation_penalty": 0.1,
  "stability_weight": 0.15,
  "covariance_span": 60,
  "covariance_shrinkage": 0.2,
  "risk_aversion_qubo": 0.55,
  "risk_aversion_weights": 2.0,
  "turnover_penalty": 0.15,
  "volatility_target": 0.0,
  "transaction_cost_bps": 25.0
}
```

## Kết quả holdout

| config_id   | sample   | method   |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |   selection_score |
|:------------|:---------|:---------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|------------------:|
| C1_NMVT_M   | holdout  | AUR      |            312 |            0.236224 |            0.186824 |                0.149652 |          1.24839 |           1.78326 |          -0.137666 |         -0.257262 |
| C1_NMVT_M   | holdout  | QAUR     |            312 |            0.236224 |            0.186824 |                0.149652 |          1.24839 |           1.78326 |          -0.137666 |         -0.257262 |

## Baseline holdout

| method           |   observations |   cumulative_return |   annualized_return |   annualized_volatility |   sharpe_zero_rf |   sortino_zero_rf |   maximum_drawdown |
|:-----------------|---------------:|--------------------:|--------------------:|------------------------:|-----------------:|------------------:|-------------------:|
| FULL_UNIVERSE_EW |            312 |            0.244105 |            0.192931 |                0.139445 |          1.38356 |           1.3899  |          -0.174571 |
| VNALLSHARE_TRI   |            312 |            0.400115 |            0.312367 |                0.199313 |          1.56722 |           1.74739 |          -0.183418 |

## Giả thuyết

| hypothesis                              |     estimate |   statistic |   pvalue_one_sided | supported_5pct   |
|:----------------------------------------|-------------:|------------:|-------------------:|:-----------------|
| H1_QAUR_higher_QUR_objective            |  0           |   nan       |         nan        | False            |
| H2_QAUR_lower_candidate_correlation     |  0           |   nan       |         nan        | False            |
| H3_QAUR_turnover_noninferior_margin_2pp |  0           |  -inf       |           0        | True             |
| H4_QAUR_higher_mean_daily_return        | -1.68985e-13 |    -1.91063 |           0.971514 | False            |
| H5_direction_robust_across_QAUR_seeds   |  0           |   nan       |         nan        | False            |

Có 1/5 giả thuyết đạt tiêu chí đã định trước. H5 là robustness direction check, không phải kiểm định quantum advantage.

## Diễn giải hợp lệ

Kết quả dương trên development không được xem là bằng chứng nếu không lặp lại trên temporal holdout. QAUR vẫn là classical surrogate cho quantum-ready QUBO; XY-QAOA là ideal statevector simulation. Không có tuyên bố quantum advantage.
