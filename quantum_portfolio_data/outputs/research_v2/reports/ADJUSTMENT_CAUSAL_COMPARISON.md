# Adjustment causal comparison

## Scope

This report holds Data A securities, target weights and rebalance dates fixed. It changes only the return definition. It is therefore a direct return-only counterfactual, not a full-pipeline causal result.

Price-adjustment gate: **blocked**. The research total-return column is labelled a candidate while the gate is blocked.

## Frozen-holdings results

| Return definition | Gross cumulative return | Net cumulative return |
|---|---:|---:|
| raw_close | -17.9056% | -21.7701% |
| research_total_return_candidate | -13.5498% | -17.6193% |
| source_adjusted | -16.7752% | -20.6929% |

## Interpretation boundary

Indirect effects on features, labels, covariance, AUR, QUBO, solver selections and weights require three independently rerun full pipelines. Those runs are not permitted until the total-return dataset and benchmark pass their confirmatory gates. No claim that corporate actions caused the baseline loss is made from this table.
