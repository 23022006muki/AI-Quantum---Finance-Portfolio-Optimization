# Frozen prospective paper-trading protocol

This lock starts with the first available session on or after
**2026-09-02**. Parameters, source policy and
evaluation gates cannot be changed during the paper window.

## September target portfolios

| config_id   |   fold | method   | ticker   | selected_downstream   |   weight |
|:------------|-------:|:---------|:---------|:----------------------|---------:|
| C1_IV_X     |     53 | AUR      | NAF      | True                  | 0.3      |
| C1_IV_X     |     53 | AUR      | VCB      | True                  | 0.24689  |
| C1_IV_X     |     53 | AUR      | VJC      | True                  | 0.242442 |
| C1_IV_X     |     53 | AUR      | STB      | True                  | 0.210668 |
| C1_IV_X     |     53 | QAUR     | NAF      | True                  | 0.3      |
| C1_IV_X     |     53 | QAUR     | VCB      | True                  | 0.24689  |
| C1_IV_X     |     53 | QAUR     | VJC      | True                  | 0.242442 |
| C1_IV_X     |     53 | QAUR     | STB      | True                  | 0.210668 |

- AUR/QAUR target Jaccard: 1.0000
- Declared transaction cost: 25 bps per turnover
- Maximum single-name weight: 30%
- Status: paper only; no real-capital or quantum-advantage authorization

The JSON lock contains file hashes, operational rules and promotion gates. Any
code or data-policy change creates a new protocol version and restarts the
prospective evidence clock for the changed strategy.
