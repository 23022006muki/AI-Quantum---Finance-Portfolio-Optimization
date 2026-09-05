# Prospective paper protocol v2 — common 30-session market gate

The methodology remains AUR versus QAUR with an identical downstream portfolio
pipeline. A common causal risk overlay is added after weight allocation:

\[
e_t=\mathbb{1}\left[\prod_{s=t-30}^{t-1}(1+r^{EW}_s)-1>0\right],
\qquad \tilde{w}_t=e_t w_t.
\]

The 30-session full-universe growth at the lock date is **-1.1352%**.
Therefore the September paper regime is **CASH**.

## Shadow and executable targets

| method   | ticker   |   shadow_weight |   risk_multiplier |   paper_target_weight | regime_state   |
|:---------|:---------|----------------:|------------------:|----------------------:|:---------------|
| AUR      | NAF      |        0.3      |                 0 |                     0 | CASH           |
| AUR      | VCB      |        0.24689  |                 0 |                     0 | CASH           |
| AUR      | VJC      |        0.242442 |                 0 |                     0 | CASH           |
| AUR      | STB      |        0.210668 |                 0 |                     0 | CASH           |
| QAUR     | NAF      |        0.3      |                 0 |                     0 | CASH           |
| QAUR     | VCB      |        0.24689  |                 0 |                     0 | CASH           |
| QAUR     | VJC      |        0.242442 |                 0 |                     0 | CASH           |
| QAUR     | STB      |        0.210668 |                 0 |                     0 | CASH           |

The shadow portfolio is retained to measure asset-selection quality even when the
common risk gate is off. Executable paper capital remains in cash until a future
monthly decision observes a positive trailing 30-session market return.
