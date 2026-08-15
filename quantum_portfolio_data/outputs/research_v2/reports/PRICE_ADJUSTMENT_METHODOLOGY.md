# Price adjustment methodology v2

Generated: 2026-08-15T06:04:16.872337+00:00

The implementation preserves raw OHLC and source-adjusted close. Research total return is constructed only from cross-source verified events using the versioned contract in `docs/contracts/price_adjustment_contract.v2.json`.

For one share held before an event, terminal wealth equals current raw close multiplied by the post-event share count, plus cash dividend, less the subscription cash paid for exercised rights. Stock dividends, bonus shares, splits, reverse splits and rights therefore have distinct terms; rights are not treated as a split.

Gate status: **blocked**. Verified events applied: 1698. Unresolved material events: 956. Unmatched source-adjustment changes: 1531.

A blocked result is a diagnostic artifact only and cannot be promoted into research mode.
