# Price adjustment and total-return methodology

The research-v2 dataset keeps source prices immutable. Raw OHLC, the provider's adjusted close and the research total-return series are separate fields with separate provenance. A vendor-adjusted price is not considered verified merely because it removes a large return.

Corporate actions are identified by stable security identity and event date. VSDC or another official disclosure supplies announcement time, record date and material event terms. CafeF or Vietstock may corroborate the ex-right date, but is not relabeled as an official source. Conflicts and incomplete events remain unresolved and block confirmatory use.

For one share held immediately before an event, the one-period total return compares prior raw close with terminal wealth after the event. Cash dividends add cash. Stock dividends and bonus shares increase the number of shares. A split multiplies shares, while a reverse split reduces them according to its own ratio. A rights issue increases shares only if the subscription cash outflow is also deducted. Consequently, rights issues are never treated as stock splits.

When several verified events share an ex-date, cash amounts and per-old-share entitlements are aggregated, split terms are multiplied and rights subscription outflows are included. Merger or conversion events fail closed unless the consideration and successor security identity are verified. Full-precision returns are retained; board-lot and whole-share rounding belong to the execution layer.

Point-in-time eligibility requires the official event information to have been available no later than the ex-date. A later correction creates a new versioned dataset and does not overwrite an earlier experiment. Every output is bound to the input price hash, ledger hash and contract hash.

The confirmatory gate fails when a material event is unresolved, a material vendor-adjustment change is unmatched, or a documented total-return benchmark is unavailable. Blocked outputs are retained as diagnostic artifacts and must not be described as research results.
