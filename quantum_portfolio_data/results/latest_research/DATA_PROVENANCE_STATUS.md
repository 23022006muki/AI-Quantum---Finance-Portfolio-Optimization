# Data provenance status

- Price source checkpoints: FinanceDataReader/Yahoo primary with vnstock fallback.
- Price panel: real, 300 tickers, 2020-01-02 to 2025-12-31.
- Security master method: `first_price_observation_proxy` — rejected for research.
- Historical universe snapshot: remaining fixture-derived snapshot — rejected for research.
- Historical membership events: verified real table not supplied.
- Corporate actions and adjusted-price policy: verified real contract not supplied.
- Fundamentals, macro and foreign flow: fixture tables from demo remain isolated and are
  rejected in real research mode.

Raw market data and secrets are not committed to GitHub.
