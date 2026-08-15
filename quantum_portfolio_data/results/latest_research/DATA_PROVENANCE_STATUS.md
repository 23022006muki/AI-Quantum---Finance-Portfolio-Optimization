# Data provenance status

- Price source checkpoints: FinanceDataReader/Yahoo primary with vnstock fallback.
- Price panel: real, 300 tickers, 467,164 rows, 2020-01-02 to 2025-12-31.
- Security master method: `first_price_observation_proxy` — rejected for research.
- Universe definition: all securities historically listed on HOSE (`hose_all_listed`), not
  an index-membership universe. VN30/VN-Index membership history is therefore not a core
  requirement for this specification.
- Historical universe snapshot: rebuilt from the first-price proxy; source trust gate fails.
- Corporate actions/adjusted-price policy: verified real contract not supplied; 47 large
  adjusted-return observations remain unresolved.
- Total-return market benchmark: verified VN-Index total-return series not supplied.
- Fundamentals, macro and foreign flow: stale fixture tables were moved recoverably to
  `outputs/quarantine/fixture_auxiliary/20260806T180922`; absent optional tables are now
  reported as limitations instead of contaminating the real-data run.

Raw market data and secrets are not committed to GitHub.
