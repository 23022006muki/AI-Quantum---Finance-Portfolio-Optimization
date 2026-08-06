# Source registry

| Adapter | Class | Status | Notes |
|---|---|---|---|
| `fixture` | synthetic fixture | Implemented | Deterministic; **NOT RESEARCH RESULT** |
| `csv` | user-authorized real import | Implemented | Requires explicit local file and source metadata |
| FinanceDataReader/Yahoo | real market prices | Implemented | Price panel only; current listings and first price observations do **not** establish historical universe membership |
| vnstock | real market prices | Implemented | Checkpoint/fallback price adapter; source terms and availability metadata retained |
| Vietstock | authenticated real source | Implemented with user-supplied session | Does not bypass authentication; historical universe/corporate actions remain separate contracts |
| HOSE official listing service | official public exchange data | Collected for 2015–2025 | 404 current equities and 96 delisted equities; ISIN identity and exchange event dates only |
| Trading Economics | official authenticated API | Implemented as OHLC cross-check | Historical endpoint lacks volume and does not certify HOSE listing history, corporate actions, adjusted prices or total-return semantics |
| SSI FastConnect official | real | Adapter implemented; credentials required | Documented official HOSE securities/OHLC/index APIs |
| Historical VN30 | real | PIT importer implemented | Requires supplied reliable history with effective dates |
| Corporate actions | real | PIT importer implemented | Requires announcement/effective timestamps |
| Financial statements | real | PIT importer implemented | Requires publication timestamps |
| FRED official | real | Adapter implemented; API key required | Release-aware international macro series |
| NSO/SBV | real | PIT CSV/Parquet import implemented | Requires release/availability timestamps |

The project does not bypass authentication, paywalls, CAPTCHAs, robots.txt or terms.
