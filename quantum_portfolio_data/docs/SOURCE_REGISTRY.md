# Source registry

| Adapter | Class | Status | Notes |
|---|---|---|---|
| `fixture` | synthetic fixture | Implemented | Deterministic; **NOT RESEARCH RESULT** |
| `csv` | user-authorized real import | Implemented | Requires explicit local file and source metadata |
| SSI FastConnect official | real | Adapter implemented; credentials required | Documented official HOSE securities/OHLC/index APIs |
| Historical VN30 | real | PIT importer implemented | Requires supplied reliable history with effective dates |
| Corporate actions | real | PIT importer implemented | Requires announcement/effective timestamps |
| Financial statements | real | PIT importer implemented | Requires publication timestamps |
| FRED official | real | Adapter implemented; API key required | Release-aware international macro series |
| NSO/SBV | real | PIT CSV/Parquet import implemented | Requires release/availability timestamps |

The project does not bypass authentication, paywalls, CAPTCHAs, robots.txt or terms.
