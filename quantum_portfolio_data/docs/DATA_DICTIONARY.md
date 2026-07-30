# Data dictionary

## Prices

Primary key: `(date, ticker)`.

Required fields: `open`, `high`, `low`, `close`, `adjusted_close`, `volume`,
`trading_value`, `source`, `source_url`, `fetched_at`, `available_at`, `raw_checksum`,
`parser_version`, `data_class`.

`data_class` is `real` only for a user-authorized real source. Deterministic smoke-test
data are always marked `fixture`.

## Security master

Fields include `listing_date`, `delisting_date`, `effective_from`, `effective_to` and
`available_at`. These fields are required to reconstruct a historical universe.

## Corporate actions

Keep `event_date`, `announcement_date`, `effective_date` and `available_at` separate.
Never adjust prices twice. An empty fixture table does not validate a real adjustment
method.

## Historical index membership

A real VN30 table must contain `ticker`, `effective_from`, `effective_to`,
`announcement_date`, `available_at` and provenance fields.

## Financial statements

A real table must contain both fiscal period end and public release/availability date.
Daily joins must be as-of joins on `available_at`, never fiscal period end.

