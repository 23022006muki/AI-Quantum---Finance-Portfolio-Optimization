# Complete single-CSV Colab dataset

`ai_quantum_complete_dataset.csv` is the complete exploratory runtime input for the
Data 17/8 AI–Quantum pipeline. It is a UTF-8 CSV with a `record_type` discriminator,
allowing one file to preserve all normalized tables required by the model.

| Record type | Rows | Purpose |
|---|---:|---|
| `PRICE` | 157,826 | OHLCV, adjusted close, trading value and row-level provenance for 120 complete-case stocks |
| `BENCHMARK` | 1,499 | Official VNAllShare total-return index observations |
| `SECURITY` | 394 | Full HOSE security master; `runtime_eligible=true` identifies the 120-stock research panel |
| `CORPORATE_ACTION` | 2,653 | Cross-referenced corporate-action ledger |
| `METADATA` | 1 | Dataset version, intended config and declared research scope |

- CSV size: 67,569,804 bytes
- CSV SHA-256: `aea9644cfafc359ed04546deca62fea83509826864463669b219f370f1433eba`
- ZIP SHA-256: `8acbae27fa0a65d652142c15bf19e858660bc1deea743835c63554211c5d429f`

The ZIP contains the same CSV and is provided only for faster upload. The Colab
notebook accepts either file. `scripts/import_colab_complete_csv.py` validates the
schema and hashes, restores Parquet runtime tables, builds the monthly universe and
runs the leakage audit before model execution.

This dataset is complete for the published exploratory model runtime. It does not
embed the 54.6 GB raw disclosure/PDF archive or optional PIT financial-statement
features and therefore must not be represented as a confirmatory full-HOSE package.
