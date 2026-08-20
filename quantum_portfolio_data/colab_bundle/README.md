# Google Colab reproducibility bundle

This directory contains the compact, non-secret package required to reproduce and
inspect the published Data 17/8 experiment in Google Colab.

## Files

| Archive | SHA-256 | Contents |
|---|---|---|
| `data_17_8_runtime.zip` | `f69b1747e8fee8412a1cc089f8c7ef4c0be5aeed4e59a2fc6da7833dd1abf37c` | The validated 120-stock complete-case price panel, benchmark, security masters, corporate-action table, universe tables, data contracts, and audit reports required by `run-data-17-8`. |
| `data_17_8_published_experiment.zip` | `5d5faefa264f18647ab86501664013d593a8038b222637f477ad586f52aae5ec` | All 54 artifacts from experiment `20260820T160429-4f2cfc123d`, including features, fold outputs, solver traces, statistical tests, figures, reports, and manifests. |

The runtime archive expands directly into `outputs/Data 17_8/`. The experiment
archive expands into the published experiment directory selected by the Colab
notebook.

## Scope limitation

The local source-document archive is approximately 54.6 GB and is intentionally
not duplicated in this GitHub repository. It contains raw disclosure PDFs and
crawl caches, not additional observations used by the 120-stock price panel during
model fitting. Its coverage and unresolved-document status are preserved in the
runtime audit reports and in the published experiment provenance. Consequently,
the bundle reproduces the accepted exploratory model run, but it does not convert
the study into a confirmatory full-HOSE data package.

No cookie, API token, authenticated request capture, or other credential is present
in either archive.
