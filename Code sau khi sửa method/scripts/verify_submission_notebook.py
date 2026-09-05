from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab" / "AUR_QAUR_RESEARCH_SUBMISSION_FULL_CODE.ipynb"
DATASET = ROOT / "data 29_8" / "data_29_8.csv"
RESULTS = ROOT / "outputs" / "submission_local_verification_20260831"


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
namespace = {
    "DATASET": DATASET,
    "RESULTS": RESULTS,
    "digest": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
}

# Cell 6 audits the local data. Cells 13 onward are the entire result,
# explanation, visualization, and fail-fast audit layer. Installation, embedded
# data recovery, %%writefile, and the 20-minute engine run were verified
# separately and are intentionally not repeated here.
for index in [6, *range(13, 37)]:
    cell = notebook["cells"][index]
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    if isinstance(source, list):
        source = "".join(source)
    print(f"Executing notebook cell {index}")
    exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
    if index == 6:
        namespace["display"] = lambda *args, **kwargs: None
        namespace["Markdown"] = lambda value: value

print("SUBMISSION_NOTEBOOK_LOCAL_VERIFICATION_OK")
