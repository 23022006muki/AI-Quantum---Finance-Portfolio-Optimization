import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
base_path = root / "colab" / "AUR_QAUR_Standalone_Full_Web_Colab_30_8.ipynb"
result_path = root / "colab" / "AUR_QAUR_RESULTS_INCLUDED_FULL_WEB_31_8.ipynb"

base_cells = json.loads(base_path.read_text(encoding="utf-8"))["cells"]
result_cells = json.loads(result_path.read_text(encoding="utf-8"))["cells"]

matched = 0
positions = []
for result_index, result_cell in enumerate(result_cells):
    if matched >= len(base_cells):
        break
    base_cell = base_cells[matched]
    same_type = result_cell.get("cell_type") == base_cell.get("cell_type")
    same_source = result_cell.get("source", []) == base_cell.get("source", [])
    if same_type and same_source:
        positions.append(result_index)
        matched += 1

print(
    {
        "base_cells": len(base_cells),
        "result_cells": len(result_cells),
        "matched_exactly": matched,
        "exact_ordered_subsequence": matched == len(base_cells),
        "first_match": positions[0] if positions else None,
        "last_match": positions[-1] if positions else None,
    }
)
