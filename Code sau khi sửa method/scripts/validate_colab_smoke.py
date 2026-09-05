"""Execute the standalone notebook core locally with the bundled CSV.

The upload/install/download cells are replaced by local equivalents. This is a
smoke validator, not a second implementation of the research pipeline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab" / "AUR_QAUR_XYQAOA_Standalone_Full_Colab.ipynb"
DATASET = ROOT / "data sau khi sửa method" / "data_sau_khi_sua_method.csv"
RUNTIME = ROOT / "tmp" / "colab_smoke"


def source(cell: dict) -> str:
    return "".join(cell["source"])


def without_colab_magics(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("!"))


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__colab_smoke__"}
    code_cells = [
        (index, source(cell))
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]
    if not code_cells:
        raise AssertionError("Notebook contains no code cells.")
    for index, text in code_cells:
        compile(without_colab_magics(text), f"cell-{index}", "exec")

    setup_index, setup = code_cells[0]
    exec(compile(without_colab_magics(setup), f"cell-{setup_index}", "exec"), namespace)
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    (RUNTIME / "results").mkdir(parents=True)
    namespace["EXECUTION_PROFILE"] = "SMOKE"
    namespace["WORKDIR"] = RUNTIME
    namespace["RESULTS"] = RUNTIME / "results"
    namespace["csv_path"] = DATASET
    namespace["DATASET_SHA256"] = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    namespace["plt"].show = lambda *args, **kwargs: None
    # Upload is replaced by the bundled dataset. Every other code cell, including
    # figures and export, is executed; the Colab /content archive path is redirected.
    for index, text in code_cells[1:]:
        first_line = text.strip().splitlines()[0]
        if first_line == "from google.colab import files":
            continue
        if first_line == "artifacts = {":
            text = text.replace(
                'Path("/content/AUR_QAUR_XYQAOA_RESULTS.zip")',
                'RUNTIME / "AUR_QAUR_XYQAOA_RESULTS.zip"',
            )
            namespace["RUNTIME"] = RUNTIME
        exec(compile(text, f"cell-{index}", "exec"), namespace)
    summary = namespace["performance_summary"]
    required = {"AUR", "QAUR", "FULL_UNIVERSE_EW"}
    if not required.issubset(set(summary["method"])):
        raise AssertionError(f"Missing methods: {required - set(summary['method'])}")
    if namespace["solver_diagnostics"]["feasibility_rate"].min() < 1.0:
        raise AssertionError("Ideal XY-QAOA smoke run produced an infeasible sample.")
    if not namespace["audit_results"]["passed"].all():
        raise AssertionError("Research audit contains a failed check.")
    if not namespace.get("interpretation_text", "").strip():
        raise AssertionError("Automatic result interpretation was not generated.")
    if not (RUNTIME / "AUR_QAUR_XYQAOA_RESULTS.zip").exists():
        raise AssertionError("Result ZIP was not generated.")
    markdown = "\n".join(
        source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    for stage in range(1, 7):
        if f"Giai đoạn {stage}" not in markdown:
            raise AssertionError(f"Missing methodology explanation for stage {stage}.")
    print("COLAB_SMOKE_OK")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
