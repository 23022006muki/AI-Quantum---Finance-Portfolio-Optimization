"""Materialize and execute the standalone notebook source in a temporary workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PREFIX = "/content/ai_quantum_standalone/"


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="ai-quantum-standalone-") as temporary:
        root = Path(temporary)
        facade_source = None
        postprocess_source = None
        display_sources: list[str] = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.startswith("%%writefile "):
                header, content = source.split("\n", 1)
                destination = header.removeprefix("%%writefile ").strip()
                if not destination.startswith(PREFIX):
                    continue
                target = root / destination.removeprefix(PREFIX)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            elif source.startswith("# Materialize the documented standalone module boundaries."):
                facade_source = source
            elif "aliases = {" in source and "research_report_vi.md" in source:
                postprocess_source = source
            elif source.startswith("display(Markdown(f\"### Experiment") or source.startswith(
                "for figure_name in ["
            ) or source.startswith("display(Markdown(\"# Kết luận kiểm định"):
                display_sources.append(source)
        if facade_source is None:
            raise RuntimeError("Facade source cell was not found.")
        (root / "ai_quantum_system").mkdir(parents=True, exist_ok=True)
        namespace = {"STANDALONE_ROOT": root}
        exec(compile(facade_source, "facade-cell", "exec"), namespace)
        workspace = root / "outputs" / "uploaded_research_data"
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        environment["PYTHONIOENCODING"] = "utf-8"
        run([
            sys.executable, "scripts/import_colab_complete_csv.py", str(args.csv.resolve()),
            "--workspace", str(workspace),
        ], cwd=root, env=environment)
        run([sys.executable, "-m", "pytest", "-q"], cwd=root, env=environment)
        expression = (
            "from pathlib import Path; from src.research import run_experiment; "
            f"print(run_experiment(Path(r'{workspace}'), Path(r'{root / 'configs/standalone_smoke.yaml'}')))"
        )
        run([sys.executable, "-c", expression], cwd=root, env=environment)
        experiments = workspace / "outputs" / "experiments"
        active = max((path for path in experiments.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
        manifest = json.loads((active / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "success", manifest
        assert manifest["folds_completed"] == 4, manifest
        assert (active / "solver_runs.csv").exists()
        assert (active / "statistical_tests.csv").exists()
        if postprocess_source is None:
            raise RuntimeError("Post-processing cell was not found.")
        normalized = workspace / "outputs" / "normalized"
        scope = {
            "ACTIVE": active, "WORKSPACE": workspace, "CSV_PATH": args.csv.resolve(),
            "CSV_SHA256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
            "EXECUTION_PROFILE": "SMOKE", "ENV_PYTHON": Path(sys.executable),
            "CONFIG_PATH": root / "configs/standalone_smoke.yaml",
            "PIPELINE_ENV": environment, "NOTEBOOK_CONFIG": {"significance_level": 0.05},
            "manifest": manifest, "prices": pd.read_parquet(normalized / "prices.parquet"),
            "benchmark": pd.read_parquet(normalized / "benchmark.parquet"),
            "master": pd.read_parquet(normalized / "security_master_full.parquet"),
            "actions": pd.read_parquet(normalized / "corporate_actions.parquet"),
            "quality": json.loads((workspace / "outputs/reports/data_quality.json").read_text(encoding="utf-8")),
            "leakage": json.loads((workspace / "outputs/reports/leakage_audit.json").read_text(encoding="utf-8")),
            "np": np, "pd": pd, "json": json, "os": os, "shutil": shutil,
            "sys": sys, "Path": Path,
        }
        exec(compile(postprocess_source, "postprocess-cell", "exec"), scope)
        scope.update({"display": lambda *args, **kwargs: None, "Markdown": str, "Image": lambda **kwargs: None})
        coverage_path = workspace / "outputs/reports/coverage_report.csv"
        scope["data_summary"] = pd.DataFrame()
        for source in display_sources:
            exec(compile(source, "display-cell", "exec"), scope)
        required = [
            "config_freeze.json", "environment.json", "dataset_hash.json", "data_quality_report.json", "folds.csv",
            "features_summary.csv", "adaptive_universe.csv", "qubo_instances.json",
            "research_report_vi.md", "latest_selected_portfolio.csv", "statistical_tests.csv",
        ]
        missing = [name for name in required if not (active / name).exists()]
        assert not missing, missing
        print(json.dumps({
            "status": manifest["status"], "folds_completed": manifest["folds_completed"],
            "experiment_id": manifest["experiment_id"],
            "artifacts": len([path for path in active.rglob("*") if path.is_file()]),
            "postprocess": "success", "required_aliases": len(required),
        }, indent=2))


if __name__ == "__main__":
    main()
