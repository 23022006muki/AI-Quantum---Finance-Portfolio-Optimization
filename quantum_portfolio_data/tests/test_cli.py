from pathlib import Path

import src.cli as cli


def test_fixture_run_full_never_mutates_real_normalized_workspace(
    tmp_path: Path, monkeypatch,
):
    normalized = tmp_path / "outputs" / "normalized"
    normalized.mkdir(parents=True)
    sentinel = normalized / "prices.parquet"
    sentinel.write_bytes(b"REAL-PANEL-SENTINEL")
    config = tmp_path / "demo.yaml"
    config.write_text(
        "mode: demo_fixture\n"
        "label: NOT RESEARCH RESULT\n"
        "seed: 7\n"
        "data:\n"
        "  source: fixture\n"
        "  start: '2022-01-01'\n"
        "  end: '2022-12-31'\n"
        "  tickers: [AAA, BBB]\n"
        "  rebalance: monthly\n"
        "universe: {definition: hose_all_listed}\n",
        encoding="utf-8",
    )

    def fake_run(project_root: Path, _config: Path) -> Path:
        artifact = project_root / "outputs" / "experiments" / "isolated-demo"
        artifact.mkdir(parents=True)
        (artifact / "marker.txt").write_text("fixture only", encoding="utf-8")
        return artifact

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "run_experiment", fake_run)
    monkeypatch.setattr(cli, "print_experiment_summary", lambda _out: None)
    assert cli.main(["run-full", "--config", str(config)]) == 0
    assert sentinel.read_bytes() == b"REAL-PANEL-SENTINEL"
    assert (tmp_path / "outputs" / "experiments" / "isolated-demo" / "marker.txt").exists()
