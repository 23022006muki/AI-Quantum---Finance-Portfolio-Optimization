from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
experiments = sorted((ROOT / "outputs" / "experiments").glob("*"), reverse=True)
st.set_page_config(page_title="AI–Quantum Portfolio Demo", layout="wide")
st.title("AI–Quantum Portfolio Research Demo")
st.warning("NOT RESEARCH RESULT when the selected experiment uses fixture data.")
if not experiments:
    st.error("No experiment artifacts found. Run: python -m src.cli run-experiment --config configs/quick.yaml")
    st.stop()
selected = st.sidebar.selectbox("Experiment", experiments, format_func=lambda p: p.name)
manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
st.sidebar.json(manifest)
if manifest.get("status") == "blocked":
    st.error("Research run was blocked before training/backtesting. No research metrics were produced.")
    st.write("Blockers:", manifest.get("blockers", []))
    if (selected / "RESEARCH_BLOCKED.md").exists():
        st.markdown((selected / "RESEARCH_BLOCKED.md").read_text(encoding="utf-8"))
    st.download_button(
        "Download blocker manifest", (selected / "manifest.json").read_bytes(),
        file_name="manifest.json",
    )
    st.stop()
tab_names = ["Overview", "Data Quality", "Ranking & Reduction", "Solver Comparison",
             "Portfolio & Backtest", "Ablation & Robustness", "Reproducibility"]
tabs = st.tabs(tab_names)
with tabs[0]:
    st.subheader("Research pipeline")
    st.code("Point-in-time data → XGBoost/EWMA → adaptive reduction → QUBO → XY-QAOA/Dicke → weights → walk-forward")
    st.json(manifest)
with tabs[1]:
    st.json(json.loads((selected / "data_quality.json").read_text()))
    st.json(json.loads((selected / "leakage_audit.json").read_text()))
with tabs[2]:
    rank = pd.read_csv(selected / "rankings.csv")
    selection = pd.read_csv(selected / "selected_universe.csv")
    st.dataframe(rank, width="stretch")
    st.dataframe(selection, width="stretch")
with tabs[3]:
    comparison = pd.read_csv(selected / "comparisons.csv")
    st.dataframe(comparison, width="stretch")
    if not comparison.empty and comparison["optimality_gap_mean"].abs().max() > 0:
        st.bar_chart(comparison.set_index("method")["optimality_gap_mean"])
    else:
        st.info("All observed quick-demo optimality gaps are zero; table shown without a degenerate chart.")
with tabs[4]:
    metrics = pd.read_csv(selected / "metrics_long.csv")
    returns = pd.read_csv(selected / "portfolio_returns.csv", parse_dates=["date"])
    st.dataframe(metrics, width="stretch")
    if not returns.empty:
        curve = returns.pivot_table(index="date", columns="strategy", values="return", aggfunc="mean")
        st.line_chart((1 + curve).cumprod())
    st.dataframe(pd.read_csv(selected / "weights.csv"), width="stretch")
    st.subheader("Trading constraints and realized costs")
    st.caption(
        "Long-only, fully invested, configured weight bounds; weights drift buy-and-hold "
        "between rebalances. The same turnover-based cost policy applies to all strategies."
    )
    st.dataframe(pd.read_csv(selected / "cost_ledger.csv"), width="stretch")
    regime_path = selected / "regime_metrics.csv"
    if regime_path.exists():
        st.subheader("Trailing-information market regimes")
        st.dataframe(pd.read_csv(regime_path), width="stretch")
with tabs[5]:
    st.info("Fixture-mode inference is descriptive and remains NOT RESEARCH RESULT.")
    st.subheader("Eight ablation configurations")
    st.dataframe(pd.read_csv(selected / "ablation_results.csv"), width="stretch")
    st.subheader("Sensitivity grid")
    st.dataframe(pd.read_csv(selected / "sensitivity_results.csv"), width="stretch")
    st.subheader("Paired block bootstrap and Holm correction")
    st.dataframe(pd.read_csv(selected / "statistical_tests.csv"), width="stretch")
with tabs[6]:
    st.code((selected / "resolved_config.yaml").read_text())
    st.download_button("Download research report", (selected / "RESEARCH_REPORT.md").read_bytes(),
                       file_name="RESEARCH_REPORT.md")
    st.download_button("Download manifest", (selected / "manifest.json").read_bytes(),
                       file_name="manifest.json")
