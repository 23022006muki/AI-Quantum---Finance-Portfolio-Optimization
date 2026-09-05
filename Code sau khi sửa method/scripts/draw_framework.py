from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 13))
ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")

def box(x, y, w, h, text, color):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color, edgecolor="#334155", linewidth=1.2)
    ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9, fontweight="bold", wrap=True)

def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4))

box(2.5, 11.7, 5, .8, "Initial point-in-time stock universe", "#e2e8f0")
box(2.5, 10.4, 5, .8, "Shared XGBoost return forecasts + EWMA risk estimates", "#dbeafe")
arrow(5,11.7,5,11.2)
box(.6, 8.7, 3.7, 1.0, "Adaptive Universe Reduction\nquality + liquidity + risk + stability + correlation", "#dcfce7")
box(5.7, 8.7, 3.7, 1.0, "Quantum-Assisted Universe Reduction\ncardinality QUBO Q^UR", "#dbeafe")
arrow(5,10.4,2.45,9.7); arrow(5,10.4,7.55,9.7)
box(.9, 7.5, 3.1, .7, "Top-K_A", "#f0fdf4"); box(6, 7.5, 3.1, .7, "Top-K_QA", "#eff6ff")
arrow(2.45,8.7,2.45,8.2); arrow(7.55,8.7,7.55,8.2)
box(2.5, 6.2, 5, .8, "Direct reduction comparison\nquality, redundancy, stability, turnover, runtime", "#ede9fe")
arrow(2.45,7.5,4.3,7.0); arrow(7.55,7.5,5.7,7.0)
box(2.5, 4.9, 5, .8, "Shared cardinality-constrained portfolio QUBO Q^PO", "#fef3c7")
box(2.5, 3.6, 5, .8, "Shared feasible-subspace XY-QAOA", "#fef3c7")
box(2.5, 2.3, 5, .8, "Shared classical weight optimization", "#fef3c7")
box(2.5, 1.0, 5, .8, "Shared walk-forward backtest and statistical evaluation", "#ffedd5")
arrow(5,6.2,5,5.7); arrow(5,4.9,5,4.4); arrow(5,3.6,5,3.1); arrow(5,2.3,5,1.8)
ax.text(9.65, 10.8, "Forecast", rotation=90, color="#2563eb", ha="center", va="center", fontweight="bold")
ax.text(9.65, 8.1, "Universe reduction", rotation=90, color="#15803d", ha="center", va="center", fontweight="bold")
ax.text(9.65, 4.0, "Shared optimization", rotation=90, color="#a16207", ha="center", va="center", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "framework_method_moi.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "framework_method_moi.svg", bbox_inches="tight")
