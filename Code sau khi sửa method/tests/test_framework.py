from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reduction import adaptive_reduce, quantum_assisted_reduce
from src.optimization import portfolio_qubo, xy_qaoa_select


def fixture():
    names = list("ABCDEFGH")
    snap = pd.DataFrame({"ticker": names, "signal": np.arange(8), "liquidity_20d": np.arange(8) + 1, "volatility_20d": np.arange(8, 0, -1)})
    history = pd.DataFrame(np.random.default_rng(1).normal(size=(80, 8)), columns=names)
    cfg = {"signal": .4, "liquidity": .3, "risk": .15, "stability": .15, "correlation_penalty": .1}
    return snap, history, cfg


def test_reducers_share_contract_and_cardinality():
    snap, history, cfg = fixture()
    a = adaptive_reduce(snap, history, set(), 4, cfg)
    q = quantum_assisted_reduce(snap, history, set(), 4, cfg, 1, 3, 20)
    assert len(a.tickers) == len(q.tickers) == 4
    assert a.method == "AUR" and q.method == "QAUR"


def test_shared_xy_solver_is_feasible():
    q = portfolio_qubo(np.array([.1, .2, .05, .12]), np.eye(4) * .02, .5)
    solved = xy_qaoa_select(q, 2, 1, 2, 5, 64)
    assert solved["bits"].sum() == 2
    assert solved["feasibility_rate"] == 1.0
