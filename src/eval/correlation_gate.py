from __future__ import annotations

from typing import Dict, Optional
import os
import pandas as pd


def correlation_gate(candidate: pd.Series, pool: Optional[pd.DataFrame] = None,
                     thr: float = 0.5) -> Dict[str, object]:
    """Return {'max_corr': float, 'violations': list[str]} vs pool.

    If pool is None, attempt to load reports/pool_alpha_daily_returns.csv.
    """
    if pool is None:
        path = os.path.join("reports", "pool_alpha_daily_returns.csv")
        if os.path.exists(path):
            pool = pd.read_csv(path, index_col=0, parse_dates=True)
    if pool is None or pool.empty:
        return {"max_corr": float("nan"), "violations": []}
    cand = candidate.rename("new").dropna()
    joined = pool.join(cand, how="inner")
    if len(joined) == 0:
        return {"max_corr": float("nan"), "violations": []}
    c = joined.corr()["new"].drop("new", errors="ignore").dropna()
    mx = float(c.abs().max()) if len(c) else float("nan")
    viol = [name for name, v in c.items() if abs(v) >= thr]
    return {"max_corr": mx, "violations": viol}

