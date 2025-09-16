from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from .metrics import safe_sharpe


def yearly_table(rets: pd.Series) -> pd.DataFrame:
    """Per-year total, Sharpe, MDD, WinRate for calendar years."""
    r = rets.dropna();
    if r.empty: return pd.DataFrame(columns=["total","sharpe","mdd","win"])
    g = r.groupby(r.index.year)
    tbl = []
    for y, s in g:
        eq = (1.0 + s).cumprod(); mdd = float(abs((eq/eq.cummax()-1).min()))
        sharpe = float(s.mean() / (s.std(ddof=1)+1e-9) * np.sqrt(252)) if len(s)>1 else np.nan
        win = float((s>0).mean())
        tbl.append((y, float(eq.iloc[-1]-1), sharpe, mdd, win))
    return pd.DataFrame(tbl, columns=["year","total","sharpe","mdd","win"]).set_index("year")


def recent_windows(rets: pd.Series, months: Tuple[int, int] = (12,24)) -> Dict[str, float]:
    """Metrics for last N months windows; returns dict keyed by 'm12_*', 'm24_*'."""
    r = rets.dropna(); out: Dict[str,float] = {}
    for m in months:
        if len(r)==0: continue
        start = r.index.max() - pd.DateOffset(months=m)
        s = r[r.index >= start]
        if len(s)==0: continue
        eq=(1+s).cumprod(); ret=float(eq.iloc[-1]-1)
        sh=float(s.mean()/(s.std(ddof=1)+1e-9)*np.sqrt(252)) if len(s)>1 else np.nan
        out[f"m{m}_ret"]=ret; out[f"m{m}_sharpe"]=sh
    return out


def buy_on_the_cheap_guard(trades: pd.DataFrame, px: pd.Series) -> Dict[str, object]:
    """Warn if median entry-price percentile vs 3Y rolling distribution < 20%."""
    if trades is None or trades.empty: return {"median_pct": np.nan, "warn": False}
    ent = trades.get("entry_idx") if "entry_idx" in trades else trades.get("i_open")
    if ent is None: return {"median_pct": np.nan, "warn": False}
    ent = ent.astype(int)
    prices = px.astype(float)
    pcts = []
    for i in ent:
        if i not in range(len(prices)): continue
        t = prices.index[i]
        wnd = prices[(prices.index>=t-pd.DateOffset(years=3)) & (prices.index<t)]
        if len(wnd)<30: continue
        p = float((wnd < prices.iloc[i]).mean())*100.0
        pcts.append(p)
    med = float(np.median(pcts)) if pcts else np.nan
    return {"median_pct": med, "warn": bool(med<20.0) if not np.isnan(med) else False}


def recency_focus(rets: pd.Series, months: int = 12) -> Dict[str, float]:
    """Return mean, std, sharpe_valid, sharpe (safe_sharpe), MDD for the last N months."""
    if rets.empty: return {"mean": np.nan, "std": np.nan, "sharpe": None, "sharpe_valid": False, "mdd": np.nan}
    start = rets.index.max() - pd.DateOffset(months=months)
    s = rets[rets.index >= start].dropna()
    if s.empty: return {"mean": np.nan, "std": np.nan, "sharpe": None, "sharpe_valid": False, "mdd": np.nan}
    mean = float(s.mean()); std = float(s.std(ddof=1))
    sh, meta = safe_sharpe(s)
    eq = (1.0 + s).cumprod(); peak = eq.cummax(); mdd = float(abs((eq/peak-1).min()))
    return {"mean": mean, "std": std, "sharpe": (None if not meta['valid'] else float(sh)), "sharpe_valid": bool(meta['valid']), "mdd": mdd}
