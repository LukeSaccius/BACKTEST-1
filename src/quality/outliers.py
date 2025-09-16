from __future__ import annotations

from typing import Dict, Tuple, Iterable
import numpy as np
import pandas as pd


def _mad(x: pd.Series) -> float:
    med = x.median()
    return float(np.median(np.abs(x - med))) or 1e-9


def _hampel(s: pd.Series, w: int, k: float) -> Tuple[pd.Series, int]:
    x = s.copy()
    flag = 0
    med = x.rolling(w, center=True).median()
    mad = x.rolling(w, center=True).apply(lambda z: _mad(pd.Series(z)), raw=False)
    thr = k * 1.4826 * mad
    mask = (x - med).abs() > thr
    flag = int(mask.sum())
    x[mask] = med[mask]
    return x, flag


def _global_mad(s: pd.Series, k: float) -> Tuple[pd.Series, int]:
    med, mad = s.median(), _mad(s)
    z = 1.4826 * (s - med) / mad
    mask = z.abs() > k
    x = s.copy(); x[mask] = med
    return x, int(mask.sum())


def _winsor_roll(s: pd.Series, w: int, q: float = 0.01) -> Tuple[pd.Series, int]:
    x = s.copy(); flag = 0
    roll_low = x.rolling(w, center=True).quantile(q)
    roll_hi = x.rolling(w, center=True).quantile(1 - q)
    lo_mask = x < roll_low; hi_mask = x > roll_hi
    flag = int(lo_mask.sum() + hi_mask.sum())
    x[lo_mask] = roll_low[lo_mask]; x[hi_mask] = roll_hi[hi_mask]
    return x, flag


def scan_and_fix_outliers(df: pd.DataFrame, method: str = "hampel", window: int = 20,
                          k: float = 3.0, clean: bool = True,
                          cols: Iterable[str] = ("open","high","low","close","volume")) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return (clean_df, report) with per-column counts and overall ratio."""
    df2 = df.copy(); counts: Dict[str, int] = {}
    for c in cols:
        if c not in df2.columns: continue
        s = df2[c].astype(float)
        if method == "hampel":
            sc, n = _hampel(s, window, k)
        elif method == "mad":
            sc, n = _global_mad(s, k)
        else:
            sc, n = _winsor_roll(s, window)
        counts[c] = int(n)
        if clean: df2[c] = sc
    total = int(sum(counts.values())); ratio = total / max(1, len(df2))
    return df2, {"method": method, "counts": counts, "anomaly_ratio": ratio}

