from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


@dataclass
class MSNRParams:
    swing_lookback: int = 3
    min_touches: int = 3
    use_bos: bool = True
    require_engulfing: bool = True
    use_body_tl: bool = True
    atr_period: int = 14
    atr_sl_mult: float = 2.0
    tp_rr: List[float] = (1.0, 2.0, 3.0)
    outlier_method: str = "hampel"
    outlier_window: int = 20
    outlier_k: float = 3.0
    commission_bp: float = 1.0
    slippage_bp: float = 1.0
    htf: str = "D"
    max_time_in_market: float = 0.90
    min_time_in_market: float = 0.05


# --------- Pure helpers (past‑only: use shift in callers) ---------
def find_swings(df: pd.DataFrame, lookback: int) -> List[Tuple[int, int]]:
    """Return list of (idx, dir) where dir=+1 pivot high, -1 pivot low."""
    h = df["high"].values; l = df["low"].values
    swings: List[Tuple[int, int]] = []
    for i in range(lookback, len(df) - lookback):
        if h[i] == np.max(h[i - lookback : i + lookback + 1]):
            swings.append((i, +1))
        if l[i] == np.min(l[i - lookback : i + lookback + 1]):
            swings.append((i, -1))
    swings.sort(key=lambda x: x[0])
    return swings


def outer_trendline(swings: List[Tuple[int, int]], use_body_tl: bool,
                    df: pd.DataFrame) -> Tuple[float, float, int] | None:
    """Return (slope, intercept, side). side=+1 up TL (lows) else -1 down TL (highs)."""
    if not swings:
        return None
    up = [(i, df["close" if use_body_tl else "low"].iloc[i]) for i, d in swings if d == -1]
    dn = [(i, df["close" if use_body_tl else "high"].iloc[i]) for i, d in swings if d == +1]
    def line(two: List[Tuple[int, float]], side: int):
        if len(two) < 2: return None
        (x1, y1), (x2, y2) = two[-2:]
        m = (y2 - y1) / max(1e-9, (x2 - x1)); b = y1 - m * x1
        return (m, b, side)
    return line(up, +1) or line(dn, -1)


def third_touch_event(df: pd.DataFrame, tl: Tuple[float, float, int], tol: float) -> bool:
    """True if latest close is within tol of TL (third touch approximation)."""
    m, b, _ = tl; i = len(df) - 1; y = m * i + b
    return abs(df["close"].iloc[i] - y) <= tol


def is_engulfing(df: pd.DataFrame, i: int, direction: int) -> bool:
    """Body engulfing: abs(body_i) > abs(body_{i-1}) and covers it."""
    if i < 1: return False
    o, c = df["open"].iloc, df["close"].iloc
    b0, b1 = abs(c[i] - o[i]), abs(c[i - 1] - o[i - 1])
    if b0 <= b1: return False
    bull_cover = (c[i] > o[i - 1]) and (o[i] < c[i - 1])
    bear_cover = (c[i] < o[i - 1]) and (o[i] > c[i - 1])
    return bull_cover if direction > 0 else bear_cover


def bos_update(last_dir: int, swings: List[Tuple[int, int]], df: pd.DataFrame) -> int:
    """Simple BOS: if last pivot high breaks prior high -> +1, low breaks -> -1 else 0."""
    highs = [i for i, d in swings if d == +1]; lows = [i for i, d in swings if d == -1]
    if len(highs) >= 2 and df["high"].iloc[highs[-1]] > df["high"].iloc[highs[-2]]:
        return +1
    if len(lows) >= 2 and df["low"].iloc[lows[-1]] < df["low"].iloc[lows[-2]]:
        return -1
    return last_dir


def htf_storyline(df: pd.DataFrame, freq: str) -> int:
    """HTF direction via simple slope of 20SMA on resampled closes."""
    if freq not in {"W", "D", "H4"}: return 0
    rule = {"W": "W", "D": "D", "H4": "4H"}[freq]
    cs = df["close"].resample(rule).last().dropna()
    if len(cs) < 25: return 0
    sma = cs.rolling(20).mean()
    slope = (sma.iloc[-1] - sma.iloc[-5])
    return 1 if slope > 0 else -1 if slope < 0 else 0


# --------- Minimal past‑only backtest (1RU risk, full exit on first TP) ---------
def run_msnr_backtest(df: pd.DataFrame, params: MSNRParams,
                      is_period: Tuple[str, str], oos_period: Tuple[str, str]) -> Dict[str, object]:
    """Return dict with daily returns for IS/OOS and filled trades table."""
    df = df.copy()
    df = df.sort_index()
    atr = (df["high"].shift(1) - df["low"].shift(1)).rolling(params.atr_period).mean().fillna(method="bfill")
    htf_dir = htf_storyline(df, params.htf)
    eq, pos, trades = 1.0, None, []  # equity, open position, trade log
    rets = pd.Series(0.0, index=df.index)
    swings = []  # updated using past bars only
    for i in range(max(2*params.swing_lookback+params.atr_period+2, 20), len(df)):
        past = df.iloc[: i]  # excludes current bar
        swings = find_swings(past, params.swing_lookback)
        tl = outer_trendline(swings, params.use_body_tl, past)
        tol = atr.iloc[i - 1] * 0.25
        # manage open
        if pos:
            c = df["close"].iloc[i]
            if pos["dir"] > 0:
                hit_tp = c >= pos["entry"] + pos["ru"] * max(params.tp_rr)
                hit_sl = c <= pos["entry"] - pos["ru"]
            else:
                hit_tp = c <= pos["entry"] - pos["ru"] * max(params.tp_rr)
                hit_sl = c >= pos["entry"] + pos["ru"]
            if hit_tp or hit_sl:
                pnl = (c - pos["entry"]) * pos["dir"]
                net = pnl - pos["cost"]
                eq *= (1.0 + net / max(1e-9, pos["entry"]))
                rets.iloc[i] = net / max(1e-9, pos["entry"])  # normalized daily
                trades.append({"entry": pos["entry"], "exit": c, "dir": pos["dir"], "i_open": pos["i"], "i_close": i})
                pos = None
        # consider entries (past‑only)
        if tl and not pos:
            m, b, side = tl; i0 = i - 1; y = m * i0 + b
            near = third_touch_event(past, tl, tol)
            bos = bos_update(0, swings, past)
            engulf = is_engulfing(past, i0, side)
            align = (htf_dir == 0) or (htf_dir == side)
            price = df["open"].iloc[i]
            ru = params.atr_sl_mult * atr.iloc[i - 1]
            cost = (params.commission_bp + params.slippage_bp) * 1e-4 * price
            # Setup A: third touch + storyline + engulf (optional)
            if near and align and (not params.require_engulfing or engulf):
                pos = {"dir": side, "entry": price, "ru": ru, "i": i, "cost": cost}
                continue
            # Setup B: break & retest (close above/below TL then pullback near TL with engulf)
            broke = (past["close"].iloc[-1] - y) * side > 0
            retest = abs(past["close"].iloc[-1] - y) <= tol
            if broke and retest and (not params.use_bos or bos == side) and engulf:
                pos = {"dir": side, "entry": price, "ru": ru, "i": i, "cost": cost}
    is_rets = rets.loc[is_period[0] : is_period[1]].copy()
    oos_rets = rets.loc[oos_period[0] : oos_period[1]].copy()
    return {"is": is_rets, "oos": oos_rets, "trades": pd.DataFrame(trades)}

