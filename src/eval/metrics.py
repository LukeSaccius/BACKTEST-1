from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def daily_equity_returns(equity: pd.Series) -> pd.Series:
    """Daily percent returns from equity; first day 0.0, no NaN."""
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def safe_sharpe(daily_rets: pd.Series, trading_days: int = 252) -> Tuple[float, dict]:
    """Sharpe with guards for variance/sample size."""
    x = daily_rets.values
    nonzero = int(np.count_nonzero(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if std < 1e-6 or nonzero < 30:
        return (float('nan'), {'valid': False, 'reason': 'low-variance-or-sample', 'std': std, 'nonzero_days': nonzero})
    mean = float(np.mean(x)); sharpe = mean / std * np.sqrt(trading_days)
    return (sharpe, {'valid': True, 'reason': '', 'std': std, 'nonzero_days': nonzero})


def _equity(rets: pd.Series) -> pd.Series:
    return (1.0 + rets.fillna(0.0)).cumprod().rename('equity')


def _mdd(eq: pd.Series) -> float:
    peak = eq.cummax(); dd = (eq/peak - 1.0).min(); return float(abs(dd))


def evaluate_daily_returns(rets: pd.Series) -> Dict[str, float]:
    """Compact metrics and reliability flags on daily returns."""
    r = rets.dropna(); n = len(r)
    if n == 0:
        return {'total_return': np.nan,'cagr': np.nan,'mean': np.nan,'vol': np.nan,'sharpe': None,
                'sortino': np.nan,'win_rate': np.nan,'profit_factor': np.nan,'mdd': np.nan,'tim': np.nan,
                'turnover': np.nan,'fitness': np.nan,'n_days': 0,'n_nonzero_days': 0,'std': np.nan,
                'sharpe_valid': False,'sharpe_reason': 'empty','valid': False}
    eq = _equity(r); total_ret = float(eq.iloc[-1]-1.0); years = max(1e-9, n/252.0)
    cagr = float(eq.iloc[-1]**(1.0/years)-1.0)
    mean = float(r.mean()); vol = float(r.std(ddof=1))
    sh, meta = safe_sharpe(r)
    neg = r[r<0]; pos = r[r>0]
    sortino = float(mean/(neg.std(ddof=1)+1e-9)*np.sqrt(252))
    win_rate = float(len(pos)/max(1,len(pos)+len(neg)))
    pf = float(pos.sum()/abs(neg.sum())) if neg.sum()!=0 else (np.inf if pos.sum()>0 else np.nan)
    mdd = _mdd(eq); tim = float((r!=0).mean()); turnover = float(r.abs().mean())
    fitness = float((sh if meta['valid'] else 0.0)/max(1e-9,mdd))
    return {"total_return": total_ret, "cagr": cagr, "mean": mean, "vol": vol,
            "sharpe": (None if not meta['valid'] else float(sh)), "sortino": sortino, "win_rate": win_rate,
            "profit_factor": pf, "mdd": mdd, "tim": tim, "turnover": turnover, "fitness": fitness,
            "n_days": n, "n_nonzero_days": int(meta.get('nonzero_days',0)), "std": float(meta.get('std',np.nan)),
            "sharpe_valid": bool(meta['valid']), "sharpe_reason": str(meta['reason']), "valid": bool(meta['valid'])}


def compare_is_oos(is_m: Dict[str, float], oos_m: Dict[str, float]) -> Dict[str, float]:
    """OOS/IS ratios and similarity flag (>=0.90 for sharpe and mean)."""
    def ratio(a,b):
        try: return float(a)/float(b) if (a is not None and b not in (0,None)) else np.nan
        except Exception: return np.nan
    sr = ratio(oos_m.get('sharpe'), is_m.get('sharpe'))
    mr = ratio(oos_m.get('mean'), is_m.get('mean'))
    return {"sharpe_ratio": sr, "mean_ratio": mr, "similarity_ge_0_90": bool((0.9<=sr<=1.1) and (0.9<=mr<=1.1))}


def turnover_from_trades(trades: pd.DataFrame, notional_col: str = 'notional') -> float:
    """Turnover = sum(|notional|)/average_equity if available, else NaN."""
    if trades is None or trades.empty or notional_col not in trades.columns:
        return float('nan')
    notional = trades[notional_col].abs().sum(); avg_eq = float(trades.get('equity', pd.Series([1.0])).mean())
    return float(notional / max(1e-9, avg_eq))

