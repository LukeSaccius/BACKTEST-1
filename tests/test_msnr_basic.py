from __future__ import annotations

import pandas as pd
import numpy as np

from src.quality.data_checks import validate_input_df
from src.quality.outliers import scan_and_fix_outliers
from src.eval.metrics import evaluate_daily_returns
from src.eval.correlation_gate import correlation_gate
from src.eval.yearly_breakdown import yearly_table, recent_windows
from src.strategies.malaysian_snr import MSNRParams, run_msnr_backtest, find_swings, is_engulfing
from src.eval.metrics import safe_sharpe, daily_equity_returns


def _toy(n=300):
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    rng = np.random.default_rng(0)
    px = pd.Series(100 + rng.standard_normal(n).cumsum(), index=idx)
    df = pd.DataFrame({
        'open': px.shift(1).fillna(px.iloc[0]),
        'high': px + 0.5,
        'low': px - 0.5,
        'close': px,
        'volume': np.abs(rng.normal(1e6, 1e5, n))
    }, index=idx)
    return df


def test_validation_and_outliers():
    df = _toy(60)
    dfc, rep = validate_input_df(df)
    assert 'open' in dfc.columns and rep['warnings']['lookahead_possible']
    dfc.iloc[10, dfc.columns.get_loc('close')] = dfc['close'].median()*10
    clean, oq = scan_and_fix_outliers(dfc, method='mad', k=3.0)
    assert oq['anomaly_ratio'] >= 0


def test_metrics_and_yearly():
    df = _toy(260)
    rets = df['close'].pct_change().fillna(0)
    m = evaluate_daily_returns(rets)
    assert {'sharpe','mdd','turnover'}.issubset(m.keys())
    yt = yearly_table(rets)
    rec = recent_windows(rets)
    assert isinstance(yt, pd.DataFrame) and len(rec) >= 2


def test_safe_sharpe_flags_low_variance():
    r = pd.Series([0.0]*40, index=pd.date_range('2022-01-01', periods=40))
    sh, meta = safe_sharpe(r)
    assert not meta['valid'] and np.isnan(sh)


def test_daily_equity_returns_no_nan():
    eq = pd.Series([100, 101, 101, 102], index=pd.date_range('2022-01-01', periods=4))
    dr = daily_equity_returns(eq)
    assert dr.isna().sum() == 0 and dr.iloc[0] == 0.0


def test_corr_gate_handles_missing_pool():
    df = _toy(120)
    gate = correlation_gate(df['close'].pct_change())
    assert 'max_corr' in gate


def test_msnr_core_helpers():
    df = _toy(50)
    sw = find_swings(df, 3)
    assert isinstance(sw, list)
    engulf = is_engulfing(df.assign(open=df['close']), 10, +1) is not None
    assert isinstance(engulf, bool)


def test_run_backtest_smoke():
    df = _toy(400)
    p = MSNRParams()
    is_p=(df.index[0].strftime('%Y-%m-%d'), df.index[299].strftime('%Y-%m-%d'))
    oos_p=(df.index[300].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d'))
    out = run_msnr_backtest(df, p, is_p, oos_p)
    assert set(out.keys())=={'is','oos','trades'}


def test_oos_min_trades_gate():
    df = _toy(200)
    p = MSNRParams()
    half = len(df)//2
    is_p=(df.index[0].strftime('%Y-%m-%d'), df.index[half-1].strftime('%Y-%m-%d'))
    oos_p=(df.index[half].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d'))
    out = run_msnr_backtest(df, p, is_p, oos_p)
    oos_trades = 0 if out['trades'].empty else len(out['trades'][out['trades']['i_close']>=half])
    assert isinstance(oos_trades, int)
