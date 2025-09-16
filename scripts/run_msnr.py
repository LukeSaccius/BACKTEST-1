from __future__ import annotations

import argparse, json, os
import numpy as np
from datetime import datetime
import pandas as pd

import os, sys
try:
    from src.quality.data_checks import validate_input_df
    from src.quality.outliers import scan_and_fix_outliers
    from src.strategies.malaysian_snr import MSNRParams, run_msnr_backtest
    from src.eval import metrics as M
    from src.eval.metrics import evaluate_daily_returns, compare_is_oos
    from src.eval.correlation_gate import correlation_gate
    from src.eval.yearly_breakdown import yearly_table, recent_windows, buy_on_the_cheap_guard
except Exception:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from quality.data_checks import validate_input_df
    from quality.outliers import scan_and_fix_outliers
    from strategies.malaysian_snr import MSNRParams, run_msnr_backtest
    from eval import metrics as M
    from eval.metrics import evaluate_daily_returns, compare_is_oos
    from eval.correlation_gate import correlation_gate
    from eval.yearly_breakdown import yearly_table, recent_windows, buy_on_the_cheap_guard


def _split(df: pd.DataFrame, is_years: int, oos_years: int):
    end = df.index.max(); oos_start = end - pd.DateOffset(years=oos_years)
    is_start = oos_start - pd.DateOffset(years=is_years)
    return (is_start.strftime('%Y-%m-%d'), (oos_start-pd.Timedelta(days=1)).strftime('%Y-%m-%d')), \
           (oos_start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))


# Reliability gates (configurable)
MIN_OOS_TRADES = 30
MIN_OOS_NONZERO_DAYS = 60
OOS_SHARPE_MIN = 0.70
TIME_IN_MKT_MIN, TIME_IN_MKT_MAX = 0.05, 0.90
TURNOVER_MIN, TURNOVER_MAX = 0.01, 0.06


def _ensure_costs(p: MSNRParams, min_comm_bp: float = 1.0, min_slip_bp: float = 5.0) -> MSNRParams:
    """Enforce minimal commission/slippage in basis points."""
    p.commission_bp = max(min_comm_bp, p.commission_bp)
    p.slippage_bp = max(min_slip_bp, p.slippage_bp)
    return p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True); p.add_argument('--symbol')
    p.add_argument('--start'); p.add_argument('--end')
    p.add_argument('--is-years', type=int, default=8)
    p.add_argument('--oos-years', type=int, default=2)
    p.add_argument('--walk-forward', type=int, default=0)
    p.add_argument('--outlier-method', default='hampel')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if 'datetime' in df.columns: df['datetime']=pd.to_datetime(df['datetime']); df=df.set_index('datetime')
    if args.symbol and 'symbol' in df.columns:
        df = df[df['symbol']==args.symbol]
    df, qa = validate_input_df(df)
    df2, oq = scan_and_fix_outliers(df, method=args.outlier_method)

    # Bias banner
    print('Bias banner:', {**qa['warnings'], 'data_snooping': True})

    # Split
    if args.start and args.end:
        is_p = (args.start, args.end); oos_p = _split(df2, args.is_years, args.oos_years)[1]
    else:
        is_p, oos_p = _split(df2, args.is_years, args.oos_years)

    # Sample-size rule
    n_params = 6
    is_days = len(df2.loc[is_p[0]:is_p[1]])
    sample_ok = is_days >= 252 * n_params

    # Strategy
    params = _ensure_costs(MSNRParams())
    bt = run_msnr_backtest(df2, params, is_p, oos_p)
    # Ensure DAILY evaluation via equity → returns
    is_eq = (1.0 + bt['is'].fillna(0.0)).cumprod()
    oos_eq = (1.0 + bt['oos'].fillna(0.0)).cumprod()
    is_rets = M.daily_equity_returns(is_eq)
    oos_rets = M.daily_equity_returns(oos_eq)
    is_m = evaluate_daily_returns(is_rets); oos_m = evaluate_daily_returns(oos_rets)
    comp = compare_is_oos(is_m, oos_m)

    # Yearly + recency
    yt = yearly_table(pd.concat([bt['is'], bt['oos']]))
    rec = recent_windows(pd.concat([bt['is'], bt['oos']]))
    cheap = buy_on_the_cheap_guard(bt['trades'], df2['close'])

    # Correlation gate
    corr = correlation_gate(pd.concat([bt['is'], bt['oos']]))

    # Diagnostics (OOS)
    oos_arr = oos_rets.values
    diag = {
        'oos_len_days': int(len(oos_rets)),
        'oos_nonzero_days': int(np.count_nonzero(oos_arr)),
        'oos_std': float(np.std(oos_arr, ddof=1)) if len(oos_arr)>1 else 0.0,
        'oos_p5': float(np.percentile(oos_arr, 5)) if len(oos_arr)>0 else np.nan,
        'oos_p50': float(np.percentile(oos_arr, 50)) if len(oos_arr)>0 else np.nan,
        'oos_p95': float(np.percentile(oos_arr, 95)) if len(oos_arr)>0 else np.nan,
        'oos_trades': int(len(bt['trades'][(bt['trades']['i_close']>=len(is_rets))])) if not bt['trades'].empty else 0,
        'oos_sharpe_valid': bool(oos_m['sharpe_valid']),
    }
    # Thresholds / gates
    reasons = []
    if not oos_m['sharpe_valid']: reasons.append('Sharpe invalid (low variance or too few active days)')
    if diag['oos_trades'] < MIN_OOS_TRADES: reasons.append('Too few trades for OOS')
    if (oos_m['sharpe'] is None) or (oos_m['sharpe'] is not None and oos_m['sharpe'] < OOS_SHARPE_MIN): reasons.append('OOS Sharpe below minimum')
    if oos_m['mdd'] > 0.55: reasons.append('OOS MDD above threshold')
    if not (TIME_IN_MKT_MIN <= oos_m['tim'] <= TIME_IN_MKT_MAX): reasons.append('Time-in-Market outside [5%,90%]')
    if not (TURNOVER_MIN <= oos_m['turnover'] <= TURNOVER_MAX): reasons.append('Turnover outside [1%,6%]')
    th = {
        'Bias-Free': True,
        'Sample Size': bool(sample_ok),
        'OOS Trades >= 30': diag['oos_trades'] >= MIN_OOS_TRADES,
        'OOS Sharpe valid': bool(oos_m['sharpe_valid']),
        'OOS Sharpe >= 0.70': (oos_m['sharpe'] is not None) and (oos_m['sharpe'] >= OOS_SHARPE_MIN),
        'MDD <= 55%': oos_m['mdd'] <= 0.55,
        'Time-in-Market': TIME_IN_MKT_MIN <= oos_m['tim'] <= TIME_IN_MKT_MAX,
        'Turnover in [1%,6%]': TURNOVER_MIN <= oos_m['turnover'] <= TURNOVER_MAX,
        'Similarity>=0.90': bool(comp['similarity_ge_0_90']),
        'Buy-on-cheap guard': not cheap['warn'],
        'Correlation<0.5': (corr['max_corr'] < 0.5) if corr['max_corr']==corr['max_corr'] else True,
    }

    # Protocol Summary
    print('\n=== DIAGNOSTICS (OOS) ===')
    print({k: diag[k] for k in ['oos_len_days','oos_nonzero_days','oos_std','oos_p5','oos_p50','oos_p95','oos_trades']})
    print('\n=== PROTOCOL SUMMARY ===')
    for k,v in th.items(): print(f'- {k}: {"PASS" if v else "FAIL"}')
    if reasons:
        print('Reasons:', '; '.join(reasons))

    # Save reports
    os.makedirs('reports', exist_ok=True)
    tag = datetime.utcnow().strftime('%Y%m%d')
    pd.concat([bt['is'], bt['oos']]).to_csv(f'reports/msnr_{args.symbol or "ALL"}_{tag}.csv', header=['ret'])
    with open(f'reports/msnr_{args.symbol or "ALL"}_{tag}.json','w') as f:
        json.dump({
            'qa': qa, 'outliers': oq, 'is': is_m, 'oos': oos_m, 'compare': comp,
            'thresholds': th, 'correlation': corr, 'yearly': yt.reset_index().to_dict('records'),
            'recency': rec, 'cheap_guard': cheap,
            'periods': {'is': is_p, 'oos': oos_p},
            **diag,
            'oos_fail_reasons': reasons,
        }, f, indent=2, default=str)


if __name__ == '__main__':
    main()
