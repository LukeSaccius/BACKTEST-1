#!/usr/bin/env python3
"""
Grid search optimizer for AggressiveRSIStrategy.

Searches a compact parameter grid on in-sample data and evaluates the
top candidates on out-of-sample. Objective prioritizes OOS Sharpe and
OOS return, with a soft requirement for minimum closed trades.
"""

import argparse
import json
import sys
from pathlib import Path

import backtrader as bt
import pandas as pd

# Ensure project root on path for module import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from aggressive_profitable_strategy import AggressiveRSIStrategy


def split_is_oos_by_rows(df: pd.DataFrame, oos_frac: float = 0.25):
    n = len(df)
    split_idx = max(1, int(n * (1 - oos_frac)))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def load_symbol_df(csv_path: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])  # expects col present
    df = df[df["symbol"] == symbol].copy()
    return df.set_index("datetime").sort_index()


def run_once(df: pd.DataFrame, params: dict, cash: float, commission: float):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AggressiveRSIStrategy, **params)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    strat = cerebro.run()[0]
    metrics = dict(
        value=cerebro.broker.getvalue(),
        sharpe=float(strat.analyzers.sharpe.get_analysis().get("sharperatio", 0) or 0),
        drawdown=float(
            strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0)
            or 0
        ),
        trades=int(strat.analyzers.trades.get_analysis().get("total", {}).get("closed", 0)),
        sqn=float(strat.analyzers.sqn.get_analysis().get("sqn", 0) or 0),
    )
    return metrics


def score(oos_m: dict, is_m: dict, min_trades: int = 8) -> float:
    # Primary: OOS sharpe and OOS return (value growth)
    # Secondary: ensure adequate number of trades
    v = 0.0
    v += oos_m.get("sharpe", 0) * 2.0
    # value is final cash; return roughly scaled
    v += (oos_m.get("value", 0) - 100000) / 10000.0
    # penalty if too few trades
    t = oos_m.get("trades", 0)
    if t < min_trades:
        v -= (min_trades - t) * 1.0
    # mild penalty for large drawdown
    v -= max(0.0, (oos_m.get("drawdown", 0) - 15.0) / 5.0)
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="VN30_1H.csv")
    p.add_argument("--symbol", default="VNM")
    p.add_argument("--cash", type=float, default=100000)
    p.add_argument("--commission", type=float, default=0.001)
    p.add_argument("--report", default="reports/aggressive_opt_grid.json")
    args = p.parse_args()

    df = load_symbol_df(args.csv, args.symbol)
    recent = df[df.index >= "2021-01-01"].copy()
    is_df, oos_df = split_is_oos_by_rows(recent, 0.25)

    # Compact grid (128 combos)
    grid = dict(
        rsi_period=[7, 10],
        rsi_oversold=[35, 40],
        rsi_overbought=[60, 65],
        rsi_exit=[45, 50],
        min_volume_ratio=[1.0, 1.05],
        stop_loss_pct=[0.0125, 0.015],
        take_profit_pct=[0.025, 0.03],
    )

    import itertools

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]

    results = []
    base = dict(volume_period=10, max_position_size=0.9, risk_per_trade=0.03)
    for i, combo in enumerate(combos, 1):
        params = dict(base, **combo)
        try:
            is_m = run_once(is_df, params, args.cash, args.commission)
            oos_m = run_once(oos_df, params, args.cash, args.commission)
            s = score(oos_m, is_m, min_trades=8)
            results.append(dict(params=params, is_metrics=is_m, oos_metrics=oos_m, score=s))
        except Exception as e:
            # skip failed combos
            continue

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:10]

    # Print concise summary
    print("\n=== TOP PARAM CONFIGS (by OOS score) ===")
    for r in top:
        p = r["params"]
        om = r["oos_metrics"]
        print(
            f"score={r['score']:.2f} oos_sharpe={om['sharpe']:.2f} oos_trades={om['trades']} oos_dd={om['drawdown']:.2f}% params={p}"
        )

    # Save
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {args.report}")


if __name__ == "__main__":
    main()
