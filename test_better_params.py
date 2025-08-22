#!/usr/bin/env python3
"""
Test script with better parameters to demonstrate the framework working
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import json


class BetterSMAStrategy(bt.Strategy):
    """SMA Strategy with better parameters for demonstration"""

    params = (
        ("fast_period", 5),
        ("slow_period", 20),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period
        )
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
        else:
            if self.crossover < 0:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None


def run_demo_backtest():
    """Run a demonstration backtest with better parameters"""

    # Load data
    print("Loading data...")
    df = pd.read_csv("VN30_1H.csv")

    # Filter VNM and prepare data
    vnm_data = df[df["symbol"] == "VNM"].copy()
    vnm_data["datetime"] = pd.to_datetime(vnm_data["datetime"])
    vnm_data = vnm_data.set_index("datetime").sort_index()

    # Use only recent data for better performance
    recent_data = vnm_data[vnm_data.index >= "2020-01-01"].copy()

    print(f"Using {len(recent_data)} rows from 2020 onwards")

    # Split into IS/OOS
    split_date = recent_data.index.max() - timedelta(days=365)  # 1 year OOS
    is_data = recent_data[recent_data.index <= split_date].copy()
    oos_data = recent_data[recent_data.index > split_date].copy()

    print(f"In-Sample: {len(is_data)} rows")
    print(f"Out-of-Sample: {len(oos_data)} rows")

    # Run IS backtest
    print("\n=== IN-SAMPLE BACKTEST ===")
    cerebro_is = bt.Cerebro()
    cerebro_is.addstrategy(BetterSMAStrategy)
    cerebro_is.adddata(bt.feeds.PandasData(dataname=is_data))
    cerebro_is.broker.setcash(100000)
    cerebro_is.broker.setcommission(commission=0.001)
    cerebro_is.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro_is.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro_is.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro_is.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    is_results = cerebro_is.run()
    is_strat = is_results[0]

    # Calculate IS metrics
    is_return = (cerebro_is.broker.getvalue() - 100000) / 100000 * 100
    is_sharpe = is_strat.analyzers.sharpe.get_analysis().get("sharperatio", 0)
    is_drawdown = (
        is_strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0)
    )
    is_trades = len(is_strat.analyzers.trades.get_analysis())

    print(f"IS Return: {is_return:.2f}%")
    print(f"IS Sharpe: {is_sharpe:.3f}")
    print(f"IS Max DD: {is_drawdown:.2f}%")
    print(f"IS Trades: {is_trades}")

    # Run OOS backtest
    print("\n=== OUT-OF-SAMPLE BACKTEST ===")
    cerebro_oos = bt.Cerebro()
    cerebro_oos.addstrategy(BetterSMAStrategy)
    cerebro_oos.adddata(bt.feeds.PandasData(dataname=oos_data))
    cerebro_oos.broker.setcash(100000)
    cerebro_oos.broker.setcommission(commission=0.001)
    cerebro_oos.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro_oos.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro_oos.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro_oos.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    oos_results = cerebro_oos.run()
    oos_strat = oos_results[0]

    # Calculate OOS metrics
    oos_return = (cerebro_oos.broker.getvalue() - 100000) / 100000 * 100
    oos_sharpe = oos_strat.analyzers.sharpe.get_analysis().get("sharperatio", 0)
    oos_drawdown = (
        oos_strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0)
    )
    oos_trades = len(oos_strat.analyzers.trades.get_analysis())

    print(f"OOS Return: {oos_return:.2f}%")
    print(f"OOS Sharpe: {oos_sharpe:.3f}")
    print(f"OOS Max DD: {oos_drawdown:.2f}%")
    print(f"OOS Trades: {oos_trades}")

    # Overall assessment
    print("\n=== OVERALL ASSESSMENT ===")

    # Check if strategy meets basic criteria
    is_passed = is_sharpe > 0.5 and is_drawdown < 20
    oos_passed = oos_sharpe > 0.3 and oos_drawdown < 25

    if is_passed and oos_passed:
        print("✅ Strategy shows reasonable performance!")
        print("✅ In-Sample: Acceptable Sharpe and Drawdown")
        print("✅ Out-of-Sample: Acceptable Sharpe and Drawdown")
    else:
        print("❌ Strategy needs improvement")
        if not is_passed:
            print("❌ In-Sample performance below threshold")
        if not oos_passed:
            print("❌ Out-of-Sample performance below threshold")

    return {
        "in_sample": {
            "return": is_return,
            "sharpe": is_sharpe,
            "drawdown": is_drawdown,
            "trades": is_trades,
        },
        "out_of_sample": {
            "return": oos_return,
            "sharpe": oos_sharpe,
            "drawdown": oos_drawdown,
            "trades": oos_trades,
        },
    }


if __name__ == "__main__":
    results = run_demo_backtest()

    # Save results
    with open("reports/demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to reports/demo_results.json")
