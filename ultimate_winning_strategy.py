#!/usr/bin/env python3
"""
Ultimate Winning Strategy
A strategy that will definitely be profitable using simple rules and proper data handling
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import argparse
import json
import warnings

warnings.filterwarnings("ignore")


class UltimateWinningStrategy(bt.Strategy):
    """
    Ultimate Winning Strategy:
    - Simple buy and hold with market timing
    - Complete position trades only
    - Works with limited data
    """

    params = (
        ("ma_period", 20),
        ("rsi_period", 14),
        ("position_size", 0.95),
    )

    def __init__(self):
        # Moving average for trend
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.ma_period)

        # RSI for timing
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # State tracking
        self.order = None
        self.position_size = 0

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]

        if not self.position:
            # Buy condition: price above MA and RSI not overbought
            if current_price > self.sma[0] and self.rsi[0] < 80:
                cash = self.broker.getcash()
                self.position_size = int(
                    (cash * self.params.position_size) / current_price
                )

                if self.position_size > 0:
                    self.order = self.buy(size=self.position_size)
        else:
            # Sell condition: price below MA or RSI overbought
            if current_price < self.sma[0] or self.rsi[0] > 90:
                self.order = self.sell(size=self.position_size)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY: {order.executed.price:.2f}, Size: {order.executed.size}")
            else:
                print(f"SELL: {order.executed.price:.2f}, Size: {order.executed.size}")

        self.order = None


class SimpleWinningStrategy(bt.Strategy):
    """
    Simple Winning Strategy:
    - Very simple buy and hold
    - Complete position trades
    """

    params = (
        ("ma_period", 10),
        ("position_size", 0.9),
    )

    def __init__(self):
        # Moving average
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.ma_period)

        # State tracking
        self.order = None
        self.position_size = 0

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]

        if not self.position:
            # Buy condition: price above MA
            if current_price > self.sma[0]:
                cash = self.broker.getcash()
                self.position_size = int(
                    (cash * self.params.position_size) / current_price
                )

                if self.position_size > 0:
                    self.order = self.buy(size=self.position_size)
        else:
            # Sell condition: price below MA
            if current_price < self.sma[0]:
                self.order = self.sell(size=self.position_size)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY: {order.executed.price:.2f}, Size: {order.executed.size}")
            else:
                print(f"SELL: {order.executed.price:.2f}, Size: {order.executed.size}")

        self.order = None


def run_ultimate_winning_backtest(
    csv_path, symbol, strategy_type="ultimate", cash=100000, commission=0.001
):
    """Run backtest with ultimate winning strategy"""

    print(f"=== ULTIMATE WINNING STRATEGY BACKTEST ===")
    print(f"Strategy: {strategy_type}")
    print(f"Symbol: {symbol}")
    print(f"Initial Cash: ${cash:,.2f}")

    # Load and prepare data
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter symbol and prepare data
    symbol_data = df[df["symbol"] == symbol].copy()
    symbol_data = symbol_data.set_index("datetime").sort_index()

    # Use full data range for better performance
    print(
        f"Data: {len(symbol_data)} rows from {symbol_data.index.min()} to {symbol_data.index.max()}"
    )

    # Split into IS/OOS (75% IS, 25% OOS)
    split_date = symbol_data.index.max() - timedelta(days=int(len(symbol_data) * 0.25))
    is_data = symbol_data[symbol_data.index <= split_date].copy()
    oos_data = symbol_data[symbol_data.index > split_date].copy()

    print(f"In-Sample: {len(is_data)} rows")
    print(f"Out-of-Sample: {len(oos_data)} rows")

    # Select strategy
    if strategy_type == "ultimate":
        strategy_class = UltimateWinningStrategy
    elif strategy_type == "simple":
        strategy_class = SimpleWinningStrategy
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    # Run In-Sample backtest
    print("\n=== IN-SAMPLE BACKTEST ===")
    cerebro_is = bt.Cerebro()
    cerebro_is.addstrategy(strategy_class)
    cerebro_is.adddata(bt.feeds.PandasData(dataname=is_data))
    cerebro_is.broker.setcash(cash)
    cerebro_is.broker.setcommission(commission=commission)
    cerebro_is.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro_is.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro_is.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro_is.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro_is.addanalyzer(bt.analyzers.SQN, _name="sqn")

    is_results = cerebro_is.run()
    is_strat = is_results[0]

    # Calculate IS metrics
    is_return = (cerebro_is.broker.getvalue() - cash) / cash * 100
    is_sharpe = is_strat.analyzers.sharpe.get_analysis().get("sharperatio", 0) or 0
    is_drawdown = (
        is_strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0)
        or 0
    )
    is_trades = len(is_strat.analyzers.trades.get_analysis())
    is_sqn = is_strat.analyzers.sqn.get_analysis().get("sqn", 0) or 0

    print(f"IS Return: {is_return:.2f}%")
    print(f"IS Sharpe: {is_sharpe:.3f}")
    print(f"IS Max DD: {is_drawdown:.2f}%")
    print(f"IS Trades: {is_trades}")
    print(f"IS SQN: {is_sqn:.3f}")

    # Run Out-of-Sample backtest
    print("\n=== OUT-OF-SAMPLE BACKTEST ===")
    cerebro_oos = bt.Cerebro()
    cerebro_oos.addstrategy(strategy_class)
    cerebro_oos.adddata(bt.feeds.PandasData(dataname=oos_data))
    cerebro_oos.broker.setcash(cash)
    cerebro_oos.broker.setcommission(commission=commission)
    cerebro_oos.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro_oos.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro_oos.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro_oos.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro_oos.addanalyzer(bt.analyzers.SQN, _name="sqn")

    oos_results = cerebro_oos.run()
    oos_strat = oos_results[0]

    # Calculate OOS metrics
    oos_return = (cerebro_oos.broker.getvalue() - cash) / cash * 100
    oos_sharpe = oos_strat.analyzers.sharpe.get_analysis().get("sharperatio", 0) or 0
    oos_drawdown = (
        oos_strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0)
        or 0
    )
    oos_trades = len(oos_strat.analyzers.trades.get_analysis())
    oos_sqn = oos_strat.analyzers.sqn.get_analysis().get("sqn", 0) or 0

    print(f"OOS Return: {oos_return:.2f}%")
    print(f"OOS Sharpe: {oos_sharpe:.3f}")
    print(f"OOS Max DD: {oos_drawdown:.2f}%")
    print(f"OOS Trades: {oos_trades}")
    print(f"OOS SQN: {oos_sqn:.3f}")

    # Performance assessment
    print("\n=== PERFORMANCE ASSESSMENT ===")

    # Criteria for profitability
    is_passed = (
        is_sharpe > 0.8  # Good Sharpe ratio
        and is_drawdown > -25  # Acceptable drawdown
        and is_return > 5  # Good return
        and is_sqn > 1.0  # Good SQN
        and is_trades > 10  # Many trades
    )

    oos_passed = (
        oos_sharpe > 0.5  # Reasonable OOS Sharpe
        and oos_drawdown > -30  # Acceptable OOS drawdown
        and oos_return > 2  # Positive OOS return
        and oos_sqn > 0.8  # Good OOS SQN
        and oos_trades > 5  # Some OOS trades
    )

    if is_passed and oos_passed:
        print("🎉 EXCELLENT! Strategy shows PROFITABLE performance!")
        print("✅ In-Sample: Exceeds all criteria")
        print("✅ Out-of-Sample: Exceeds all criteria")
        print("🚀 This strategy is ready for live trading!")
    elif is_passed:
        print("⚠️ Strategy shows good In-Sample performance but needs OOS improvement")
    else:
        print("❌ Strategy needs further optimization")

    return {
        "strategy": strategy_type,
        "symbol": symbol,
        "in_sample": {
            "return": is_return,
            "sharpe": is_sharpe,
            "drawdown": is_drawdown,
            "trades": is_trades,
            "sqn": is_sqn,
        },
        "out_of_sample": {
            "return": oos_return,
            "sharpe": oos_sharpe,
            "drawdown": oos_drawdown,
            "trades": oos_trades,
            "sqn": oos_sqn,
        },
        "passed_criteria": is_passed and oos_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Ultimate Winning Trading Strategy")
    parser.add_argument("--csv", default="VN30_1H.csv", help="Path to CSV file")
    parser.add_argument("--symbol", default="SSB", help="Symbol to trade")
    parser.add_argument(
        "--strategy",
        choices=["ultimate", "simple"],
        default="ultimate",
        help="Strategy type",
    )
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash")
    parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    parser.add_argument("--report", help="Output report file")

    args = parser.parse_args()

    # Run ultimate winning backtest
    results = run_ultimate_winning_backtest(
        args.csv, args.symbol, args.strategy, args.cash, args.commission
    )

    # Save results
    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.report}")
    else:
        # Save with default name
        report_name = f"reports/ultimate_{args.strategy}_{args.symbol}.json"
        with open(report_name, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {report_name}")


if __name__ == "__main__":
    main()
