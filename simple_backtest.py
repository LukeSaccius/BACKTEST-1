#!/usr/bin/env python3
"""
Simple Backtesting Framework
Clean, simple implementation for testing trading strategies
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import argparse
import json


class SimpleSMAStrategy(bt.Strategy):
    """Simple Moving Average Crossover Strategy"""

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )

    def __init__(self):
        # Create moving averages
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period
        )

        # Create crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

        # Track orders
        self.order = None

    def next(self):
        # Check if we have a pending order
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # We are not in the market, look for entry signal
            if self.crossover > 0:  # Fast MA crosses above slow MA
                self.order = self.buy()
        else:
            # We are in the market, look for exit signal
            if self.crossover < 0:  # Fast MA crosses below slow MA
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            else:
                print(f"SELL EXECUTED, Price: {order.executed.price:.2f}")

        self.order = None


class SimpleRSIStrategy(bt.Strategy):
    """Simple RSI Mean Reversion Strategy"""

    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
    )

    def __init__(self):
        # Create RSI indicator
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # Track orders
        self.order = None

    def next(self):
        # Check if we have a pending order
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # We are not in the market, look for entry signal
            if self.rsi < self.params.rsi_oversold:
                self.order = self.buy()
        else:
            # We are in the market, look for exit signal
            if self.rsi > self.params.rsi_overbought:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            else:
                print(f"SELL EXECUTED, Price: {order.executed.price:.2f}")

        self.order = None


def load_data(csv_path, symbol=None, start_date=None, end_date=None):
    """Load and prepare data for backtesting"""
    print(f"Loading data from {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Normalize column names
    column_mapping = {
        "datetime": ["datetime", "date", "timestamp"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close"],
        "volume": ["volume", "vol"],
        "symbol": ["symbol", "sym", "ticker"],
    }

    # Find and rename columns
    for target, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                df = df.rename(columns={name: target})
                break

    # Convert datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter by symbol if specified
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
        print(f"Filtered to symbol {symbol}: {len(df)} rows")

    # Filter by date range if specified
    if start_date and "datetime" in df.columns:
        df = df[df["datetime"] >= start_date]
    if end_date and "datetime" in df.columns:
        df = df[df["datetime"] <= end_date]

    # Sort by datetime
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    # Set datetime as index for backtrader
    if "datetime" in df.columns:
        df = df.set_index("datetime")

    print(f"Final dataset: {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def run_backtest(data, strategy_class, strategy_params, cash=100000, commission=0.001):
    """Run a simple backtest"""
    print(f"\n=== Running Backtest ===")
    print(f"Strategy: {strategy_class.__name__}")
    print(f"Parameters: {strategy_params}")
    print(f"Initial Cash: ${cash:,.2f}")
    print(f"Commission: {commission:.3f}")

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(strategy_class, **strategy_params)

    # Add data
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set initial cash
    cerebro.broker.setcash(cash)

    # Set commission
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run backtest
    print("\nStarting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    results = cerebro.run()
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Get results
    strat = results[0]

    # Extract metrics
    metrics = {
        "initial_value": cash,
        "final_value": cerebro.broker.getvalue(),
        "total_return": (cerebro.broker.getvalue() - cash) / cash * 100,
        "sharpe_ratio": strat.analyzers.sharpe.get_analysis().get("sharperatio", 0),
        "annual_return": strat.analyzers.returns.get_analysis().get("rnorm100", 0),
        "max_drawdown": strat.analyzers.drawdown.get_analysis()
        .get("max", {})
        .get("drawdown", 0),
        "total_trades": len(strat.analyzers.trades.get_analysis()),
    }

    print(f"\n=== Results ===")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Annual Return: {metrics['annual_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Simple Backtesting Framework")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--symbol", help="Symbol to trade (if multi-symbol data)")
    parser.add_argument(
        "--strategy", choices=["sma", "rsi"], default="sma", help="Strategy to use"
    )
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash")
    parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    parser.add_argument("--report", help="Output report file")

    args = parser.parse_args()

    # Load data
    data = load_data(args.csv, args.symbol)

    # Select strategy
    if args.strategy == "sma":
        strategy_class = SimpleSMAStrategy
        strategy_params = {"fast_period": 10, "slow_period": 30}
    elif args.strategy == "rsi":
        strategy_class = SimpleRSIStrategy
        strategy_params = {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}

    # Run backtest
    results = run_backtest(
        data, strategy_class, strategy_params, args.cash, args.commission
    )

    # Save results if requested
    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.report}")


if __name__ == "__main__":
    main()
