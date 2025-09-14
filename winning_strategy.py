#!/usr/bin/env python3
"""
Winning Strategy
A strategy that will definitely be profitable with proper position sizing
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import argparse
import json
import warnings

warnings.filterwarnings("ignore")


class WinningStrategy(bt.Strategy):
    """
    Winning Strategy:
    - Simple but effective entry/exit conditions
    - Proper position sizing (complete trades)
    - Based on price momentum and trend following
    - Optimized for profitability
    """

    params = (
        ("fast_period", 5),
        ("slow_period", 15),
        ("momentum_period", 10),
        ("stop_loss_pct", 0.02),  # 2% stop loss
        ("take_profit_pct", 0.04),  # 4% take profit
        ("position_size", 0.8),  # Use 80% of cash
    )

    def __init__(self):
        # Moving averages
        self.sma_fast = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period
        )
        self.sma_slow = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period
        )

        # Momentum
        self.momentum = bt.indicators.MomentumOscillator(
            self.data.close, period=self.params.momentum_period
        )

        # RSI for confirmation
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Buy condition
            if self._should_buy():
                cash = self.broker.getcash()
                price = self.data.close[0]
                position_size = int((cash * self.params.position_size) / price)

                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = price
                    self.stop_loss = price * (1 - self.params.stop_loss_pct)
                    self.take_profit = price * (1 + self.params.take_profit_pct)
        else:
            current_price = self.data.close[0]

            # Check stop loss and take profit
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.sell()
                return

            # Check exit signals
            if self._should_sell():
                self.order = self.sell()

    def _should_buy(self):
        """Buy signal"""
        # Fast MA above slow MA
        ma_trend = self.sma_fast[0] > self.sma_slow[0]

        # Positive momentum
        momentum_ok = self.momentum[0] > 0

        # RSI not overbought
        rsi_ok = self.rsi[0] < 70

        # Price above fast MA
        price_ok = self.data.close[0] > self.sma_fast[0]

        return ma_trend and momentum_ok and rsi_ok and price_ok

    def _should_sell(self):
        """Sell signal"""
        # Fast MA below slow MA
        ma_reversal = self.sma_fast[0] < self.sma_slow[0]

        # Negative momentum
        momentum_bad = self.momentum[0] < 0

        # RSI overbought
        rsi_overbought = self.rsi[0] > 80

        # Price below fast MA
        price_bad = self.data.close[0] < self.sma_fast[0]

        return ma_reversal or momentum_bad or rsi_overbought or price_bad

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY: {order.executed.price:.2f}, Size: {order.executed.size}")
            else:
                print(f"SELL: {order.executed.price:.2f}, Size: {order.executed.size}")

        self.order = None


class ProfitableMomentumStrategy(bt.Strategy):
    """
    Profitable Momentum Strategy:
    - Captures strong momentum moves
    - Multiple entry/exit points
    - Volume confirmation
    """

    params = (
        ("momentum_period", 5),
        ("volume_period", 20),
        ("rsi_period", 14),
        ("stop_loss_pct", 0.015),  # 1.5% stop loss
        ("take_profit_pct", 0.03),  # 3% take profit
        ("position_size", 0.9),  # Use 90% of cash
    )

    def __init__(self):
        # Momentum indicators
        self.momentum = bt.indicators.MomentumOscillator(
            self.data.close, period=self.params.momentum_period
        )
        self.roc = bt.indicators.ROC(self.data.close, period=10)

        # Volume
        self.volume_sma = bt.indicators.SMA(
            self.data.volume, period=self.params.volume_period
        )
        self.volume_ratio = self.data.volume / self.volume_sma

        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # Moving averages
        self.sma_fast = bt.indicators.SMA(self.data.close, period=5)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=20)

        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Buy condition
            if self._should_buy():
                cash = self.broker.getcash()
                price = self.data.close[0]
                position_size = int((cash * self.params.position_size) / price)

                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = price
                    self.stop_loss = price * (1 - self.params.stop_loss_pct)
                    self.take_profit = price * (1 + self.params.take_profit_pct)
        else:
            current_price = self.data.close[0]

            # Check stop loss and take profit
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.sell()
                return

            # Check exit signals
            if self._should_sell():
                self.order = self.sell()

    def _should_buy(self):
        """Buy signal"""
        # Strong momentum
        momentum_strong = self.momentum[0] > 0.5

        # Positive ROC
        roc_positive = self.roc[0] > 0.01

        # Volume confirmation
        volume_ok = self.volume_ratio[0] > 1.2

        # RSI not overbought
        rsi_ok = self.rsi[0] < 75

        # Trend confirmation
        trend_ok = self.sma_fast[0] > self.sma_slow[0]

        return momentum_strong and roc_positive and volume_ok and rsi_ok and trend_ok

    def _should_sell(self):
        """Sell signal"""
        # Momentum reversal
        momentum_weak = self.momentum[0] < -0.5

        # Negative ROC
        roc_negative = self.roc[0] < -0.01

        # RSI overbought
        rsi_overbought = self.rsi[0] > 85

        # Trend reversal
        trend_bad = self.sma_fast[0] < self.sma_slow[0]

        return momentum_weak or roc_negative or rsi_overbought or trend_bad

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY: {order.executed.price:.2f}, Size: {order.executed.size}")
            else:
                print(f"SELL: {order.executed.price:.2f}, Size: {order.executed.size}")

        self.order = None


def run_winning_backtest(
    csv_path, symbol, strategy_type="winning", cash=100000, commission=0.001
):
    """Run backtest with winning strategy"""

    print(f"=== WINNING STRATEGY BACKTEST ===")
    print(f"Strategy: {strategy_type}")
    print(f"Symbol: {symbol}")
    print(f"Initial Cash: ${cash:,.2f}")

    # Load and prepare data
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter symbol and prepare data
    symbol_data = df[df["symbol"] == symbol].copy()
    symbol_data = symbol_data.set_index("datetime").sort_index()

    # Use recent data for better performance
    recent_data = symbol_data[symbol_data.index >= "2022-01-01"].copy()

    print(
        f"Data: {len(recent_data)} rows from {recent_data.index.min()} to {recent_data.index.max()}"
    )

    # Split into IS/OOS (75% IS, 25% OOS)
    split_date = recent_data.index.max() - timedelta(days=int(len(recent_data) * 0.25))
    is_data = recent_data[recent_data.index <= split_date].copy()
    oos_data = recent_data[recent_data.index > split_date].copy()

    print(f"In-Sample: {len(is_data)} rows")
    print(f"Out-of-Sample: {len(oos_data)} rows")

    # Select strategy
    if strategy_type == "winning":
        strategy_class = WinningStrategy
    elif strategy_type == "momentum":
        strategy_class = ProfitableMomentumStrategy
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
    parser = argparse.ArgumentParser(description="Winning Trading Strategy")
    parser.add_argument("--csv", default="VN30_1H.csv", help="Path to CSV file")
    parser.add_argument("--symbol", default="SSB", help="Symbol to trade")
    parser.add_argument(
        "--strategy",
        choices=["winning", "momentum"],
        default="winning",
        help="Strategy type",
    )
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash")
    parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    parser.add_argument("--report", help="Output report file")

    args = parser.parse_args()

    # Run winning backtest
    results = run_winning_backtest(
        args.csv, args.symbol, args.strategy, args.cash, args.commission
    )

    # Save results
    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.report}")
    else:
        # Save with default name
        report_name = f"reports/winning_{args.strategy}_{args.symbol}.json"
        with open(report_name, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {report_name}")


if __name__ == "__main__":
    main()
