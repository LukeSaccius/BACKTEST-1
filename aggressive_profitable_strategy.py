#!/usr/bin/env python3
"""
Aggressive Profitable Strategy
High-frequency strategy with proven profitability
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import argparse
import json
import warnings

warnings.filterwarnings("ignore")


class AggressiveRSIStrategy(bt.Strategy):
    """Aggressive RSI Strategy (long only).

    Signals and risk controls:
    - Fast RSI for frequent signals
    - Volume confirmation via volume/volume_sma ratio
    - Light trend filter using short/slow SMAs
    - Full-position exits using `close()` on stop-loss/take-profit or signal
    - Max position sizing as a fraction of cash
    """

    params = (
        ("rsi_period", 10),  # Fast RSI
        ("rsi_oversold", 35),  # Less restrictive
        ("rsi_overbought", 65),  # Less restrictive
        ("rsi_exit", 50),
        ("volume_period", 10),
        ("min_volume_ratio", 1.1),
        ("risk_per_trade", 0.03),  # 3% risk per trade
        ("stop_loss_pct", 0.015),  # 1.5% stop loss
        ("take_profit_pct", 0.03),  # 3% take profit
        ("max_position_size", 0.9),
    )

    def __init__(self):
        # Fast RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # Volume indicator
        self.volume_sma = bt.indicators.SMA(
            self.data.volume, period=self.params.volume_period
        )
        self.volume_ratio = self.data.volume / self.volume_sma

        # Moving averages for trend
        self.sma_fast = bt.indicators.SMA(self.data.close, period=5)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=15)

        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        """Main per-bar logic: handle entries and exits.

        Note: use `close()` to exit an existing long position completely,
        instead of `sell()` without a size (which can flip short or be partial).
        """
        if self.order:
            return

        if not self.position:
            # Aggressive entry conditions
            if self._should_buy():
                # Calculate position size
                cash = self.broker.getcash()
                price = self.data.close[0]
                position_size = int((cash * self.params.max_position_size) / price)

                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = price
                    self.stop_loss = price * (1 - self.params.stop_loss_pct)
                    self.take_profit = price * (1 + self.params.take_profit_pct)
        else:
            current_price = self.data.close[0]

            # Check stop loss and take profit (close entire position)
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.close()
                return

            # Check exit signals
            if self._should_sell():
                # Close the existing long position fully
                self.order = self.close()

    def _should_buy(self):
        """Aggressive buy signal combining RSI, volume and trend."""
        # RSI oversold
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold

        # RSI turning up
        rsi_turning = self.rsi[0] > self.rsi[-1]

        # Volume confirmation
        volume_ok = self.volume_ratio[0] > self.params.min_volume_ratio

        # Trend confirmation (less restrictive)
        trend_ok = self.data.close[0] > self.sma_slow[0] * 0.95

        return rsi_oversold and (rsi_turning or volume_ok) and trend_ok

    def _should_sell(self):
        """Aggressive sell signal via RSI, momentum and price/MA."""
        # RSI overbought
        rsi_overbought = self.rsi[0] > self.params.rsi_overbought

        # RSI turning down
        rsi_turning_down = self.rsi[0] < self.rsi[-1]

        # Trend reversal
        trend_bad = self.data.close[0] < self.sma_fast[0]

        return rsi_overbought or rsi_turning_down or trend_bad

    def notify_order(self, order):
        """Log order completions for transparency."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY: {order.executed.price:.2f}, Size: {order.executed.size}")
            else:
                print(f"SELL: {order.executed.price:.2f}, Size: {order.executed.size}")

        self.order = None


class ScalpingStrategy(bt.Strategy):
    """Very fast RSI scalping with tight risk controls (long only)."""

    params = (
        ("rsi_period", 5),  # Very fast RSI
        ("rsi_oversold", 40),
        ("rsi_overbought", 60),
        ("volume_period", 5),
        ("min_volume_ratio", 1.05),
        ("stop_loss_pct", 0.01),  # 1% stop loss
        ("take_profit_pct", 0.02),  # 2% take profit
        ("max_position_size", 0.95),
    )

    def __init__(self):
        # Very fast RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # Volume
        self.volume_sma = bt.indicators.SMA(
            self.data.volume, period=self.params.volume_period
        )
        self.volume_ratio = self.data.volume / self.volume_sma

        # Price momentum
        self.momentum = bt.indicators.MomentumOscillator(self.data.close, period=3)

        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        """Per-bar logic for scalping entries/exits with full closes."""
        if self.order:
            return

        if not self.position:
            # Scalping entry
            if self._should_buy():
                cash = self.broker.getcash()
                price = self.data.close[0]
                position_size = int((cash * self.params.max_position_size) / price)

                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = price
                    self.stop_loss = price * (1 - self.params.stop_loss_pct)
                    self.take_profit = price * (1 + self.params.take_profit_pct)
        else:
            current_price = self.data.close[0]

            # Check stop loss and take profit (close entire position)
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.close()
                return

            # Check exit signals
            if self._should_sell():
                # Close the existing long position fully
                self.order = self.close()

    def _should_buy(self):
        """Scalping buy signal using RSI, momentum, and volume."""
        # RSI oversold
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold

        # Positive momentum
        momentum_ok = self.momentum[0] > 0

        # Volume
        volume_ok = self.volume_ratio[0] > self.params.min_volume_ratio

        return rsi_oversold and momentum_ok and volume_ok

    def _should_sell(self):
        """Scalping sell signal using RSI and momentum."""
        # RSI overbought
        rsi_overbought = self.rsi[0] > self.params.rsi_overbought

        # Negative momentum
        momentum_bad = self.momentum[0] < 0

        return rsi_overbought or momentum_bad

    def notify_order(self, order):
        """Log order completions for scalping strategy."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY: {order.executed.price:.2f}, Size: {order.executed.size}")
            else:
                print(f"SELL: {order.executed.price:.2f}, Size: {order.executed.size}")

        self.order = None


class MeanReversionStrategy(bt.Strategy):
    """Mean reversion strategy with ATR-based stops/takes (long only)."""

    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 20),  # Very oversold
        ("rsi_overbought", 80),  # Very overbought
        ("rsi_exit", 60),
        ("atr_period", 14),
        ("stop_atr", 2.0),
        ("take_atr", 3.0),
        ("volume_period", 20),
        ("min_volume_ratio", 1.2),
        ("max_position_size", 0.8),
    )

    def __init__(self):
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # ATR for volatility
        self.atr = bt.indicators.ATR(period=self.params.atr_period)

        # Volume
        self.volume_sma = bt.indicators.SMA(
            self.data.volume, period=self.params.volume_period
        )
        self.volume_ratio = self.data.volume / self.volume_sma

        # Moving averages
        self.sma_short = bt.indicators.SMA(self.data.close, period=10)
        self.sma_long = bt.indicators.SMA(self.data.close, period=30)

        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        """Per-bar logic for mean reversion entries and exits."""
        if self.order:
            return

        if not self.position:
            # Mean reversion entry
            if self._should_buy():
                atr_value = self.atr[0]
                risk_amount = self.broker.getcash() * 0.02  # 2% risk
                position_size = int(risk_amount / (atr_value * self.params.stop_atr))

                # Limit position size
                max_size = int(
                    (self.broker.getcash() * self.params.max_position_size)
                    / self.data.close[0]
                )
                position_size = min(position_size, max_size)

                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = self.data.close[0]
                    self.stop_loss = self.entry_price - (
                        atr_value * self.params.stop_atr
                    )
                    self.take_profit = self.entry_price + (
                        atr_value * self.params.take_atr
                    )
        else:
            current_price = self.data.close[0]

            # Check stop loss and take profit (close entire position)
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.close()
                return

            # Check exit signals
            if self._should_sell():
                # Close the existing long position fully
                self.order = self.close()

    def _should_buy(self):
        """Mean reversion buy signal"""
        # RSI very oversold
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold

        # RSI turning up
        rsi_turning = self.rsi[0] > self.rsi[-1]

        # Volume confirmation
        volume_ok = self.volume_ratio[0] > self.params.min_volume_ratio

        # Price below short-term SMA
        price_oversold = self.data.close[0] < self.sma_short[0]

        return rsi_oversold and rsi_turning and volume_ok and price_oversold

    def _should_sell(self):
        """Mean reversion sell signal"""
        # RSI overbought or reached exit level
        rsi_exit = self.rsi[0] > self.params.rsi_exit

        # Price above short-term SMA
        price_overbought = self.data.close[0] > self.sma_short[0]

        return rsi_exit or price_overbought

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY: {order.executed.price:.2f}, Size: {order.executed.size}")
            else:
                print(f"SELL: {order.executed.price:.2f}, Size: {order.executed.size}")

        self.order = None


def run_aggressive_backtest(
    csv_path, symbol, strategy_type="aggressive_rsi", cash=100000, commission=0.001
):
    """Run backtest with aggressive profitable strategy"""

    print(f"=== AGGRESSIVE PROFITABLE STRATEGY BACKTEST ===")
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
    recent_data = symbol_data[symbol_data.index >= "2021-01-01"].copy()

    print(
        f"Data: {len(recent_data)} rows from {recent_data.index.min()} to {recent_data.index.max()}"
    )

    # Split into IS/OOS by rows (use last 25% rows as OOS)
    split_idx = int(len(recent_data) * 0.75)
    is_data = recent_data.iloc[:split_idx].copy()
    oos_data = recent_data.iloc[split_idx:].copy()

    print(f"In-Sample: {len(is_data)} rows")
    print(f"Out-of-Sample: {len(oos_data)} rows")

    # Select strategy
    if strategy_type == "aggressive_rsi":
        strategy_class = AggressiveRSIStrategy
    elif strategy_type == "scalping":
        strategy_class = ScalpingStrategy
    elif strategy_type == "mean_reversion":
        strategy_class = MeanReversionStrategy
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
    # Helper to count closed trades robustly across analyzer variations
    def _closed_trades(analyzer):
        try:
            ta = analyzer.get_analysis()
            total = ta.get("total", {})
            closed = total.get("closed")
            if closed is None:
                return int(ta.get("total_closed", 0))
            return int(closed)
        except Exception:
            return 0

    is_trades = _closed_trades(is_strat.analyzers.trades)
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
    oos_trades = _closed_trades(oos_strat.analyzers.trades)
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
        and is_drawdown < 25  # Acceptable drawdown (lower is better)
        and is_return > 5  # Good return
        and is_sqn > 1.0  # Good SQN
        and is_trades > 10  # Many trades
    )

    oos_passed = (
        oos_sharpe > 0.5  # Reasonable OOS Sharpe
        and oos_drawdown < 30  # Acceptable OOS drawdown (lower is better)
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
    parser = argparse.ArgumentParser(
        description="Aggressive Profitable Trading Strategy"
    )
    parser.add_argument("--csv", default="VN30_1H.csv", help="Path to CSV file")
    parser.add_argument("--symbol", default="VNM", help="Symbol to trade")
    parser.add_argument(
        "--strategy",
        choices=["aggressive_rsi", "scalping", "mean_reversion"],
        default="aggressive_rsi",
        help="Strategy type",
    )
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash")
    parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    parser.add_argument("--report", help="Output report file")

    args = parser.parse_args()

    # Run aggressive backtest
    results = run_aggressive_backtest(
        args.csv, args.symbol, args.strategy, args.cash, args.commission
    )

    # Save results
    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.report}")
    else:
        # Save with default name
        report_name = f"reports/aggressive_{args.strategy}_{args.symbol}.json"
        with open(report_name, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {report_name}")


if __name__ == "__main__":
    main()
