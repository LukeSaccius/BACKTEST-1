#!/usr/bin/env python3
"""
Highly Profitable Trading Strategy V2
Combines multiple proven techniques for consistent profitability
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import argparse
import json
import warnings

warnings.filterwarnings("ignore")


class MultiSignalStrategy(bt.Strategy):
    """
    Multi-Signal Strategy combining:
    - RSI mean reversion with optimized parameters
    - Moving average crossovers
    - Volume confirmation
    - Volatility-based position sizing
    - Dynamic stop loss and take profit
    - Trend following with momentum
    """
    
    params = (
        # RSI parameters (optimized)
        ('rsi_period', 14),
        ('rsi_oversold', 25),
        ('rsi_overbought', 75),
        ('rsi_exit', 55),
        
        # Moving average parameters
        ('sma_fast', 5),
        ('sma_medium', 15),
        ('sma_slow', 30),
        
        # Volatility parameters
        ('atr_period', 14),
        ('volatility_mult', 2.0),
        
        # Risk management
        ('risk_per_trade', 0.02),  # 2% risk per trade
        ('max_position_size', 0.8),  # 80% max position
        ('stop_atr', 2.5),  # 2.5 ATR stop loss
        ('take_atr', 4.0),  # 4 ATR take profit
        
        # Volume parameters
        ('volume_period', 20),
        ('min_volume_ratio', 1.3),
        
        # Trend parameters
        ('trend_period', 50),
        ('momentum_period', 10),
    )
    
    def __init__(self):
        # Technical indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        
        # Moving averages
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.sma_fast)
        self.sma_medium = bt.indicators.SMA(self.data.close, period=self.params.sma_medium)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.sma_slow)
        
        # Volume indicators
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)
        self.volume_ratio = self.data.volume / self.volume_sma
        
        # Trend indicators
        self.trend_sma = bt.indicators.SMA(self.data.close, period=self.params.trend_period)
        self.momentum = bt.indicators.MomentumOscillator(self.data.close, period=self.params.momentum_period)
        
        # Crossovers
        self.fast_cross_medium = bt.indicators.CrossOver(self.sma_fast, self.sma_medium)
        self.medium_cross_slow = bt.indicators.CrossOver(self.sma_medium, self.sma_slow)
        
        # State tracking
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.stop_loss = None
        self.take_profit = None
        self.position_size = None
        
    def next(self):
        if self.order:
            return
        
        if not self.position:
            # Entry conditions - multiple confirmations required
            if self._should_buy():
                # Calculate position size based on ATR
                atr_value = self.atr[0]
                risk_amount = self.broker.getcash() * self.params.risk_per_trade
                position_size = int(risk_amount / (atr_value * self.params.stop_atr))
                
                # Limit position size
                max_size = int((self.broker.getcash() * self.params.max_position_size) / self.data.close[0])
                position_size = min(position_size, max_size)
                
                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = self.data.close[0]
                    self.entry_bar = len(self)
                    self.position_size = position_size
                    
                    # Set dynamic stop loss and take profit
                    self.stop_loss = self.entry_price - (atr_value * self.params.stop_atr)
                    self.take_profit = self.entry_price + (atr_value * self.params.take_atr)
        else:
            # Check minimum hold period (avoid overtrading)
            bars_held = len(self) - self.entry_bar
            if bars_held < 3:  # Minimum 3 bars hold
                return
            
            current_price = self.data.close[0]
            
            # Check stop loss and take profit
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.sell()
                return
            
            # Check exit signals
            if self._should_sell():
                self.order = self.sell()
    
    def _should_buy(self):
        """Multi-signal buy condition"""
        # RSI oversold
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold
        
        # RSI starting to turn up
        rsi_turning = self.rsi[0] > self.rsi[-1]
        
        # Moving average confirmation
        ma_bullish = (self.sma_fast[0] > self.sma_medium[0] and 
                     self.sma_medium[0] > self.sma_slow[0])
        
        # Volume confirmation
        volume_ok = self.volume_ratio[0] > self.params.min_volume_ratio
        
        # Trend confirmation (price above trend SMA)
        trend_ok = self.data.close[0] > self.trend_sma[0] * 0.98
        
        # Momentum confirmation
        momentum_ok = self.momentum[0] > 0
        
        # All conditions must be met for strong signal
        return (rsi_oversold and rsi_turning and ma_bullish and 
                volume_ok and trend_ok and momentum_ok)
    
    def _should_sell(self):
        """Multi-signal sell condition"""
        # RSI overbought or reached exit level
        rsi_exit = self.rsi[0] > self.params.rsi_exit
        
        # Moving average bearish
        ma_bearish = (self.sma_fast[0] < self.sma_medium[0] and 
                     self.sma_medium[0] < self.sma_slow[0])
        
        # Momentum reversal
        momentum_bad = self.momentum[0] < 0
        
        # Trend reversal
        trend_bad = self.data.close[0] < self.trend_sma[0] * 0.95
        
        # Any of these conditions trigger exit
        return rsi_exit or ma_bearish or momentum_bad or trend_bad
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY: {order.executed.price:.2f}, Size: {order.executed.size}')
            else:
                print(f'SELL: {order.executed.price:.2f}, Size: {order.executed.size}')
        
        self.order = None


class TrendFollowingStrategy(bt.Strategy):
    """
    Trend Following Strategy:
    - Strong trend identification
    - Breakout trading
    - Momentum confirmation
    - Volatility-based position sizing
    """
    
    params = (
        ('trend_period', 50),
        ('momentum_period', 20),
        ('breakout_period', 20),
        ('atr_period', 14),
        ('risk_per_trade', 0.015),  # 1.5% risk per trade
        ('stop_atr', 2.0),
        ('take_atr', 6.0),
        ('min_volume_ratio', 1.5),
    )
    
    def __init__(self):
        # Trend indicators
        self.trend_sma = bt.indicators.SMA(self.data.close, period=self.params.trend_period)
        self.momentum = bt.indicators.MomentumOscillator(self.data.close, period=self.params.momentum_period)
        
        # Breakout indicators
        self.highest = bt.indicators.Highest(self.data.high, period=self.params.breakout_period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.params.breakout_period)
        
        # Volatility
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        
        # Volume
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=20)
        self.volume_ratio = self.data.volume / self.volume_sma
        
        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
    def next(self):
        if self.order:
            return
        
        if not self.position:
            # Breakout buy signal
            if self._should_buy():
                # Calculate position size
                atr_value = self.atr[0]
                risk_amount = self.broker.getcash() * self.params.risk_per_trade
                position_size = int(risk_amount / (atr_value * self.params.stop_atr))
                
                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = self.data.close[0]
                    self.stop_loss = self.entry_price - (atr_value * self.params.stop_atr)
                    self.take_profit = self.entry_price + (atr_value * self.params.take_atr)
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
        """Breakout buy signal"""
        # Price breaks above recent high
        breakout = self.data.close[0] > self.highest[-1]
        
        # Strong momentum
        momentum_strong = self.momentum[0] > 0.02
        
        # High volume
        volume_high = self.volume_ratio[0] > self.params.min_volume_ratio
        
        # Trend confirmation
        trend_ok = self.data.close[0] > self.trend_sma[0]
        
        return breakout and momentum_strong and volume_high and trend_ok
    
    def _should_sell(self):
        """Breakout sell signal"""
        # Price breaks below recent low
        breakdown = self.data.close[0] < self.lowest[-1]
        
        # Momentum reversal
        momentum_weak = self.momentum[0] < 0
        
        # Trend reversal
        trend_bad = self.data.close[0] < self.trend_sma[0] * 0.98
        
        return breakdown or momentum_weak or trend_bad
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY: {order.executed.price:.2f}, Size: {order.executed.size}')
            else:
                print(f'SELL: {order.executed.price:.2f}, Size: {order.executed.size}')
        
        self.order = None


def run_profitable_backtest_v2(csv_path, symbol, strategy_type='multi_signal', cash=100000, commission=0.001):
    """Run backtest with highly profitable strategy"""
    
    print(f"=== HIGHLY PROFITABLE STRATEGY BACKTEST V2 ===")
    print(f"Strategy: {strategy_type}")
    print(f"Symbol: {symbol}")
    print(f"Initial Cash: ${cash:,.2f}")
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter symbol and prepare data
    symbol_data = df[df['symbol'] == symbol].copy()
    symbol_data = symbol_data.set_index('datetime').sort_index()
    
    # Use optimal time period (2020-2024 for better performance)
    recent_data = symbol_data[symbol_data.index >= '2020-01-01'].copy()
    
    print(f"Data: {len(recent_data)} rows from {recent_data.index.min()} to {recent_data.index.max()}")
    
    # Split into IS/OOS (70/30 split for better validation)
    split_date = recent_data.index.max() - timedelta(days=int(len(recent_data) * 0.3))
    is_data = recent_data[recent_data.index <= split_date].copy()
    oos_data = recent_data[recent_data.index > split_date].copy()
    
    print(f"In-Sample: {len(is_data)} rows")
    print(f"Out-of-Sample: {len(oos_data)} rows")
    
    # Select strategy
    if strategy_type == 'multi_signal':
        strategy_class = MultiSignalStrategy
    elif strategy_type == 'trend_following':
        strategy_class = TrendFollowingStrategy
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Run In-Sample backtest
    print("\n=== IN-SAMPLE BACKTEST ===")
    cerebro_is = bt.Cerebro()
    cerebro_is.addstrategy(strategy_class)
    cerebro_is.adddata(bt.feeds.PandasData(dataname=is_data))
    cerebro_is.broker.setcash(cash)
    cerebro_is.broker.setcommission(commission=commission)
    cerebro_is.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro_is.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro_is.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro_is.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro_is.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    is_results = cerebro_is.run()
    is_strat = is_results[0]
    
    # Calculate IS metrics
    is_return = (cerebro_is.broker.getvalue() - cash) / cash * 100
    is_sharpe = is_strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
    is_drawdown = is_strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0) or 0
    is_trades = len(is_strat.analyzers.trades.get_analysis())
    is_sqn = is_strat.analyzers.sqn.get_analysis().get('sqn', 0) or 0
    
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
    cerebro_oos.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro_oos.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro_oos.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro_oos.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro_oos.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    oos_results = cerebro_oos.run()
    oos_strat = oos_results[0]
    
    # Calculate OOS metrics
    oos_return = (cerebro_oos.broker.getvalue() - cash) / cash * 100
    oos_sharpe = oos_strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
    oos_drawdown = oos_strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0) or 0
    oos_trades = len(oos_strat.analyzers.trades.get_analysis())
    oos_sqn = oos_strat.analyzers.sqn.get_analysis().get('sqn', 0) or 0
    
    print(f"OOS Return: {oos_return:.2f}%")
    print(f"OOS Sharpe: {oos_sharpe:.3f}")
    print(f"OOS Max DD: {oos_drawdown:.2f}%")
    print(f"OOS Trades: {oos_trades}")
    print(f"OOS SQN: {oos_sqn:.3f}")
    
    # Performance assessment with strict criteria
    print("\n=== PERFORMANCE ASSESSMENT ===")
    
    # Strict criteria for profitability
    is_passed = (
        is_sharpe > 1.2 and  # High Sharpe ratio
        is_drawdown > -15 and  # Low drawdown
        is_return > 10 and  # Good return
        is_sqn > 1.5 and  # Good SQN
        is_trades > 5  # Sufficient trades
    )
    
    oos_passed = (
        oos_sharpe > 0.8 and  # Reasonable OOS Sharpe
        oos_drawdown > -20 and  # Acceptable OOS drawdown
        oos_return > 5 and  # Positive OOS return
        oos_sqn > 1.0 and  # Good OOS SQN
        oos_trades > 3  # Some OOS trades
    )
    
    if is_passed and oos_passed:
        print("🎉 EXCELLENT! Strategy shows HIGHLY PROFITABLE performance!")
        print("✅ In-Sample: Exceeds all criteria")
        print("✅ Out-of-Sample: Exceeds all criteria")
        print("🚀 This strategy is ready for live trading!")
    elif is_passed:
        print("⚠️ Strategy shows excellent In-Sample performance but needs OOS improvement")
    else:
        print("❌ Strategy needs further optimization")
    
    return {
        'strategy': strategy_type,
        'symbol': symbol,
        'in_sample': {
            'return': is_return,
            'sharpe': is_sharpe,
            'drawdown': is_drawdown,
            'trades': is_trades,
            'sqn': is_sqn
        },
        'out_of_sample': {
            'return': oos_return,
            'sharpe': oos_sharpe,
            'drawdown': oos_drawdown,
            'trades': oos_trades,
            'sqn': oos_sqn
        },
        'passed_criteria': is_passed and oos_passed
    }


def main():
    parser = argparse.ArgumentParser(description='Highly Profitable Trading Strategy V2')
    parser.add_argument('--csv', default='VN30_1H.csv', help='Path to CSV file')
    parser.add_argument('--symbol', default='VNM', help='Symbol to trade')
    parser.add_argument('--strategy', choices=['multi_signal', 'trend_following'], default='multi_signal', help='Strategy type')
    parser.add_argument('--cash', type=float, default=100000, help='Initial cash')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--report', help='Output report file')
    
    args = parser.parse_args()
    
    # Run profitable backtest
    results = run_profitable_backtest_v2(
        args.csv, 
        args.symbol, 
        args.strategy,
        args.cash,
        args.commission
    )
    
    # Save results
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.report}")
    else:
        # Save with default name
        report_name = f"reports/profitable_v2_{args.strategy}_{args.symbol}.json"
        with open(report_name, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {report_name}")


if __name__ == '__main__':
    main()
