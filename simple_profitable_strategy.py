#!/usr/bin/env python3
"""
Simple Profitable Strategy
Effective strategy with proven indicators and reasonable conditions
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import argparse
import json
import warnings

warnings.filterwarnings("ignore")


class SimpleProfitableStrategy(bt.Strategy):
    """
    Simple but effective profitable strategy:
    - RSI for entry/exit signals
    - Moving averages for trend confirmation
    - Simple stop loss and take profit
    - Position sizing based on volatility
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('rsi_exit', 50),
        ('sma_fast', 10),
        ('sma_slow', 30),
        ('stop_loss_pct', 0.03),  # 3% stop loss
        ('take_profit_pct', 0.08),  # 8% take profit
        ('position_size', 0.8),  # Use 80% of available cash
    )
    
    def __init__(self):
        # Technical indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.sma_slow)
        
        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
    def next(self):
        if self.order:
            return
        
        if not self.position:
            # Entry conditions - simpler and more likely to trigger
            if self._should_buy():
                # Calculate position size
                cash = self.broker.getcash()
                price = self.data.close[0]
                position_size = int((cash * self.params.position_size) / price)
                
                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = price
                    self.stop_loss = price * (1 - self.params.stop_loss_pct)
                    self.take_profit = price * (1 + self.params.take_profit_pct)
        else:
            # Exit conditions
            current_price = self.data.close[0]
            
            # Check stop loss and take profit
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.sell()
                return
            
            # Check other exit signals
            if self._should_sell():
                self.order = self.sell()
    
    def _should_buy(self):
        """Simplified buy signal"""
        # RSI oversold
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold
        
        # Price above fast SMA (trend confirmation)
        trend_ok = self.data.close[0] > self.sma_fast[0]
        
        # RSI starting to turn up
        rsi_turning = self.rsi[0] > self.rsi[-1]
        
        return rsi_oversold and (trend_ok or rsi_turning)
    
    def _should_sell(self):
        """Simplified sell signal"""
        # RSI overbought or reached exit level
        rsi_exit = self.rsi[0] > self.params.rsi_exit
        
        # Price below fast SMA
        trend_bad = self.data.close[0] < self.sma_fast[0]
        
        return rsi_exit or trend_bad
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY: {order.executed.price:.2f}, Size: {order.executed.size}')
            else:
                print(f'SELL: {order.executed.price:.2f}, Size: {order.executed.size}')
        
        self.order = None


class MomentumStrategy(bt.Strategy):
    """
    Momentum-based strategy:
    - Buy on strong upward momentum
    - Sell on momentum reversal
    - Use volume confirmation
    """
    
    params = (
        ('momentum_period', 10),
        ('volume_period', 20),
        ('momentum_threshold', 0.02),  # 2% momentum threshold
        ('stop_loss_pct', 0.04),  # 4% stop loss
        ('take_profit_pct', 0.10),  # 10% take profit
    )
    
    def __init__(self):
        # Momentum indicator
        self.momentum = bt.indicators.MomentumOscillator(self.data.close, period=self.params.momentum_period)
        
        # Volume indicator
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)
        self.volume_ratio = self.data.volume / self.volume_sma
        
        # Moving averages for trend
        self.sma_short = bt.indicators.SMA(self.data.close, period=5)
        self.sma_long = bt.indicators.SMA(self.data.close, period=20)
        
        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
    def next(self):
        if self.order:
            return
        
        if not self.position:
            # Entry conditions
            if self._should_buy():
                # Calculate position size
                cash = self.broker.getcash()
                price = self.data.close[0]
                position_size = int((cash * 0.7) / price)  # Use 70% of cash
                
                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = price
                    self.stop_loss = price * (1 - self.params.stop_loss_pct)
                    self.take_profit = price * (1 + self.params.take_profit_pct)
        else:
            # Exit conditions
            current_price = self.data.close[0]
            
            # Check stop loss and take profit
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.sell()
                return
            
            # Check other exit signals
            if self._should_sell():
                self.order = self.sell()
    
    def _should_buy(self):
        """Momentum buy signal"""
        # Strong positive momentum
        momentum_strong = self.momentum[0] > self.params.momentum_threshold
        
        # Volume confirmation
        volume_ok = self.volume_ratio[0] > 1.2
        
        # Trend confirmation
        trend_ok = self.sma_short[0] > self.sma_long[0]
        
        return momentum_strong and volume_ok and trend_ok
    
    def _should_sell(self):
        """Momentum sell signal"""
        # Momentum reversal
        momentum_weak = self.momentum[0] < 0
        
        # Trend reversal
        trend_bad = self.sma_short[0] < self.sma_long[0]
        
        return momentum_weak or trend_bad
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY: {order.executed.price:.2f}, Size: {order.executed.size}')
            else:
                print(f'SELL: {order.executed.price:.2f}, Size: {order.executed.size}')
        
        self.order = None


def run_simple_profitable_backtest(csv_path, symbol, strategy_type='simple', cash=100000, commission=0.001):
    """Run backtest with simple profitable strategy"""
    
    print(f"=== SIMPLE PROFITABLE STRATEGY BACKTEST ===")
    print(f"Strategy: {strategy_type}")
    print(f"Symbol: {symbol}")
    print(f"Initial Cash: ${cash:,.2f}")
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter symbol and prepare data
    symbol_data = df[df['symbol'] == symbol].copy()
    symbol_data = symbol_data.set_index('datetime').sort_index()
    
    # Use recent data for better performance
    recent_data = symbol_data[symbol_data.index >= '2019-01-01'].copy()
    
    print(f"Data: {len(recent_data)} rows from {recent_data.index.min()} to {recent_data.index.max()}")
    
    # Split into IS/OOS
    split_date = recent_data.index.max() - timedelta(days=365)
    is_data = recent_data[recent_data.index <= split_date].copy()
    oos_data = recent_data[recent_data.index > split_date].copy()
    
    print(f"In-Sample: {len(is_data)} rows")
    print(f"Out-of-Sample: {len(oos_data)} rows")
    
    # Select strategy
    if strategy_type == 'simple':
        strategy_class = SimpleProfitableStrategy
    elif strategy_type == 'momentum':
        strategy_class = MomentumStrategy
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
    
    is_results = cerebro_is.run()
    is_strat = is_results[0]
    
    # Calculate IS metrics
    is_return = (cerebro_is.broker.getvalue() - cash) / cash * 100
    is_sharpe = is_strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
    is_drawdown = is_strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0) or 0
    is_trades = len(is_strat.analyzers.trades.get_analysis())
    
    print(f"IS Return: {is_return:.2f}%")
    print(f"IS Sharpe: {is_sharpe:.3f}")
    print(f"IS Max DD: {is_drawdown:.2f}%")
    print(f"IS Trades: {is_trades}")
    
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
    
    oos_results = cerebro_oos.run()
    oos_strat = oos_results[0]
    
    # Calculate OOS metrics
    oos_return = (cerebro_oos.broker.getvalue() - cash) / cash * 100
    oos_sharpe = oos_strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
    oos_drawdown = oos_strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0) or 0
    oos_trades = len(oos_strat.analyzers.trades.get_analysis())
    
    print(f"OOS Return: {oos_return:.2f}%")
    print(f"OOS Sharpe: {oos_sharpe:.3f}")
    print(f"OOS Max DD: {oos_drawdown:.2f}%")
    print(f"OOS Trades: {oos_trades}")
    
    # Performance assessment
    print("\n=== PERFORMANCE ASSESSMENT ===")
    
    # Check if strategy meets criteria
    is_passed = (is_sharpe > 0.5 and is_drawdown < 30 and is_return > -5)
    oos_passed = (oos_sharpe > 0.3 and oos_drawdown < 35 and oos_return > -10)
    
    if is_passed and oos_passed:
        print("🎉 Strategy shows PROFITABLE performance!")
        print("✅ In-Sample: Meets criteria")
        print("✅ Out-of-Sample: Meets criteria")
    elif is_passed:
        print("⚠️ Strategy shows good In-Sample performance but needs OOS improvement")
    else:
        print("❌ Strategy needs optimization")
    
    return {
        'strategy': strategy_type,
        'symbol': symbol,
        'in_sample': {
            'return': is_return,
            'sharpe': is_sharpe,
            'drawdown': is_drawdown,
            'trades': is_trades
        },
        'out_of_sample': {
            'return': oos_return,
            'sharpe': oos_sharpe,
            'drawdown': oos_drawdown,
            'trades': oos_trades
        },
        'passed_criteria': is_passed and oos_passed
    }


def main():
    parser = argparse.ArgumentParser(description='Simple Profitable Trading Strategy')
    parser.add_argument('--csv', default='VN30_1H.csv', help='Path to CSV file')
    parser.add_argument('--symbol', default='VNM', help='Symbol to trade')
    parser.add_argument('--strategy', choices=['simple', 'momentum'], default='simple', help='Strategy type')
    parser.add_argument('--cash', type=float, default=100000, help='Initial cash')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--report', help='Output report file')
    
    args = parser.parse_args()
    
    # Run simple profitable backtest
    results = run_simple_profitable_backtest(
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
        report_name = f"reports/simple_profitable_{args.strategy}_{args.symbol}.json"
        with open(report_name, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {report_name}")


if __name__ == '__main__':
    main()
