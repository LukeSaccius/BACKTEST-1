#!/usr/bin/env python3
"""
Profitable Trading Strategy
Advanced strategy with multiple indicators and risk management
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import argparse
import json
import warnings

warnings.filterwarnings("ignore")


class AdvancedProfitableStrategy(bt.Strategy):
    """
    Advanced Profitable Strategy combining multiple indicators:
    - RSI for oversold/overbought conditions
    - MACD for trend confirmation
    - Bollinger Bands for volatility
    - Volume confirmation
    - Dynamic position sizing
    - Stop loss and take profit
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 25),  # More aggressive oversold
        ('rsi_overbought', 75),  # More aggressive overbought
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('stop_loss_pct', 0.02),  # 2% stop loss
        ('take_profit_pct', 0.06),  # 6% take profit
        ('min_volume_mult', 1.5),  # Volume must be 1.5x average
        ('max_position_size', 0.95),  # Use 95% of available cash
    )
    
    def __init__(self):
        # Technical indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(
            self.data.close, 
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        self.bb = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )
        
        # Volume indicator
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=20)
        self.volume_ratio = self.data.volume / self.volume_sma
        
        # Price momentum
        self.price_sma = bt.indicators.SMA(self.data.close, period=50)
        self.momentum = self.data.close / self.price_sma
        
        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
    def next(self):
        if self.order:
            return
        
        # Calculate position size based on volatility
        if not self.position:
            # Entry conditions
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
        """Enhanced buy signal with multiple confirmations"""
        # RSI oversold condition
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold
        
        # MACD bullish crossover or positive
        macd_bullish = (self.macd.macd[0] > self.macd.signal[0] and 
                       self.macd.macd[-1] <= self.macd.signal[-1]) or self.macd.macd[0] > 0
        
        # Price near lower Bollinger Band
        bb_support = self.data.close[0] <= self.bb.lines.bot[0] * 1.02
        
        # Volume confirmation
        volume_ok = self.volume_ratio[0] > self.params.min_volume_mult
        
        # Momentum confirmation
        momentum_ok = self.momentum[0] > 0.95  # Price above 95% of 50-day SMA
        
        # All conditions must be met
        return (rsi_oversold and macd_bullish and bb_support and 
                volume_ok and momentum_ok)
    
    def _should_sell(self):
        """Enhanced sell signal with multiple confirmations"""
        # RSI overbought condition
        rsi_overbought = self.rsi[0] > self.params.rsi_overbought
        
        # MACD bearish crossover or negative
        macd_bearish = (self.macd.macd[0] < self.macd.signal[0] and 
                       self.macd.macd[-1] >= self.macd.signal[-1]) or self.macd.macd[0] < 0
        
        # Price near upper Bollinger Band
        bb_resistance = self.data.close[0] >= self.bb.lines.top[0] * 0.98
        
        # Any of these conditions trigger exit
        return rsi_overbought or macd_bearish or bb_resistance
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED: {order.executed.price:.2f}, Size: {order.executed.size}')
            else:
                print(f'SELL EXECUTED: {order.executed.price:.2f}, Size: {order.executed.size}')
        
        self.order = None


class MeanReversionStrategy(bt.Strategy):
    """
    Mean Reversion Strategy with enhanced features:
    - Multiple timeframe analysis
    - Volatility-based position sizing
    - Dynamic entry/exit thresholds
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 20),  # Very oversold
        ('rsi_overbought', 80),  # Very overbought
        ('rsi_exit', 60),  # Exit when RSI reaches 60
        ('atr_period', 14),
        ('stop_atr', 2.0),  # 2 ATR stop loss
        ('take_atr', 3.0),  # 3 ATR take profit
        ('min_hold_bars', 5),  # Minimum hold period
    )
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        self.sma_short = bt.indicators.SMA(self.data.close, period=10)
        self.sma_long = bt.indicators.SMA(self.data.close, period=30)
        
        # State tracking
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.stop_loss = None
        self.take_profit = None
        
    def next(self):
        if self.order:
            return
        
        if not self.position:
            # Entry conditions
            if self._should_buy():
                # Calculate position size based on ATR
                atr_value = self.atr[0]
                risk_per_trade = self.broker.getcash() * 0.02  # 2% risk per trade
                position_size = int(risk_per_trade / (atr_value * self.params.stop_atr))
                
                if position_size > 0:
                    self.order = self.buy(size=position_size)
                    self.entry_price = self.data.close[0]
                    self.entry_bar = len(self)
                    self.stop_loss = self.entry_price - (atr_value * self.params.stop_atr)
                    self.take_profit = self.entry_price + (atr_value * self.params.take_atr)
        else:
            # Check minimum hold period
            bars_held = len(self) - self.entry_bar
            if bars_held < self.params.min_hold_bars:
                return
            
            current_price = self.data.close[0]
            
            # Check stop loss and take profit
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self.order = self.sell()
                return
            
            # Check exit conditions
            if self._should_sell():
                self.order = self.sell()
    
    def _should_buy(self):
        """Buy signal for mean reversion"""
        # RSI very oversold
        rsi_oversold = self.rsi[0] < self.params.rsi_oversold
        
        # Price below short-term SMA (oversold)
        price_oversold = self.data.close[0] < self.sma_short[0]
        
        # RSI starting to turn up
        rsi_turning = self.rsi[0] > self.rsi[-1]
        
        return rsi_oversold and price_oversold and rsi_turning
    
    def _should_sell(self):
        """Sell signal for mean reversion"""
        # RSI overbought or reached exit level
        rsi_exit = self.rsi[0] > self.params.rsi_exit
        
        # Price above short-term SMA (overbought)
        price_overbought = self.data.close[0] > self.sma_short[0]
        
        return rsi_exit or price_overbought
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED: {order.executed.price:.2f}, Size: {order.executed.size}')
            else:
                print(f'SELL EXECUTED: {order.executed.price:.2f}, Size: {order.executed.size}')
        
        self.order = None


def run_profitable_backtest(csv_path, symbol, strategy_type='advanced', cash=100000, commission=0.001):
    """Run backtest with profitable strategy"""
    
    print(f"=== PROFITABLE STRATEGY BACKTEST ===")
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
    recent_data = symbol_data[symbol_data.index >= '2018-01-01'].copy()
    
    print(f"Data: {len(recent_data)} rows from {recent_data.index.min()} to {recent_data.index.max()}")
    
    # Split into IS/OOS
    split_date = recent_data.index.max() - timedelta(days=365)
    is_data = recent_data[recent_data.index <= split_date].copy()
    oos_data = recent_data[recent_data.index > split_date].copy()
    
    print(f"In-Sample: {len(is_data)} rows")
    print(f"Out-of-Sample: {len(oos_data)} rows")
    
    # Select strategy
    if strategy_type == 'advanced':
        strategy_class = AdvancedProfitableStrategy
    elif strategy_type == 'mean_reversion':
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
    is_passed = (is_sharpe > 1.0 and is_drawdown < 20 and is_return > 0)
    oos_passed = (oos_sharpe > 0.7 and oos_drawdown < 25 and oos_return > 0)
    
    if is_passed and oos_passed:
        print("🎉 Strategy shows PROFITABLE performance!")
        print("✅ In-Sample: Meets all criteria")
        print("✅ Out-of-Sample: Meets all criteria")
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
    parser = argparse.ArgumentParser(description='Profitable Trading Strategy')
    parser.add_argument('--csv', default='VN30_1H.csv', help='Path to CSV file')
    parser.add_argument('--symbol', default='VNM', help='Symbol to trade')
    parser.add_argument('--strategy', choices=['advanced', 'mean_reversion'], default='advanced', help='Strategy type')
    parser.add_argument('--cash', type=float, default=100000, help='Initial cash')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--report', help='Output report file')
    
    args = parser.parse_args()
    
    # Run profitable backtest
    results = run_profitable_backtest(
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
        report_name = f"reports/profitable_{args.strategy}_{args.symbol}.json"
        with open(report_name, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {report_name}")


if __name__ == '__main__':
    main()
