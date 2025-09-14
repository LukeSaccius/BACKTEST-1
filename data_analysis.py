#!/usr/bin/env python3
"""
Data Analysis for Strategy Development
Analyze market patterns to find profitable opportunities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_market_data(csv_path):
    """Analyze market data to find profitable patterns"""
    
    print("=== MARKET DATA ANALYSIS ===")
    
    # Load data
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    # Get unique symbols
    symbols = df['symbol'].unique()
    print(f"Found {len(symbols)} symbols: {symbols[:10]}...")
    
    # Analyze each symbol
    symbol_analysis = {}
    
    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol].copy()
        
        if len(symbol_data) < 100:  # Skip symbols with insufficient data
            continue
            
        # Calculate returns
        symbol_data['returns'] = symbol_data['close'].pct_change()
        symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
        
        # Calculate metrics
        total_return = (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1) * 100
        annual_return = total_return * (252 / len(symbol_data))
        volatility = symbol_data['returns'].std() * np.sqrt(252) * 100
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + symbol_data['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calculate trend strength
        symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
        symbol_data['sma_50'] = symbol_data['close'].rolling(50).mean()
        trend_strength = ((symbol_data['close'] - symbol_data['sma_50']) / symbol_data['sma_50']).mean() * 100
        
        # Calculate volatility clustering
        volatility_clustering = symbol_data['returns'].rolling(20).std().autocorr()
        
        symbol_analysis[symbol] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trend_strength': trend_strength,
            'volatility_clustering': volatility_clustering,
            'data_points': len(symbol_data),
            'date_range': f"{symbol_data.index.min().date()} to {symbol_data.index.max().date()}"
        }
    
    # Convert to DataFrame for easier analysis
    analysis_df = pd.DataFrame(symbol_analysis).T
    
    # Filter for good candidates
    good_candidates = analysis_df[
        (analysis_df['annual_return'] > 5) &  # Positive annual return
        (analysis_df['sharpe_ratio'] > 0.3) &  # Reasonable Sharpe ratio
        (analysis_df['max_drawdown'] > -40) &  # Not too volatile
        (analysis_df['data_points'] > 500)  # Sufficient data
    ].sort_values('sharpe_ratio', ascending=False)
    
    print(f"\nFound {len(good_candidates)} good candidates:")
    print(good_candidates.head(10))
    
    return good_candidates, symbol_analysis

def analyze_time_periods(csv_path, symbol):
    """Analyze different time periods for the best performing ones"""
    
    print(f"\n=== TIME PERIOD ANALYSIS FOR {symbol} ===")
    
    # Load data
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['symbol'] == symbol].set_index('datetime').sort_index()
    
    # Define time periods
    periods = {
        '2018-2020': ('2018-01-01', '2020-12-31'),
        '2020-2022': ('2020-01-01', '2022-12-31'),
        '2022-2024': ('2022-01-01', '2024-12-31'),
        '2019-2024': ('2019-01-01', '2024-12-31'),
        '2020-2024': ('2020-01-01', '2024-12-31'),
        '2021-2024': ('2021-01-01', '2024-12-31')
    }
    
    period_analysis = {}
    
    for period_name, (start_date, end_date) in periods.items():
        period_data = df[(df.index >= start_date) & (df.index <= end_date)].copy()
        
        if len(period_data) < 100:
            continue
            
        # Calculate metrics
        period_data['returns'] = period_data['close'].pct_change()
        total_return = (period_data['close'].iloc[-1] / period_data['close'].iloc[0] - 1) * 100
        annual_return = total_return * (252 / len(period_data))
        volatility = period_data['returns'].std() * np.sqrt(252) * 100
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + period_data['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        period_analysis[period_name] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'data_points': len(period_data)
        }
    
    period_df = pd.DataFrame(period_analysis).T
    print(period_df.sort_values('sharpe_ratio', ascending=False))
    
    return period_df

def find_best_parameters(csv_path, symbol):
    """Find the best parameters for technical indicators"""
    
    print(f"\n=== PARAMETER OPTIMIZATION FOR {symbol} ===")
    
    # Load data
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['symbol'] == symbol].set_index('datetime').sort_index()
    
    # Use recent data for optimization
    recent_data = df[df.index >= '2020-01-01'].copy()
    
    # Test different RSI parameters
    rsi_results = []
    for rsi_period in [10, 14, 20]:
        for rsi_oversold in [20, 25, 30]:
            for rsi_overbought in [70, 75, 80]:
                # Calculate RSI signals
                recent_data['rsi'] = calculate_rsi(recent_data['close'], rsi_period)
                recent_data['rsi_signal'] = 0
                recent_data.loc[recent_data['rsi'] < rsi_oversold, 'rsi_signal'] = 1  # Buy
                recent_data.loc[recent_data['rsi'] > rsi_overbought, 'rsi_signal'] = -1  # Sell
                
                # Calculate returns
                recent_data['strategy_returns'] = recent_data['rsi_signal'].shift(1) * recent_data['close'].pct_change()
                total_return = recent_data['strategy_returns'].sum() * 100
                sharpe = recent_data['strategy_returns'].mean() / recent_data['strategy_returns'].std() * np.sqrt(252) if recent_data['strategy_returns'].std() > 0 else 0
                
                rsi_results.append({
                    'rsi_period': rsi_period,
                    'rsi_oversold': rsi_oversold,
                    'rsi_overbought': rsi_overbought,
                    'total_return': total_return,
                    'sharpe': sharpe
                })
    
    rsi_df = pd.DataFrame(rsi_results)
    best_rsi = rsi_df.loc[rsi_df['sharpe'].idxmax()]
    
    print(f"Best RSI parameters: {best_rsi['rsi_period']}, {best_rsi['rsi_oversold']}, {best_rsi['rsi_overbought']}")
    print(f"Best RSI Sharpe: {best_rsi['sharpe']:.3f}")
    
    return best_rsi

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    # Analyze market data
    good_candidates, symbol_analysis = analyze_market_data('VN30_1H.csv')
    
    if len(good_candidates) > 0:
        # Get the best symbol
        best_symbol = good_candidates.index[0]
        print(f"\nBest symbol: {best_symbol}")
        
        # Analyze time periods for the best symbol
        period_analysis = analyze_time_periods('VN30_1H.csv', best_symbol)
        
        # Find best parameters
        best_params = find_best_parameters('VN30_1H.csv', best_symbol)
        
        # Save analysis results
        results = {
            'best_symbol': best_symbol,
            'symbol_analysis': symbol_analysis,
            'period_analysis': period_analysis.to_dict(),
            'best_parameters': best_params.to_dict()
        }
        
        import json
        with open('market_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nAnalysis results saved to market_analysis_results.json")
    else:
        print("No good candidates found. Using default symbol VNM.")
        best_symbol = 'VNM'
        period_analysis = analyze_time_periods('VN30_1H.csv', best_symbol)
        best_params = find_best_parameters('VN30_1H.csv', best_symbol)

if __name__ == '__main__':
    main()
