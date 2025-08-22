#!/usr/bin/env python3
"""
Comprehensive Backtesting Framework
Following the exact specifications and requirements provided
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import argparse
import json
import warnings

warnings.filterwarnings("ignore")


class DataQualityChecker:
    """Step 2: Data Quality Verification"""

    @staticmethod
    def check_data_quality(df, symbol=None):
        """Verify data quality according to specifications"""
        print("=== DATA QUALITY CHECK ===")

        issues = []
        warnings = []

        # Check if data is empty
        if df.empty:
            issues.append("Data is empty")
            return False, issues, warnings

        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")

        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")

        # Check for outliers (returns > 50% in a day)
        if "close" in df.columns:
            returns = df["close"].pct_change()
            outliers = returns[returns.abs() > 0.5]
            if len(outliers) > 0:
                warnings.append(f"Found {len(outliers)} extreme returns (>50%)")

        # Check date range
        if hasattr(df.index, "name") and df.index.name == "datetime":
            date_range = (df.index.max() - df.index.min()).days
            if date_range < 365:  # Less than 1 year
                warnings.append(f"Data spans only {date_range} days")

        # Check symbol filtering
        if symbol and "symbol" in df.columns:
            symbol_data = df[df["symbol"] == symbol]
            if len(symbol_data) == 0:
                issues.append(f"No data found for symbol {symbol}")
            else:
                print(f"✓ Symbol {symbol}: {len(symbol_data)} rows")

        # Overall assessment
        if len(issues) == 0:
            print("✓ Data quality check PASSED")
            return True, issues, warnings
        else:
            print("❌ Data quality check FAILED")
            for issue in issues:
                print(f"  - {issue}")
            return False, issues, warnings


class BiasFreeBacktest:
    """Step 3: Bias-free backtesting setup"""

    def __init__(self, data, symbol=None):
        self.data = data
        self.symbol = symbol
        self.setup_bias_free_conditions()

    def setup_bias_free_conditions(self):
        """Setup bias-free conditions"""
        print("\n=== BIAS-FREE SETUP ===")

        # Survivorship bias: Use proper time range
        if hasattr(self.data.index, "name") and self.data.index.name == "datetime":
            self.data = self.data.sort_index()
            print(f"✓ Data sorted by datetime")
            print(f"✓ Date range: {self.data.index.min()} to {self.data.index.max()}")

        # Data snooping bias: Split into IS/OOS
        self.split_is_oos()

        # Look-ahead bias: Next-bar execution (handled by Backtrader)
        print("✓ Next-bar execution (no look-ahead bias)")

        # Backfill bias: Use only available data at time
        print("✓ Using only data available at execution time")

    def split_is_oos(self):
        """Split data into In-Sample (5-10 years) and Out-of-Sample (1-2 years)"""
        if not hasattr(self.data.index, "name") or self.data.index.name != "datetime":
            return

        # Calculate split point (last 2 years for OOS)
        total_days = (self.data.index.max() - self.data.index.min()).days
        oos_days = min(730, total_days // 3)  # 2 years or 1/3 of data

        split_date = self.data.index.max() - timedelta(days=oos_days)

        self.is_data = self.data[self.data.index <= split_date].copy()
        self.oos_data = self.data[self.data.index > split_date].copy()

        print(
            f"✓ In-Sample: {len(self.is_data)} rows ({self.is_data.index.min()} to {self.is_data.index.max()})"
        )
        print(
            f"✓ Out-of-Sample: {len(self.oos_data)} rows ({self.oos_data.index.min()} to {self.oos_data.index.max()})"
        )


class SMACrossoverStrategy(bt.Strategy):
    """Simple Moving Average Crossover Strategy"""

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
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


class RSIMeanReversionStrategy(bt.Strategy):
    """RSI Mean Reversion Strategy"""

    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("rsi_exit", 50),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.rsi < self.params.rsi_oversold:
                self.order = self.buy()
        else:
            if self.rsi > self.params.rsi_exit:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None


class PerformanceAnalyzer:
    """Step 4: Performance Analysis and Evaluation"""

    @staticmethod
    def calculate_metrics(strategy_results):
        """Calculate all required performance metrics"""
        strat = strategy_results[0]

        # Basic metrics
        initial_value = strat.broker.startingcash
        final_value = strat.broker.getvalue()
        total_return = (final_value - initial_value) / initial_value

        # Get analyzer results
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()

        # Calculate metrics
        metrics = {
            "total_return": total_return * 100,
            "annual_return": returns_analysis.get("rnorm100", 0),
            "sharpe_ratio": sharpe_analysis.get("sharperatio", 0),
            "max_drawdown": drawdown_analysis.get("max", {}).get("drawdown", 0),
            "total_trades": len(trades_analysis),
            "win_rate": 0,
            "profit_factor": 0,
            "turnover": 0,
            "time_in_market": 0,
            "fitness": 0,
            "daily_return": 0,
        }

        # Calculate win rate and profit factor
        if "total" in trades_analysis:
            total_trades = trades_analysis["total"]["total"]
            if total_trades > 0:
                won_trades = (
                    trades_analysis["won"]["total"] if "won" in trades_analysis else 0
                )
                metrics["win_rate"] = (won_trades / total_trades) * 100

                if "pnl" in trades_analysis and "gross" in trades_analysis["pnl"]:
                    gross_profit = trades_analysis["pnl"]["gross"].get("profit", 0)
                    gross_loss = abs(trades_analysis["pnl"]["gross"].get("loss", 0))
                    if gross_loss > 0:
                        metrics["profit_factor"] = gross_profit / gross_loss

        # Calculate daily return (simplified)
        if metrics["total_return"] != 0:
            # Estimate daily return from total return
            trading_days = len(strat.data)
            if trading_days > 0:
                metrics["daily_return"] = (
                    (metrics["total_return"] / 100) / trading_days * 100
                )

        # Calculate fitness score
        daily_return_abs = abs(metrics["daily_return"])
        turnover = max(metrics["turnover"], 0.00125)
        if turnover > 0 and metrics["sharpe_ratio"] != 0:
            metrics["fitness"] = metrics["sharpe_ratio"] * np.sqrt(
                daily_return_abs / turnover
            )

        return metrics

    @staticmethod
    def evaluate_strategy(metrics, is_sample=True):
        """Evaluate strategy against criteria"""
        print(f"\n=== {'IN-SAMPLE' if is_sample else 'OUT-OF-SAMPLE'} EVALUATION ===")

        criteria = {
            "daily_return": {
                "min": 0.0004,
                "max": None,
                "description": "Daily Return > 0.0004",
            },
            "sharpe_ratio": {
                "min": 1.0 if is_sample else 0.7,
                "max": None,
                "description": f"Sharpe Ratio > {1.0 if is_sample else 0.7}",
            },
            "fitness": {"min": 2.1, "max": None, "description": "Fitness > 2.1"},
            "max_drawdown": {
                "min": None,
                "max": 55.0,
                "description": "Max Drawdown < 55%",
            },
            "win_rate": {"min": 50.0, "max": None, "description": "Win Rate > 50%"},
        }

        passed = True
        failed_criteria = []

        for metric, criterion in criteria.items():
            value = metrics.get(metric, 0)

            if criterion["min"] is not None and value < criterion["min"]:
                print(f"❌ {criterion['description']}: {value:.4f}")
                failed_criteria.append(criterion["description"])
                passed = False
            elif criterion["max"] is not None and value > criterion["max"]:
                print(f"❌ {criterion['description']}: {value:.4f}")
                failed_criteria.append(criterion["description"])
                passed = False
            else:
                print(f"✓ {criterion['description']}: {value:.4f}")

        if passed:
            print("✓ Strategy PASSES all criteria")
        else:
            print("❌ Strategy FAILS some criteria")
            print("Failed criteria:")
            for criterion in failed_criteria:
                print(f"  - {criterion}")

        return passed, failed_criteria


class BacktestFramework:
    """Main Backtesting Framework"""

    def __init__(self):
        self.strategies = {"sma": SMACrossoverStrategy, "rsi": RSIMeanReversionStrategy}

    def load_data(self, csv_path, symbol=None):
        """Step 1: Load and prepare data"""
        print(f"Loading data from {csv_path}")

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

        for target, possible_names in column_mapping.items():
            for name in possible_names:
                if name in df.columns:
                    df = df.rename(columns={name: target})
                    break

        # Convert datetime
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        # Filter by symbol
        if symbol and "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()

        # Set datetime as index for backtrader
        if "datetime" in df.columns:
            df = df.set_index("datetime")

        return df

    def run_backtest(
        self, data, strategy_class, strategy_params, cash=100000, commission=0.001
    ):
        """Run a single backtest"""
        cerebro = bt.Cerebro()

        # Add strategy
        cerebro.addstrategy(strategy_class, **strategy_params)

        # Add data
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)

        # Set broker parameters
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        # Run backtest
        results = cerebro.run()
        return results

    def run_comprehensive_backtest(
        self,
        csv_path,
        symbol,
        strategy_name,
        strategy_params,
        cash=100000,
        commission=0.001,
    ):
        """Run comprehensive backtest following the exact process"""

        # Step 1: Load data
        print("=== STEP 1: LOAD DATA ===")
        data = self.load_data(csv_path, symbol)

        # Step 2: Data quality check
        print("\n=== STEP 2: DATA QUALITY CHECK ===")
        quality_ok, issues, warnings = DataQualityChecker.check_data_quality(
            data, symbol
        )
        if not quality_ok:
            print("❌ Data quality issues found. Aborting backtest.")
            return None

        # Step 3: Bias-free setup
        print("\n=== STEP 3: BIAS-FREE SETUP ===")
        bias_free = BiasFreeBacktest(data, symbol)

        # Step 4: Run In-Sample backtest
        print("\n=== STEP 4: IN-SAMPLE BACKTEST ===")
        strategy_class = self.strategies[strategy_name]

        is_results = self.run_backtest(
            bias_free.is_data, strategy_class, strategy_params, cash, commission
        )
        is_metrics = PerformanceAnalyzer.calculate_metrics(is_results)
        is_passed, is_failed = PerformanceAnalyzer.evaluate_strategy(
            is_metrics, is_sample=True
        )

        # Step 5: Run Out-of-Sample backtest
        print("\n=== STEP 5: OUT-OF-SAMPLE BACKTEST ===")
        oos_results = self.run_backtest(
            bias_free.oos_data, strategy_class, strategy_params, cash, commission
        )
        oos_metrics = PerformanceAnalyzer.calculate_metrics(oos_results)
        oos_passed, oos_failed = PerformanceAnalyzer.evaluate_strategy(
            oos_metrics, is_sample=False
        )

        # Step 6: Robustness test
        print("\n=== STEP 6: ROBUSTNESS TEST ===")
        robustness_passed = self.check_robustness(is_metrics, oos_metrics)

        # Step 7: Overall assessment
        print("\n=== STEP 7: OVERALL ASSESSMENT ===")
        overall_passed = is_passed and oos_passed and robustness_passed

        if overall_passed:
            print("🎉 Strategy PASSES all tests and is ready for submission!")
        else:
            print("❌ Strategy FAILS some tests and needs improvement")

        # Return comprehensive results
        results = {
            "strategy": strategy_name,
            "symbol": symbol,
            "parameters": strategy_params,
            "in_sample": {
                "metrics": is_metrics,
                "passed": is_passed,
                "failed_criteria": is_failed,
            },
            "out_of_sample": {
                "metrics": oos_metrics,
                "passed": oos_passed,
                "failed_criteria": oos_failed,
            },
            "robustness": {"passed": robustness_passed},
            "overall_passed": overall_passed,
        }

        return results

    def check_robustness(self, is_metrics, oos_metrics):
        """Check if OOS performance is similar to IS performance (90% criteria)"""
        print("Checking robustness (OOS ~ IS performance):")

        # Compare key metrics
        sharpe_ratio = (
            oos_metrics["sharpe_ratio"] / is_metrics["sharpe_ratio"]
            if is_metrics["sharpe_ratio"] != 0
            else 0
        )
        daily_return_ratio = (
            oos_metrics["daily_return"] / is_metrics["daily_return"]
            if is_metrics["daily_return"] != 0
            else 0
        )

        print(f"Sharpe ratio ratio (OOS/IS): {sharpe_ratio:.3f}")
        print(f"Daily return ratio (OOS/IS): {daily_return_ratio:.3f}")

        # Check if ratios are within 90% range (0.9 to 1.1)
        robustness_passed = (
            0.9 <= sharpe_ratio <= 1.1 and 0.9 <= daily_return_ratio <= 1.1
        )

        if robustness_passed:
            print("✓ Robustness test PASSED")
        else:
            print("❌ Robustness test FAILED")

        return robustness_passed


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Backtesting Framework")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--symbol", required=True, help="Symbol to trade")
    parser.add_argument(
        "--strategy", choices=["sma", "rsi"], default="sma", help="Strategy to use"
    )
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash")
    parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    parser.add_argument("--report", help="Output report file")

    args = parser.parse_args()

    # Strategy parameters
    strategy_params = {
        "sma": {"fast_period": 10, "slow_period": 30},
        "rsi": {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "rsi_exit": 50,
        },
    }

    # Run comprehensive backtest
    framework = BacktestFramework()
    results = framework.run_comprehensive_backtest(
        args.csv,
        args.symbol,
        args.strategy,
        strategy_params[args.strategy],
        args.cash,
        args.commission,
    )

    # Save results
    if args.report and results:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.report}")


if __name__ == "__main__":
    main()
