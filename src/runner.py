import argparse, json, os, glob, hashlib
import pandas as pd, backtrader as bt
from pathlib import Path

from strategies.sma_cross import SmaCross
from strategies.rsi_meanrev import RsiMeanRev
from analyzers.customs import TimeInMarket, Turnover
from analyzers.portfolio import PortfolioReturns
from utils import (
    read_price_csv,
    data_quality_report,
    split_is_oos,
    daily_returns_from_bar_returns,
    sharpe_from_daily,  # keep for compatibility
    profit_factor_from_trades,
    win_rate_from_trades,
    fitness_score,
    annualized_return_from_daily,
    fmt6,
    param_product,
    max_drawdown_pct_from_equity,
    profit_factor_from_daily,
    trades_closed_from_tradeanalyzer,
)


def to_bt_feed(df):
    df = df.rename(columns={"date": "datetime"}).set_index("datetime")
    return bt.feeds.PandasData(dataname=df)


def get_strategy_class(strategy_name: str):
    """Factory function to get strategy class by name."""
    strategies = {
        "sma": SmaCross,
        "rsi": RsiMeanRev,
    }
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}"
        )
    return strategies[strategy_name]


class PercentCashSizer(bt.Sizer):
    params = dict(perc=0.95)

    def _getsizing(self, comminfo, cash, data, isbuy):
        price = max(data.close[0], 1e-12)
        size = int((cash * self.p.perc) / price)
        return max(size, 1)  # ensure at least 1 share if possible


def _hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(8192), b""):
            h.update(ch)
    return h.hexdigest()


def run_once(df, strategy_name, strategy_params, cash, commission, slippage_bps=0, sizer_perc=0.95, tf="days"):
    cerebro = bt.Cerebro()
    strategy_class = get_strategy_class(strategy_name)
    cerebro.addstrategy(strategy_class, **strategy_params)
    cerebro.addsizer(PercentCashSizer, perc=sizer_perc)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    if slippage_bps > 0:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)

    cerebro.adddata(to_bt_feed(df))
    # Use our portfolio-based analyzer
    cerebro.addanalyzer(PortfolioReturns, _name="port", tf=tf)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(TimeInMarket, _name="tim")
    cerebro.addanalyzer(Turnover, _name="to")

    strat = cerebro.run()[0]

    # Get portfolio analysis with debug info
    ana = strat.analyzers.port.get_analysis()
    equity = pd.Series(ana.get("equity", {}))
    daily = pd.Series(ana.get("daily", {}))
    dbg = ana.get("debug", {})

    # Debug in console
    print("\n[DEBUG] Equity/Daily sanity")
    print({k: dbg.get(k) for k in ["equity_min","equity_max","equity_bad_nonpos","spike_count_abs_gt_50pct"]})
    print(f"[DEBUG] points: equity={len(equity)}, daily={len(daily)}")

    # Calculate metrics from daily & equity
    sharpe = sharpe_from_daily(daily)
    ann = annualized_return_from_daily(daily)
    dd_pct = max_drawdown_pct_from_equity(equity)
    avg_daily = float(daily.mean()) if len(daily) else float("nan")

    # ProfitFactor: ưu tiên từ TradeAnalyzer nếu có số liệu PnL; nếu thiếu, fallback daily
    ta = strat.analyzers.trades.get_analysis()
    closed_trades = trades_closed_from_tradeanalyzer(ta)
    pf = None
    try:
        pnl_gross = ta['pnl']['gross']['total']
        pnl_loss = ta['pnl']['gross']['loss']
        pf = (pnl_gross / abs(pnl_loss)) if pnl_loss != 0 else (float('inf') if pnl_gross>0 else float('nan'))
    except Exception:
        pf = profit_factor_from_daily(daily)

    winrate = win_rate_from_trades(ta)
    tim = strat.analyzers.tim.get_analysis().get("time_in_market_pct", float("nan"))
    turnover = strat.analyzers.to.get_analysis().get("turnover", float("nan"))
    fit = fitness_score(sharpe, avg_daily, turnover)

    return (
        dict(
            annual_return_pct=(ann * 100.0 if pd.notna(ann) else float("nan")),
            avg_daily_return=avg_daily,
            sharpe=sharpe,
            win_rate_pct=winrate,
            profit_factor=pf,
            max_drawdown_pct=dd_pct,
            turnover_per_day=turnover,
            time_in_market_pct=tim,
            fitness=fit,
            n_daily_points=int(len(daily)),
            n_trades=closed_trades,
        ),
        daily,
    )


def walk_forward(df, strategy_name, strategy_params, is_years=5, oos_years=1, **kwargs):
    segs = []
    start = df["date"].min()
    end = df["date"].max()
    cur = start
    while True:
        is_end = cur + pd.DateOffset(years=is_years)
        oos_end = is_end + pd.DateOffset(years=oos_years)
        if oos_end > end:
            break
        is_df = df[(df["date"] >= cur) & (df["date"] < is_end)]
        oos_df = df[(df["date"] >= is_end) & (df["date"] < oos_end)]
        if len(is_df) < 50 or len(oos_df) < 10:
            break
        is_m, _ = run_once(is_df, strategy_name, strategy_params, tf="days", **kwargs)
        oos_m, _ = run_once(oos_df, strategy_name, strategy_params, tf="days", **kwargs)
        segs.append(
            {
                "start": str(cur.date()),
                "is_end": str(is_end.date()),
                "oos_end": str(oos_end.date()),
                "is": is_m,
                "oos": oos_m,
            }
        )
        cur = cur + pd.DateOffset(years=oos_years)
    return segs


def grid_search(df, strategy_name, grid_spec, cash, commission, slippage_bps, sizer_perc, topk=5):
    """Run grid search on IS data and evaluate top-k on OOS."""
    import json
    
    # Parse grid specification
    if isinstance(grid_spec, str):
        grid_spec = json.loads(grid_spec)
    
    # Generate all parameter combinations
    param_combos = param_product(grid_spec)
    print(f"\n=== GRID SEARCH ===")
    print(f"Strategy: {strategy_name}")
    print(f"Total combinations: {len(param_combos)}")
    
    # Run IS evaluation for all combinations
    is_results = []
    for i, params in enumerate(param_combos):
        try:
            metrics, _ = run_once(df, strategy_name, params, cash, commission, slippage_bps, sizer_perc, tf="days")
            
            # Apply filters and scoring
            score = metrics['sharpe']
            discarded = False
            penalty = 1.0
            
            # Filter 1: Minimum data points (0.5 years)
            if metrics['n_daily_points'] < 126:
                discarded = True
                
            # Filter 2: Minimum trades (estimate from win rate and other metrics)
            trades = metrics.get('n_trades', 0)
            if trades == 0:
                # Try to estimate from other metrics
                if 'win_rate_pct' in metrics and not pd.isna(metrics['win_rate_pct']):
                    trades = 1  # At least one trade if we have win rate
                else:
                    discarded = True
            elif trades < 20:
                discarded = True
                
            # Filter 3: Turnover penalty
            turnover = metrics.get('turnover_per_day', 0)
            if turnover < 0.01 or turnover > 0.06:
                penalty = 0.8
                
            if not discarded:
                final_score = score * penalty
                is_results.append({
                    'params': params,
                    'metrics': metrics,
                    'score': final_score,
                    'penalty': penalty
                })
                
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
    
    # Sort by score and take top-k
    is_results.sort(key=lambda x: x['score'], reverse=True)
    top_is = is_results[:topk]
    
    print(f"Combinations after filtering: {len(is_results)}")
    print(f"Top {topk} IS configurations:")
    
    # Evaluate top-k on OOS
    oos_results = []
    for rank, result in enumerate(top_is, 1):
        params = result['params']
        is_metrics = result['metrics']
        
        # Split data for OOS evaluation
        is_df, oos_df = split_is_oos(df, 8, 1)
        
        try:
            oos_metrics, _ = run_once(oos_df, strategy_name, params, cash, commission, slippage_bps, sizer_perc, tf="days")
            
            # Check pass/fail criteria
            pass_sharpe = oos_metrics['sharpe'] >= 0.7
            pass_dd = oos_metrics['max_drawdown_pct'] < 55.0
            pass_turnover = 0.01 <= oos_metrics['turnover_per_day'] <= 0.06
            pass_fitness = oos_metrics['fitness'] > 2.1
            
            oos_results.append({
                'rank': rank,
                'params': params,
                'sharpe_is': is_metrics['sharpe'],
                'sharpe_oos': oos_metrics['sharpe'],
                'maxdd_oos': oos_metrics['max_drawdown_pct'],
                'fitness_oos': oos_metrics['fitness'],
                'trades': is_metrics.get('n_trades', 0),
                'turnover_oos': oos_metrics['turnover_per_day'],
                'pass_sharpe': pass_sharpe,
                'pass_dd': pass_dd,
                'pass_turnover': pass_turnover,
                'pass_fitness': pass_fitness,
                'all_pass': all([pass_sharpe, pass_dd, pass_turnover, pass_fitness])
            })
            
        except Exception as e:
            print(f"Error evaluating OOS for rank {rank}: {e}")
            continue
    
    return oos_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--dtfmt", default="%Y-%m-%d")
    p.add_argument("--tf", choices=["days", "hours", "minutes"], default="days")
    p.add_argument(
        "--symbol", default=None, help="Filter this symbol if CSV has a 'symbol' column"
    )
    p.add_argument("--strategy", choices=["sma", "rsi"], default="sma")
    p.add_argument("--fast", type=int, default=10)
    p.add_argument("--slow", type=int, default=30)
    p.add_argument("--cash", type=float, default=1_000_000.0)
    p.add_argument("--commission", type=float, default=0.001)
    p.add_argument("--slippage_bps", type=float, default=0.0)
    p.add_argument("--sizer_perc", type=float, default=0.95)
    p.add_argument("--is_years", type=int, default=8)
    p.add_argument("--oos_years", type=int, default=1)
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--grid_spec", default="")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--report", default="reports/report.json")
    args = p.parse_args()

    # Load & basic QA
    df_raw = read_price_csv(args.csv, dtfmt=args.dtfmt)
    # Optional symbol filter if column exists
    import pandas as pd

    if "symbol" in df_raw.columns and args.symbol:
        before = len(df_raw)
        df_raw = df_raw[df_raw["symbol"].astype(str) == str(args.symbol)].copy()
        print(f"Filtered by symbol '{args.symbol}': {before} -> {len(df_raw)} rows")

    qual = dict(report=data_quality_report(df_raw), warnings=[])
    if qual["report"]["empty"]:
        qual["warnings"].append("CSV empty")
    if qual["report"]["n_dupes"] > 0:
        qual["warnings"].append("Duplicate timestamps")
    if qual["report"]["n_nans"] > 0:
        qual["warnings"].append("NaNs present")
    if not qual["report"]["dates_monotonic"]:
        qual["warnings"].append("Dates not monotonic")

    is_df, oos_df = split_is_oos(df_raw, args.is_years, args.oos_years)
    common = dict(
        cash=args.cash,
        commission=args.commission,
        slippage_bps=args.slippage_bps,
        sizer_perc=args.sizer_perc,
    )

    # Build strategy parameters based on strategy type
    if args.strategy == "sma":
        strategy_params = {"fast": args.fast, "slow": args.slow}
    else:  # rsi
        strategy_params = {
            "rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70, "rsi_exit": 50,
            "sma_trend": 200, "use_trend_filter": True,
            "atr_period": 14, "stop_atr": 0.0, "take_atr": 0.0, "min_hold": 0
        }
    
    # Print IS/OOS ranges for debugging
    print(f"[DEBUG] IS range: {is_df['date'].min().date()} → {is_df['date'].max().date()} | rows={len(is_df)}")
    print(f"[DEBUG] OOS range: {oos_df['date'].min().date()} → {oos_df['date'].max().date()} | rows={len(oos_df)}")
    
    is_m, is_daily = run_once(is_df, args.strategy, strategy_params, tf=args.tf, **common)
    oos_m, oos_daily = run_once(oos_df, args.strategy, strategy_params, tf=args.tf, **common)
    
    # Cảnh báo nếu OOS "không có lệnh"
    if oos_m.get('n_trades', 0) == 0 and len(oos_daily) > 0 and abs(oos_daily).sum() < 1e-9:
        print("[WARN] OOS produced no trades or zero-return days. Check strategy filters (e.g., RSI thresholds, trend filter) and sizing.")

    result = dict(
        data_quality=qual,
        strategy=args.strategy,
        params=dict(
            strategy=args.strategy,
            **strategy_params,
            cash=args.cash,
            commission=args.commission,
            slippage_bps=args.slippage_bps,
            sizer_perc=args.sizer_perc,
            tf=args.tf,
            symbol=args.symbol,
        ),
        IS=is_m,
        OOS=oos_m,
        pass_criteria=dict(
            sharpe=(is_m["sharpe"] > 1.0 and oos_m["sharpe"] >= 0.7),
            drawdown=(oos_m["max_drawdown_pct"] < 55.0),
            turnover=(0.01 < oos_m["turnover_per_day"] < 0.06),
            fitness=(oos_m["fitness"] > 2.1),
            daily_return=(abs(oos_m["avg_daily_return"]) >= 0.0004),
        ),
    )

    if args.walkforward:
        result["walkforward"] = walk_forward(
            df_raw,
            args.strategy,
            strategy_params,
            is_years=args.is_years,
            oos_years=args.oos_years,
            **common,
        )
    
    # Grid search mode
    if args.grid:
        if not args.grid_spec:
            # Use default grids based on strategy
            if args.strategy == "sma":
                default_grid = {"fast": [5, 10, 15], "slow": [20, 30, 50]}
            else:  # rsi
                default_grid = {
                    "rsi_period": [7, 14],
                    "rsi_lower": [20, 30],
                    "rsi_exit": [45, 55],
                    "sma_trend": [100, 200],
                    "use_trend_filter": [False],  # Disable trend filter to avoid infinity issues
                    "stop_atr": [0, 1.5],
                    "take_atr": [0, 2.0],
                    "min_hold": [0, 3]
                }
            grid_spec = default_grid
        else:
            # Parse the provided grid specification
            import json
            grid_spec = json.loads(args.grid_spec)
        
        grid_results = grid_search(
            df_raw, args.strategy, grid_spec,
            args.cash, args.commission, args.slippage_bps, args.sizer_perc, args.topk
        )
        
        # Save grid results
        rp = Path(args.report)
        grid_csv = rp.parent / (rp.stem + "_grid.csv")
        grid_json = rp.parent / (rp.stem + "_grid.json")
        
        # Convert to DataFrame for CSV
        grid_df = pd.DataFrame(grid_results)
        grid_df.to_csv(grid_csv, index=False)
        
        # Save JSON
        import json
        with open(grid_json, "w") as f:
            json.dump(grid_results, f, indent=2, default=str)
        
        print(f"\nGrid results saved:")
        print(f"- CSV: {grid_csv}")
        print(f"- JSON: {grid_json}")
        
        # Print summary table
        print(f"\n=== GRID SEARCH RESULTS ===")
        print(f"{'Rank':<4} {'Sharpe_IS':<10} {'Sharpe_OOS':<12} {'MaxDD_OOS':<11} {'Fitness_OOS':<13} {'Trades':<7} {'Turnover_OOS':<15} {'Status':<8}")
        print("-" * 90)
        
        for result in grid_results:
            status = "PASS" if result['all_pass'] else "FAIL"
            print(f"{result['rank']:<4} {result['sharpe_is']:<10.3f} {result['sharpe_oos']:<12.3f} {result['maxdd_oos']:<11.3f} {result['fitness_oos']:<13.3f} {result['trades']:<7} {result['turnover_oos']:<15.3f} {status:<8}")
        
        # Summary statistics
        oos_sharpes = [r['sharpe_oos'] for r in grid_results if not pd.isna(r['sharpe_oos'])]
        if oos_sharpes:
            import numpy as np
            median_sharpe = np.median(oos_sharpes)
            passing_configs = sum(1 for r in grid_results if r['sharpe_oos'] >= 0.7)
            print(f"\nSummary:")
            print(f"- Median Sharpe_OOS: {median_sharpe:.3f}")
            print(f"- Configs with Sharpe_OOS ≥ 0.7: {passing_configs}/{len(grid_results)}")
        
        # Clean grid results for JSON serialization
        clean_grid_results = []
        for r in grid_results:
            clean_result = r.copy()
            # Convert numpy types to native Python types
            for key, value in clean_result.items():
                if hasattr(value, 'item'):  # numpy scalar
                    clean_result[key] = value.item()
                elif isinstance(value, dict):
                    clean_result[key] = {k: v.item() if hasattr(v, 'item') else v for k, v in value.items()}
            clean_grid_results.append(clean_result)
        
        result["grid_search"] = clean_grid_results

    # Save artifacts
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    
    # Save daily returns
    is_daily.to_csv(rp.parent / (rp.stem + "_IS_daily.csv"), header=["ret"])
    oos_daily.to_csv(rp.parent / (rp.stem + "_OOS_daily.csv"), header=["ret"])
    
    # Save equity files for debugging
    # Note: We need to get equity data from the run_once calls
    # For now, save the daily returns and we'll enhance this later
    
    import json
    with open(rp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved -> {rp}")

    # Console summary (compact)
    print("\n=== DATA QUALITY ===")
    print(f"- rows: {qual['report'].get('n_rows')}")
    print(f"- duplicates: {qual['report'].get('n_dupes')}")
    print(f"- NaNs: {qual['report'].get('n_nans')}")
    print(f"- outliers: {qual['report'].get('n_outliers')}")
    print(f"- dates_monotonic: {qual['report'].get('dates_monotonic')}")
    print(f"- range: {qual['report'].get('start')} -> {qual['report'].get('end')}")
    
    print(f"\n=== STRATEGY ===")
    print(f"- Type: {args.strategy.upper()}")
    print(f"- Params: {strategy_params}")

    def _fmt(m):  # shorten floats
        m2 = m.copy()
        for k in [
            "avg_daily_return",
            "sharpe",
            "win_rate_pct",
            "profit_factor",
            "max_drawdown_pct",
            "turnover_per_day",
            "time_in_market_pct",
            "fitness",
            "annual_return_pct",
        ]:
            if k in m2 and isinstance(m2[k], (int, float)):
                if k == "avg_daily_return":
                    m2[k] = fmt6(m2[k])
                else:
                    m2[k] = float(m2[k])
        return m2

    print("\n=== METRICS (IN-SAMPLE) ===")
    print(_fmt(is_m))
    print("\n=== METRICS (OUT-OF-SAMPLE) ===")
    print(_fmt(oos_m))

    # Correlation vs peers (filter self & duplicates)
    new_oos = str(rp.parent / (rp.stem + "_OOS_daily.csv"))

    def _md5(p):
        h = hashlib.md5()
        with open(p, "rb") as f:
            for ch in iter(lambda: f.read(8192), b""):
                h.update(ch)
        return h.hexdigest()

    new_md5 = _md5(new_oos)
    peers = []
    for pth in glob.glob(str(rp.parent / "*_OOS_daily.csv")):
        if os.path.abspath(pth) == os.path.abspath(new_oos):  # self
            continue
        if _md5(pth) == new_md5:  # duplicate content
            continue
        peers.append(pth)
    if peers:
        print("\nCorrelation warnings (>=0.5):")
        new = pd.read_csv(new_oos, index_col=0, parse_dates=True).rename(
            columns={"ret": "new"}
        )
        for q in peers:
            other = pd.read_csv(q, index_col=0, parse_dates=True).rename(
                columns={"ret": "peer"}
            )
            joined = new.join(other, how="inner").dropna()
            if len(joined) >= 30:
                c = joined["new"].corr(joined["peer"])
                if c >= 0.5:
                    print(f"- {Path(q).name}: {c:.3f}")
    else:
        print("\nCorrelation: no peers to compare.")


if __name__ == "__main__":
    main()
