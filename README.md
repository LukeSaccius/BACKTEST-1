## Backtesting Lab

Complete Backtesting Lab for event-driven strategy evaluation using Backtrader with multiple strategies and grid search capabilities.

### Structure

```
project/
  README.md
  environment.yml
  data/
  reports/
  src/
    utils.py
    runner.py
    strategies/
      sma_cross.py
      rsi_meanrev.py
    analyzers/
      customs.py
      portfolio.py
```

### CSV Format

- Columns (case-insensitive): date/datetime, open, high, low, close, volume
- Use `--dtfmt` to specify date format (default `%Y-%m-%d`)

### Strategies

- **SMA Crossover**: Simple moving average crossover strategy
  - Parameters: `fast`, `slow` (default: 10, 30)
- **RSI Mean Reversion**: RSI-based mean reversion strategy
  - Parameters: `rsi_period`, `rsi_lower`, `rsi_upper`, `rsi_exit`, `sma_trend`, `use_trend_filter`, `atr_period`, `stop_atr`, `take_atr`, `min_hold`

### Environment

```bash
conda env create -f environment.yml
conda activate backtesting-lab
```

### Run

#### Data Check

```bash
python src/data_checker.py --csv data/VN30_1H.csv --dtfmt "%Y-%m-%d"
```

#### Backtest

```bash
# SMA baseline (single symbol)
python src/runner.py --csv data/VN30_1H.csv --symbol VNM --strategy sma \
  --fast 5 --slow 20 --cash 1000000 --commission 0.0005 \
  --dtfmt "%Y-%m-%d" --tf days --report reports/vnm_sma.json

# RSI mean-reversion (single run)
python src/runner.py --csv data/VN30_1H.csv --symbol VNM --strategy rsi \
  --cash 1000000 --commission 0.0005 --dtfmt "%Y-%m-%d" --tf days \
  --report reports/vnm_rsi.json

# Grid search on IS, then evaluate top-5 on OOS (RSI)
python src/runner.py --csv data/VN30_1H.csv --symbol VNM --strategy rsi --grid --topk 5 \
  --grid_spec '{"rsi_period":[7,14],"rsi_lower":[20,30],"rsi_exit":[45,55],"sma_trend":[100,200],"use_trend_filter":[true],"stop_atr":[0,1.5],"take_atr":[0,2.0],"min_hold":[0,3]}' \
  --cash 1000000 --commission 0.0005 --dtfmt "%Y-%m-%d" --tf days \
  --report reports/vnm_rsi_grid.json

# Do the same for VIC
python src/runner.py --csv data/VN30_1H.csv --symbol VIC --strategy rsi --grid --topk 5 \
  --grid_spec '{"rsi_period":[7,14],"rsi_lower":[20,30],"rsi_exit":[45,55],"sma_trend":[100,200],"use_trend_filter":[true],"stop_atr":[0,1.5],"take_atr":[0,2.0],"min_hold":[0,3]}' \
  --cash 1000000 --commission 0.0005 --dtfmt "%Y-%m-%d" --tf days \
  --report reports/vic_rsi_grid.json

## Malaysian SNR + Trendline

```
python scripts/run_msnr.py --csv data/VN30_1H.csv --symbol VNM --is-years 8 --oos-years 2
```
Outputs daily returns CSV and a JSON report in reports/ with protocol flags and diagnostics.
```

Walk-forward:

```bash
python src/runner.py --csv data/VN30_1H.csv --symbol ACB --strategy sma --dtfmt "%Y-%m-%d" --tf days \
  --fast 10 --slow 30 --cash 1000000 --commission 0.001 --slippage_bps 0 \
  --sizer_perc 0.95 --walkforward --report reports/acb_p95_wf.json
```

### Metrics & Criteria (Protocol)

- Annualized Return (%), Avg Daily Return, Sharpe, Win Rate, Profit Factor, Max Drawdown (%), Turnover/day, Time-in-Market (%), Fitness
- Pass if: see Metrics & Criteria (Protocol) below

### Grid Search

- **IS Evaluation**: Test all parameter combinations on in-sample data
- **Filtering**: Discard configs with <126 data points or <20 trades
- **Scoring**: Sharpe(IS) with turnover penalty (×0.8 if outside [0.01, 0.06])
- **OOS Validation**: Evaluate top-k configurations on out-of-sample data
- **Output**: CSV table with pass/fail status against all criteria

### Correlation

Runner compares new OOS daily returns against prior `*_OOS_daily.csv` and flags correlations >= 0.5

### Bias Awareness

- Look‑ahead: OFF (next‑bar execution)
- Fees/slippage: configurable in CLI
- Beware survivorship/backfill/data‑snooping

#### Bias‑Free Protocol Checks
- data-snooping:
  - IS sample size ≥ 252 × number of free parameters
  - OOS share ≥ 1/3 of total rows (runner supports `--oos_years`)
  - Robustness: OOS/IS Sharpe and OOS/IS AvgDaily within ~90% band
- Survivorship/backfill: Single‑symbol CSVs cannot guarantee this; for full protection use point‑in‑time universes including delisted names.

#### Pass gates (default runner)
- IS Sharpe > 1.0, OOS Sharpe ≥ 0.7
- OOS Max Drawdown < 55%
- OOS Turnover/day in [0.01, 0.06]
- OOS Fitness > 2.1
- OOS Avg Daily Return ≥ 0.0004 (~10% annualized)

### Pre‑Push Backtest Check

Run a quick smoke test before pushing—this script runs the aggressive RSI strategy and the generic runner, failing non‑zero on errors:

```
python scripts/prepush_check.py
```

To use it as a Git hook, add a pre‑push hook that calls the script.

### Aggressive RSI Strategy (Standalone)

```
python aggressive_profitable_strategy.py --csv VN30_1H.csv --symbol VNM \
  --strategy aggressive_rsi --cash 100000 --commission 0.001 \
  --report reports/aggressive_vnm.json
```

### Artifacts

- **Normal Run**: JSON report, IS/OOS daily returns CSVs
- **Grid Search**: Additional `*_grid.csv` and `*_grid.json` files
- **All files**: Saved in `reports/` directory with clear naming



