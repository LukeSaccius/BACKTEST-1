import pandas as pd
import numpy as np
import itertools
import logging


def read_price_csv(path, dtfmt="%Y-%m-%d"):
    df = pd.read_csv(path)
    # Normalize headers
    rename = {}
    lower_to_orig = {c.lower().strip(): c for c in df.columns}

    aliases = {
        "date": ["date", "datetime", "timestamp", "time"],
        "symbol": ["symbol", "sym", "ticker", "instrument"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close"],
        "volume": ["volume", "vol", "turnover", "qty", "quantity"],
    }

    for need, candidates in aliases.items():
        cand_orig = None
        for cand in candidates:
            if cand in lower_to_orig:
                cand_orig = lower_to_orig[cand]
                break
        if cand_orig is None:
            if need == "symbol":  # symbol is optional
                continue
            raise ValueError(
                f"Missing required column '{need}' (aliases: {candidates}) in CSV {path}. Found: {list(df.columns)}"
            )
        rename[cand_orig] = need
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], format=dtfmt, errors="coerce")

    # Count and drop rows with invalid dates
    n_nat = int(df["date"].isna().sum())
    if n_nat > 0:
        logging.warning(f"Dropping {n_nat} rows with NaT dates from {path}")
        df = df.dropna(subset=["date"])

    if df.empty:
        raise ValueError(f"No valid rows remain after dropping NaT dates from {path}")

    # Determine which columns to return
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    if "symbol" in df.columns:
        required_cols.append("symbol")

    return df[required_cols].sort_values("date").reset_index(drop=True)


def param_product(grid: dict) -> list:
    """
    Generate all parameter combinations from a grid specification.

    Args:
        grid: Dict with parameter names as keys and lists of values as values

    Returns:
        List of dicts, each containing one parameter combination

    Example:
        grid = {"fast": [5, 10], "slow": [20, 30]}
        result = [{"fast": 5, "slow": 20}, {"fast": 5, "slow": 30}, ...]
    """
    keys = grid.keys()
    values = grid.values()
    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def detect_outliers_zscore(series, z=6.0):
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        return pd.Series([False] * len(series), index=series.index)
    zscores = (s - s.mean()) / s.std(ddof=0)
    mask = zscores.abs() > z
    out = pd.Series([False] * len(series), index=series.index)
    out.loc[mask.index] = mask.values
    return out


def data_quality_report(df: pd.DataFrame):
    rep = {}
    rep["empty"] = df.empty
    rep["n_rows"] = len(df)
    rep["n_dupes"] = int(df.duplicated(subset=["date"]).sum())
    rep["n_nans"] = int(df.isna().sum().sum())
    # return series uses pct change of close
    ret = df["close"].pct_change()
    outliers = detect_outliers_zscore(ret, z=8.0)  # conservative
    rep["n_outliers"] = int(outliers.sum())
    # check monotonic dates
    rep["dates_monotonic"] = df["date"].is_monotonic_increasing
    # approx day gaps
    df_day = df.set_index("date").resample("1D").size()
    rep["n_day_gaps"] = int((df_day == 0).sum())
    # start/end
    if len(df) > 0:
        rep["start"] = df["date"].iloc[0].isoformat()
        rep["end"] = df["date"].iloc[-1].isoformat()
        rep["n_days"] = int((df["date"].iloc[-1] - df["date"].iloc[0]).days) or 1
    return rep


def split_is_oos(df, is_years=8, oos_years=1):
    if df.empty:
        raise ValueError("Empty dataframe")
    end = df["date"].iloc[-1]
    oos_start = end - pd.DateOffset(years=oos_years)
    is_start = oos_start - pd.DateOffset(years=is_years)
    is_df = df[(df["date"] >= is_start) & (df["date"] < oos_start)].copy()
    oos_df = df[(df["date"] >= oos_start) & (df["date"] <= end)].copy()
    return is_df, oos_df


def daily_returns_from_bar_returns(bar_returns: dict):
    """
    bar_returns: mapping datetime->bar_return (float), as from bt.analyzers.TimeReturn
    Returns a pandas Series of DAILY returns.
    """
    if not bar_returns:
        return pd.Series(dtype=float)
    s = pd.Series(bar_returns)
    # s.index is datetime, values are per-bar returns
    daily = (1.0 + s).groupby(pd.Grouper(freq="1D")).prod() - 1.0
    daily = daily.dropna()
    return daily


def sharpe_from_daily(daily_returns, rf=0.0, periods_per_year=252):
    if len(daily_returns) < 2:
        return float("nan")
    excess = daily_returns - rf / periods_per_year
    mean = excess.mean()
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    ann = (mean / std) * np.sqrt(periods_per_year)
    return float(ann)


def annualized_return_from_daily(daily, periods=252):
    """
    Annualize from the actual daily return series (not from avgDaily),
    robust to partial-year sample lengths.
    """
    if daily is None or len(daily) == 0:
        return float("nan")
    total = float((1.0 + daily).prod())
    years = len(daily) / periods
    if years <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def profit_factor_from_trades(trade_analyzer_dict):
    won = trade_analyzer_dict.get("won", {}).get("pnl", {}).get("gross", 0.0)
    lost = trade_analyzer_dict.get("lost", {}).get("pnl", {}).get("gross", 0.0)
    if lost == 0:
        return float("inf") if won > 0 else float("nan")
    return abs(won / lost)


def win_rate_from_trades(trade_analyzer_dict):
    total = trade_analyzer_dict.get("total", {}).get("total", 0)
    won = trade_analyzer_dict.get("won", {}).get("total", 0)
    if total == 0:
        return float("nan")
    return 100.0 * won / total


def fitness_score(sharpe, daily_return, turnover):
    denom = max(turnover, 0.00125)
    try:
        return float(sharpe) * float(np.sqrt(abs(daily_return) / denom))
    except Exception:
        return float("nan")


def fmt6(x):
    try:
        return f"{x:.6f}"
    except Exception:
        return str(x)


def max_drawdown_pct_from_equity(eq):
    if eq is None or len(eq) == 0:
        return float("nan")
    eq = eq.astype(float)
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()  # negative
    return abs(float(dd)) * 100.0  # in percent


def profit_factor_from_daily(d):
    if d is None or len(d) == 0:
        return float("nan")
    pos = d[d > 0].sum()
    neg = d[d < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / abs(neg))


def trades_closed_from_tradeanalyzer(ta):
    # Backtrader TradeAnalyzer structure can vary; try common paths
    try:
        total = ta.get("total", {})
        closed = total.get("closed")
        if closed is None:
            closed = ta.get("total_closed")
        return int(closed) if closed is not None else 0
    except Exception:
        return 0
