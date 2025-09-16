from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_input_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Validate OHLCV dataframe.

    Fatal: non-empty, DateTimeIndex sorted unique, OHLCV present, no NaN in OHLC.
    Returns (clean_df, report) where report includes bias banner warnings.
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe")
    out = {"fatal": [], "warnings": {}}
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index(pd.to_datetime(df["datetime"]))
        else:
            raise ValueError("Provide DateTimeIndex or 'datetime' column")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    _require_columns(df, ["open", "high", "low", "close", "volume"])
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("NaN found in OHLC")
    # Bias banner (minimal)
    out["warnings"]["survivorship_bias"] = True
    out["warnings"]["backfill_bias"] = True
    out["warnings"]["lookahead_possible"] = True
    return df, out

