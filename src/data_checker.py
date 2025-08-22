#!/usr/bin/env python3
"""
Data Checker for Multi-Symbol CSV Files
Analyzes data quality, detects multi-symbol data, and provides warnings.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def detect_outliers_zscore(series, z=8.0):
    """Detect outliers using z-score method."""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        return pd.Series([False] * len(series), index=series.index)

    zscores = (s - s.mean()) / s.std(ddof=0)
    mask = zscores.abs() > z
    out = pd.Series([False] * len(series), index=series.index)
    out.loc[mask.index] = mask.values
    return out


def run_check(csv_path, dtfmt="%Y-%m-%d"):
    """Run comprehensive data quality check on CSV file."""
    print(f"=== DATA QUALITY CHECK ===")
    print(f"File: {csv_path}")
    print(f"Date format: {dtfmt}")
    print()

    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"✓ CSV loaded successfully")

        # Normalize headers (case-insensitive)
        rename = {}
        lower_to_orig = {c.lower().strip(): c for c in df.columns}

        aliases = {
            "datetime": ["datetime", "date", "timestamp", "time"],
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
                    f"Missing required column '{need}' (aliases: {candidates})"
                )
            rename[cand_orig] = need

        df = df.rename(columns=rename)

        # Basic info
        print(f"✓ Headers normalized")
        print(f"✓ Columns: {list(df.columns)}")
        print()

        # Convert datetime
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["datetime"], format=dtfmt, errors="coerce"
            )
            invalid_dates = df["datetime"].isna().sum()
            if invalid_dates > 0:
                print(f"⚠ Warning: {invalid_dates} invalid dates found")

        # Quality metrics
        print("=== QUALITY METRICS ===")
        print(f"Row count: {len(df):,}")

        if "datetime" in df.columns:
            valid_dates = df["datetime"].dropna()
            if len(valid_dates) > 0:
                print(f"Date range: {valid_dates.min()} -> {valid_dates.max()}")
                print(f"Date span: {(valid_dates.max() - valid_dates.min()).days} days")

        # NaNs
        total_nans = df.isna().sum().sum()
        print(f"Total NaNs: {total_nans:,}")

        # Duplicates
        total_dupes = df.duplicated().sum()
        print(f"Total duplicates: {total_dupes:,}")

        # Symbol analysis
        if "symbol" in df.columns:
            print(f"Unique symbols: {df['symbol'].nunique():,}")
            print(
                f"Symbols: {sorted(df['symbol'].unique())[:10]}{'...' if df['symbol'].nunique() > 10 else ''}"
            )

            # Duplicates by (datetime, symbol)
            datetime_symbol_dupes = df.duplicated(subset=["datetime", "symbol"]).sum()
            print(f"Duplicates by (datetime, symbol): {datetime_symbol_dupes:,}")

            # Duplicates by datetime only
            datetime_dupes = df.duplicated(subset=["datetime"]).sum()
            print(f"Duplicates by datetime only: {datetime_dupes:,}")

            # Multi-symbol warning
            if datetime_dupes > 0 and datetime_symbol_dupes == 0:
                print(
                    f"⚠ WARNING: High datetime duplicates ({datetime_dupes:,}) but no (datetime, symbol) duplicates"
                )
                print(
                    f"   This suggests multi-symbol data - consider filtering by symbol for backtesting"
                )
        else:
            # Single instrument
            datetime_dupes = df.duplicated(subset=["datetime"]).sum()
            print(f"Duplicates by datetime: {datetime_dupes:,}")

        # Outlier analysis
        if "close" in df.columns:
            returns = df["close"].pct_change()
            outliers = detect_outliers_zscore(returns, z=8.0)
            outlier_count = outliers.sum()
            print(f"Outliers (z-score > 8): {outlier_count:,}")

            if outlier_count > 0:
                print(f"⚠ Warning: {outlier_count} extreme return outliers detected")

        # Quality score
        print()
        print("=== QUALITY SCORE ===")
        score = 100

        if total_nans > 0:
            score -= min(20, total_nans / len(df) * 100)

        if total_dupes > 0:
            score -= min(30, total_dupes / len(df) * 100)

        if "close" in df.columns and outlier_count > 0:
            score -= min(20, outlier_count / len(df) * 100)

        if "datetime" in df.columns and invalid_dates > 0:
            score -= min(20, invalid_dates / len(df) * 100)

        score = max(0, score)

        if score >= 80:
            quality = "EXCELLENT"
        elif score >= 60:
            quality = "GOOD"
        elif score >= 40:
            quality = "FAIR"
        else:
            quality = "POOR"

        print(f"Overall quality: {quality} ({score:.1f}/100)")

        # Recommendations
        print()
        print("=== RECOMMENDATIONS ===")
        if total_nans > 0:
            print(f"- Clean {total_nans:,} NaN values")

        if total_dupes > 0:
            print(f"- Remove {total_dupes:,} duplicate rows")

        if "close" in df.columns and outlier_count > 0:
            print(f"- Investigate {outlier_count:,} extreme return outliers")

        if "symbol" in df.columns and datetime_dupes > 0 and datetime_symbol_dupes == 0:
            print(
                "- Filter by specific symbol for backtesting to avoid multi-symbol contamination"
            )

        print("✓ Data check completed")

    except Exception as e:
        print(f"❌ Error during data check: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check CSV data quality for multi-symbol data"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--dtfmt", default="%Y-%m-%d", help="Date format string")

    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"❌ Error: CSV file not found: {args.csv}")
        return 1

    success = run_check(args.csv, args.dtfmt)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
