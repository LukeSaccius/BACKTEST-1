#!/usr/bin/env python3
"""
Pre-push sanity check for backtests.

Runs a quick backtest via:
- aggressive_profitable_strategy.py (aggressive_rsi)
- src/runner.py (RSI)

Exits non-zero if any command fails so git can block the push
when used as a pre-push/pre-commit hook.
"""

import os, subprocess as sp, sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n$", " ".join(cmd))
    try:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        res = sp.run(
            cmd,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        return 0
    except sp.CalledProcessError as e:
        print("Command failed:", e)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        return e.returncode or 1


def main():
    py = sys.executable
    rc = 0

    # 1) Aggressive profitable strategy
    rc |= run([
        py,
        str(ROOT / "aggressive_profitable_strategy.py"),
        "--csv", str(ROOT / "VN30_1H.csv"),
        "--symbol", "VNM",
        "--strategy", "aggressive_rsi",
        "--cash", "100000",
        "--commission", "0.001",
        "--report", str(ROOT / "reports" / "prepush_aggressive.json"),
    ])

    # 2) Runner (RSI strategy), with 1/3 OOS
    rc |= run([
        py,
        str(ROOT / "src" / "runner.py"),
        "--csv", str(ROOT / "VN30_1H.csv"),
        "--symbol", "VNM",
        "--strategy", "rsi",
        "--cash", "1000000",
        "--commission", "0.001",
        "--oos_years", "3",
        "--report", str(ROOT / "reports" / "prepush_runner.json"),
    ])

    if rc != 0:
        print("\n[FAIL] Pre-push backtest checks encountered errors.")
    else:
        print("\n[OK] Pre-push backtest checks passed.")

    sys.exit(rc)


if __name__ == "__main__":
    main()
