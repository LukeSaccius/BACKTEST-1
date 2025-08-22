
import backtrader as bt
from datetime import datetime, timedelta

class TimeInMarket(bt.Analyzer):
    """
    Percentage of bars with a non-zero position.
    """
    def start(self):
        self.bars = 0
        self.in_bars = 0

    def next(self):
        self.bars += 1
        if self.strategy.position.size != 0:
            self.in_bars += 1

    def get_analysis(self):
        tim = 0.0 if self.bars == 0 else (100.0 * self.in_bars / self.bars)
        return dict(time_in_market_pct=tim, bars=self.bars, in_bars=self.in_bars)

class Turnover(bt.Analyzer):
    """
    Turnover = number of trades / number of *days* in sample.
    """
    def start(self):
        self.n_trades = 0
        self.first_dt = None
        self.last_dt = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.n_trades += 1

    def next(self):
        dt = self.strategy.datas[0].datetime.datetime(0)
        if self.first_dt is None:
            self.first_dt = dt
        self.last_dt = dt

    def get_analysis(self):
        if self.first_dt and self.last_dt:
            n_days = max((self.last_dt.date() - self.first_dt.date()).days, 1)
        else:
            n_days = 1
        return dict(turnover=self.n_trades / n_days, n_trades=self.n_trades, n_days=n_days)
