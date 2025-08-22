import backtrader as bt
import pandas as pd


class PortfolioReturns(bt.Analyzer):
    """
    Ghi lại broker equity mỗi bar và trả về:
    - equity series (per bar)
    - daily returns (được tính về tần suất ngày)
    Có chèn kiểm tra an toàn để tránh %change vô lý.
    """
    params = dict(tf='days')  # 'days' | 'hours' | 'minutes'

    def start(self):
        self.values = []
        self.dts = []

    def next(self):
        v = float(self.strategy.broker.getvalue())
        dt = self.strategy.datas[0].datetime.datetime(0)
        self.values.append(v)
        self.dts.append(dt)

    def get_analysis(self):
        out = {"equity": {}, "daily": {}, "debug": {}}
        if not self.values:
            return out
        equity = pd.Series(self.values, index=pd.DatetimeIndex(self.dts)).astype(float)

        # Sanity: equity không được <= 0
        bad_equity = (equity <= 0).sum()
        out["debug"]["equity_min"] = float(equity.min())
        out["debug"]["equity_max"] = float(equity.max())
        out["debug"]["equity_bad_nonpos"] = int(bad_equity)

        # Tránh %change trên 0 → drop các điểm không hợp lệ
        equity = equity[equity > 0]

        # Tính returns theo tf
        if self.p.tf == 'days':
            rets = equity.pct_change()
        else:
            # Gộp về 1D nếu intraday
            rets = equity.pct_change()
            rets = (1.0 + rets).groupby(pd.Grouper(freq="1D")).prod() - 1.0

        # Loại các daily quá lớn (ngưỡng an toàn 50%/ngày)
        spikes = rets[rets.abs() > 0.5]
        out["debug"]["spike_count_abs_gt_50pct"] = int(spikes.count())
        rets = rets[rets.abs() <= 0.5]

        rets = rets.dropna()
        out["equity"] = {k.to_pydatetime(): float(v) for k, v in equity.items()}
        out["daily"]  = {k.to_pydatetime(): float(v) for k, v in rets.items()}
        return out
