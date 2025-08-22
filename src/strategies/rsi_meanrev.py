import backtrader as bt


class RsiMeanRev(bt.Strategy):
    """
    RSI Mean Reversion Strategy (Long Only)
    
    Entry: RSI < rsi_lower AND (optional trend filter: close > sma_trend)
    Exit: RSI > rsi_exit OR take profit/stop loss via ATR OR RSI > rsi_upper
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_lower', 30),
        ('rsi_upper', 70),
        ('rsi_exit', 50),
        ('sma_trend', 200),
        ('use_trend_filter', True),
        ('atr_period', 14),
        ('stop_atr', 0.0),  # 0 = disabled
        ('take_atr', 0.0),  # 0 = disabled
        ('min_hold', 0),
    )
    
    def __init__(self):
        # Indicators
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.sma = bt.ind.SMA(self.data.close, period=self.p.sma_trend)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        
        # State tracking
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.bars_held = 0
        
    def next(self):
        # Skip if we have a pending order
        if self.order:
            return
            
        # Update bars held counter
        if self.position:
            self.bars_held += 1
        
        # Entry logic (long only)
        if not self.position:
            trend_ok = not self.p.use_trend_filter or self.data.close[0] > self.sma[0]
            if self.rsi[0] < self.p.rsi_lower and trend_ok:
                self.order = self.buy()
                self.entry_price = self.data.close[0]
                self.entry_bar = len(self)
                self.bars_held = 0
                
        # Exit logic
        elif self.position:
            # Check minimum hold period (except for hard stops)
            can_exit = self.bars_held >= self.p.min_hold
            
            # Exit conditions
            exit_rsi = self.rsi[0] > self.p.rsi_exit
            exit_upper = self.rsi[0] > self.p.rsi_upper
            
            # ATR-based exits
            exit_take = False
            exit_stop = False
            
            if self.p.take_atr > 0:
                take_level = self.entry_price + self.p.take_atr * self.atr[0]
                exit_take = self.data.close[0] >= take_level
                
            if self.p.stop_atr > 0:
                stop_level = self.entry_price - self.p.stop_atr * self.atr[0]
                exit_stop = self.data.close[0] <= stop_level
                
            # Execute exit if conditions met
            if can_exit and (exit_rsi or exit_upper or exit_take or exit_stop):
                self.order = self.sell()
                
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                pass  # Entry completed
            else:
                pass  # Exit completed
                
        self.order = None
