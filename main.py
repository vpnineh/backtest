import os
import glob
import zipfile
import datetime
import pandas as pd
import backtrader as bt

# ==========================================
# 1. DATA PIPELINE (AUTOMATED ZIP/CSV LOADER)
# ==========================================
def load_historical_data(pair, data_dir='data'):
    """
    Scans the data directory for ZIP and CSV files matching the pair,
    extracts them in-memory, parses HistData/ASCII formats, and returns a sorted Pandas DataFrame.
    """
    print(f"[*] Scanning '{data_dir}' for {pair} data...")
    
    zip_pattern = os.path.join(data_dir, f'*{pair}*.zip')
    csv_pattern = os.path.join(data_dir, f'*{pair}*.csv')
    
    zip_files = glob.glob(zip_pattern)
    csv_files = glob.glob(csv_pattern)
    
    all_dfs = []
    
    # Process ZIP files (HistData format usually uses ';' as separator)
    for zf in zip_files:
        print(f"    -> Extracting from: {os.path.basename(zf)}")
        with zipfile.ZipFile(zf, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.csv'):
                    with z.open(file_info) as f:
                        # Histdata ASCII format: 20100103 170000;1.430100;1.430400;1.430100;1.430400;0
                        df = pd.read_csv(f, sep=';', header=None,
                                         names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                                         engine='python', on_bad_lines='skip')
                        all_dfs.append(df)

    # Process Standalone CSV files
    for cf in csv_files:
        print(f"    -> Reading: {os.path.basename(cf)}")
        # Check standard comma or semicolon separator
        with open(cf, 'r') as f:
            first_line = f.readline()
        sep = ';' if ';' in first_line else ','
        
        df = pd.read_csv(cf, sep=sep, header=None,
                         names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                         engine='python', on_bad_lines='skip')
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"[!] No data found for {pair} in '{data_dir}' directory.")

    print(f"[*] Merging {len(all_dfs)} files and formatting dates...")
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Parse DateTime and Set Index
    # Format 'YYYYMMDD HHMMSS' matching Histdata
    master_df['datetime'] = pd.to_datetime(master_df['datetime'], format='%Y%m%d %H%M%S', errors='coerce')
    master_df.dropna(subset=['datetime'], inplace=True)
    master_df.set_index('datetime', inplace=True)
    
    # Sort chronologically (CRUCIAL for backtesting)
    master_df.sort_index(inplace=True)
    
    # Clean up exact matching columns for Backtrader PandasData
    master_df = master_df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"[*] Total Rows Loaded: {len(master_df)}")
    return master_df


# ==========================================
# 2. STRATEGY ENGINE (NO LOOK-AHEAD BIAS)
# ==========================================
class ProfessionalMartingale(bt.Strategy):
    params = (
        ('pair', 'AUDNZD'),          
        ('risk_pct', 0.002),         # 0.20% risk
        ('max_levels', 7),           
        ('grid_multipliers', [1.00, 1.35, 1.80, 2.40, 3.20, 4.30, 5.0]), 
        ('target_profit_pct', 0.01), # 1% basket target
        ('max_daily_dd', 0.03),      # 3%
        ('max_weekly_dd', 0.08),     # 8%
    )

    def __init__(self):
        # Timframes setup (Data0=M1, Data1=M5, Data2=M15, Data3=H4)
        self.m1 = self.datas[0]
        self.m5 = self.datas[1]
        self.m15 = self.datas[2]
        self.h4 = self.datas[3]

        # Indicators strictly bound to their respective timeframes
        self.ema200_h4 = bt.indicators.EMA(self.h4.close, period=200)

        self.rsi_m15 = bt.indicators.RSI(self.m15.close, period=14)
        self.bbands_m15 = bt.indicators.BollingerBands(self.m15.close, period=20, devfactor=2.0)
        self.atr100_m15 = bt.indicators.ATR(self.m15, period=100)
        self.adx_m15 = bt.indicators.ADX(self.m15, period=14)
        
        self.atr14_m5 = bt.indicators.ATR(self.m5, period=14)

        # State Variables
        self.basket_direction = 0  
        self.grid_levels_open = 0
        self.initial_lot = 0
        self.last_trade_price = 0.0
        self.initial_equity_basket = 0.0
        
        # Drawdown tracking
        self.day_start_equity = None
        self.week_start_equity = None
        self.last_day = None
        self.last_week = None
        self.trading_disabled_today = False
        self.trading_disabled_this_week = False
        
        self.pip_value = 0.0001 if 'JPY' not in self.p.pair else 0.01

    def start(self):
        self.initial_equity = self.broker.getvalue()
        self.day_start_equity = self.initial_equity
        self.week_start_equity = self.initial_equity

    def check_drawdowns(self):
        current_equity = self.broker.getvalue()
        current_date = self.data.datetime.date(0)
        current_week = current_date.isocalendar()[1]

        if self.last_day != current_date:
            self.day_start_equity = current_equity
            self.last_day = current_date
            self.trading_disabled_today = False

        if self.last_week != current_week:
            self.week_start_equity = current_equity
            self.last_week = current_week
            self.trading_disabled_this_week = False

        daily_dd = (self.day_start_equity - current_equity) / self.day_start_equity
        weekly_dd = (self.week_start_equity - current_equity) / self.week_start_equity

        if daily_dd > self.p.max_daily_dd:
            self.trading_disabled_today = True
        if weekly_dd > self.p.max_weekly_dd:
            self.trading_disabled_this_week = True

    def get_dynamic_grid_distance(self):
        dist = 0.8 * self.atr14_m5[0]
        dist_pips = dist / self.pip_value
        dist_pips = max(25, min(dist_pips, 45))
        return dist_pips * self.pip_value

    def get_initial_lot_size(self):
        return 10000  # Default 0.1 Lot size for standard accounts

    def check_candle_pattern(self, is_bullish):
        o1, c1 = self.m15.open[-1], self.m15.close[-1]
        o0, c0 = self.m15.open[0], self.m15.close[0]
        if is_bullish:
            return (c1 < o1) and (c0 > o0) and (c0 > o1) and (o0 < c1)
        return (c1 > o1) and (c0 < o0) and (c0 < o1) and (o0 > c1)

    def next(self):
        # Execute logic at M1 tick level
        self.check_drawdowns()
        
        if self.trading_disabled_today or self.trading_disabled_this_week:
            return

        pos = self.getposition()
        current_equity = self.broker.getvalue()
        
        # --- BASKET MANAGEMENT ---
        if self.basket_direction != 0 and pos.size != 0:
            unrealized_pnl = current_equity - self.initial_equity_basket
            target_profit = self.initial_equity_basket * self.p.target_profit_pct
            
            # Emergency Stop logic
            if self.adx_m15[0] > 35: 
                self.close()
                self.basket_direction = 0
                self.grid_levels_open = 0
                return

            # Target Reached
            if unrealized_pnl >= target_profit:
                self.close()
                self.basket_direction = 0
                self.grid_levels_open = 0
                return
                
            # Grid Scaling
            if self.grid_levels_open < self.p.max_levels:
                current_price = self.data.close[0]
                grid_dist = self.get_dynamic_grid_distance()
                
                # Basket Protection Limit
                if unrealized_pnl < (-2 * target_profit):
                    return 
                
                if self.basket_direction == 1 and current_price <= (self.last_trade_price - grid_dist):
                    new_lot = self.initial_lot * self.p.grid_multipliers[self.grid_levels_open]
                    self.buy(size=new_lot)
                    self.last_trade_price = current_price
                    self.grid_levels_open += 1
                        
                elif self.basket_direction == -1 and current_price >= (self.last_trade_price + grid_dist):
                    new_lot = self.initial_lot * self.p.grid_multipliers[self.grid_levels_open]
                    self.sell(size=new_lot)
                    self.last_trade_price = current_price
                    self.grid_levels_open += 1
            return

        # --- ENTRY CONDITIONS ---
        # Session Filter (London 08:00 - 17:00)
        curr_time = self.data.datetime.time(0)
        if not (datetime.time(8, 0) <= curr_time <= datetime.time(17, 0)):
            return

        # Wait for Indicators
        if len(self.ema200_h4) < 200 or len(self.atr100_m15) < 100:
            return

        price = self.m15.close[0]
        
        # Long Entry
        if price > self.ema200_h4[0]:
            if self.rsi_m15[0] < 28 and self.adx_m15[0] < 25:
                if price < self.bbands_m15.lines.bot[0]: 
                    if self.atr100_m15[0] < (1.3 * self.atr100_m15[-1]): 
                        if self.check_candle_pattern(is_bullish=True):
                            self.initial_equity_basket = current_equity
                            self.initial_lot = self.get_initial_lot_size()
                            self.buy(size=self.initial_lot)
                            self.basket_direction = 1
                            self.grid_levels_open = 1
                            self.last_trade_price = self.data.close[0]
                            return

        # Short Entry
        if price < self.ema200_h4[0]:
            if self.rsi_m15[0] > 72 and self.adx_m15[0] < 25:
                if price > self.bbands_m15.lines.top[0]:
                    if self.atr100_m15[0] < (1.3 * self.atr100_m15[-1]):
                        if self.check_candle_pattern(is_bullish=False):
                            self.initial_equity_basket = current_equity
                            self.initial_lot = self.get_initial_lot_size()
                            self.sell(size=self.initial_lot)
                            self.basket_direction = -1
                            self.grid_levels_open = 1
                            self.last_trade_price = self.data.close[0]
                            return

# ==========================================
# 3. EXECUTION BOOTSTRAPPER
# ==========================================
if __name__ == '__main__':
    pair_to_test = 'AUDNZD' # You can change this to EURGBP or EURUSD based on files
    
    print("="*50)
    print("Starting Professional Backtest Pipeline")
    print("="*50)
    
    try:
        df = load_historical_data(pair=pair_to_test, data_dir='data')
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.00005, margin=100.0, mult=1.0)
    
    # Add Data0 (M1 Feed)
    data_m1 = bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Minutes, compression=1)
    cerebro.adddata(data_m1)

    # Resample Timeframes (No lookahead bias)
    cerebro.resampledata(data_m1, timeframe=bt.TimeFrame.Minutes, compression=5)   # Data1
    cerebro.resampledata(data_m1, timeframe=bt.TimeFrame.Minutes, compression=15)  # Data2
    cerebro.resampledata(data_m1, timeframe=bt.TimeFrame.Minutes, compression=240) # Data3

    # Add Strategy & Analyzers
    cerebro.addstrategy(ProfessionalMartingale, pair=pair_to_test)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print("\n[*] Initializing Engine (This might take time based on RAM and CPU)...")
    results = cerebro.run()
    strat = results[0]

    # --- Print Professional Report ---
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    
    dd = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    print(f"Final Balance: ${cerebro.broker.getvalue():.2f}")
    print(f"Max Drawdown:  {dd.max.drawdown:.2f}%")
    
    if 'total' in trades and trades.total.closed > 0:
        print(f"Total Trades:  {trades.total.closed}")
        print(f"Win Rate:      {(trades.won.total / trades.total.closed) * 100:.2f}%")
    else:
        print("Total Trades:  0 (No entries triggered)")
    
    print("="*50)
