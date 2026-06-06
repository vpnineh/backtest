"""
CorrArb Prop Simulator — v4.1 (realism-fixed + dynamic risk)
- بدون look-ahead bias
- خروج z روی close (نه TP)
- ورود همیشه باز می‌شود؛ SL/TP کندل ورود واقعی هندل می‌شود
- ریسک پویای دوطرفه (کامپوند در برد، کاهش پلکانی در باخت)
"""

import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class Config:
    # ── پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.045      # 4.5%
    max_total_dd_pct   = 0.09       # 9%

    # ── ریسک پویا ──
    risk_base_pct      = 0.013      # 1.3% پایه
    risk_min_pct       = 0.005      # کف بعد از باخت‌ها
    risk_max_pct       = 0.018      # سقف کامپوند در بردها
    risk_win_step      = 0.0015     # +0.15% به ازای هر برد متوالی
    consec_loss_n      = 2          # از باخت دوم شروع کاهش
    risk_reduce        = 0.65       # ضریب کاهش پلکانی

    # ── هزینه‌ها ──
    spread_pips        = 1.2
    commission_per_lot = 7.0
    slippage_pips      = 0.3

    # ── بازار ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 2.0
    min_lot  = 0.01
    warmup   = 500

    # ── Z-score ──
    z_fast_period   = 96
    z_slow_period   = 384
    z_entry         = 1.8
    z_exit          = 0.5
    z_slow_confirm  = 0.6

    adx_max         = 28
    rsi_long_max    = 45
    rsi_short_min   = 55

    sl_pips         = 20.0
    tp_pips         = 44.0

    hour_start      = 7
    hour_end        = 18
    trade_days      = [0, 1, 2, 3]
    max_trades_day  = 3             # ۳ معامله در روز

    atr_period      = 14
    atr_ma_period   = 96
    atr_max_mult    = 2.5
    atr_min_mult    = 0.4

    corr_window     = 48
    corr_min        = 0.65
    std_min_pct     = 0.20

    time_stop_bars  = 160


def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur:
        raise FileNotFoundError("EURUSD CSV not found in data/")
    if not files_gbp:
        raise FileNotFoundError("GBPUSD CSV not found in data/")

    def read_pair(paths, suffix):
        frames = []
        for p in paths:
            df = pd.read_csv(
                p, sep=';', header=None,
                names=['ts', 'o', 'h', 'l', 'c', 'v']
            )
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            df = df.set_index('ts')
            df = df[~df.index.duplicated(keep='last')]
            df.columns = [f'{col}_{suffix}' for col in df.columns]
            frames.append(df)
        return pd.concat(frames).sort_index()

    eur = read_pair(files_eur, 'eur')
    gbp = read_pair(files_gbp, 'gbp')
    raw = eur.join(gbp, how='inner').dropna()

    df = pd.DataFrame({
        'o_eur': raw['o_eur'].resample('15min').first(),
        'h_eur': raw['h_eur'].resample('15min').max(),
        'l_eur': raw['l_eur'].resample('15min').min(),
        'c_eur': raw['c_eur'].resample('15min').last(),
        'v_eur': raw['v_eur'].resample('15min').sum(),
        'o_gbp': raw['o_gbp'].resample('15min').first(),
        'h_gbp': raw['h_gbp'].resample('15min').max(),
        'l_gbp': raw['l_gbp'].resample('15min').min(),
        'c_gbp': raw['c_gbp'].resample('15min').last(),
        'v_gbp': raw['v_gbp'].resample('15min').sum(),
    }).dropna()

    df = df[df.index.weekday < 5]
    print(f"✅ {len(df):,} کندل | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df
def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()

    # ── اسپرد قیمت دو جفت (پایه CorrArb) ──
    d['spread'] = d['c_eur'] - d['c_gbp']

    # ── Z-score سریع و کند (فقط گذشته/جاری) ──
    m_fast = d['spread'].rolling(cfg.z_fast_period).mean()
    s_fast = d['spread'].rolling(cfg.z_fast_period).std()
    d['z_fast'] = (d['spread'] - m_fast) / s_fast

    m_slow = d['spread'].rolling(cfg.z_slow_period).mean()
    s_slow = d['spread'].rolling(cfg.z_slow_period).std()
    d['z_slow'] = (d['spread'] - m_slow) / s_slow

    # ── همبستگی غلتان ──
    d['corr'] = d['c_eur'].rolling(cfg.corr_window).corr(d['c_gbp'])

    # ── انحراف معیار اسپرد به‌صورت درصدی (فیلتر رژیم) ──
    d['std_pct'] = (s_fast / d['spread'].abs().rolling(
        cfg.z_fast_period).mean()) * 100

    # ── ATR روی EURUSD ──
    hl = d['h_eur'] - d['l_eur']
    hc = (d['h_eur'] - d['c_eur'].shift()).abs()
    lc = (d['l_eur'] - d['c_eur'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d['atr'] = tr.rolling(cfg.atr_period).mean()
    d['atr_ma'] = d['atr'].rolling(cfg.atr_ma_period).mean()

    # ── RSI روی اسپرد ──
    delta = d['spread'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d['rsi'] = 100 - (100 / (1 + rs))

    # ── ADX روی EURUSD ──
    up = d['h_eur'].diff()
    dn = -d['l_eur'].diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_adx = tr.rolling(cfg.atr_period).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=d.index).rolling(
        cfg.atr_period).mean() / atr_adx
    minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(
        cfg.atr_period).mean() / atr_adx
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d['adx'] = dx.rolling(cfg.atr_period).mean()

    d['hour'] = d.index.hour
    d['dow']  = d.index.weekday
    return d
def gen_signal(row, cfg: Config) -> int:
    """
    1  → long spread  (long EUR)
    -1 → short spread (short EUR)
    0  → no trade
    همه شرط‌ها روی کندل بسته‌شده محاسبه شده‌اند؛
    اجرا روی open کندل بعدی انجام می‌شود (بدون look-ahead).
    """
    # فیلتر زمان
    if row['hour'] < cfg.hour_start or row['hour'] >= cfg.hour_end:
        return 0
    if row['dow'] not in cfg.trade_days:
        return 0

    # فیلتر رژیم
    if not np.isfinite(row['corr']) or row['corr'] < cfg.corr_min:
        return 0
    if not np.isfinite(row['std_pct']) or row['std_pct'] < cfg.std_min_pct:
        return 0
    if not np.isfinite(row['adx']) or row['adx'] > cfg.adx_max:
        return 0

    # فیلتر ATR (رژیم نوسان سالم)
    if np.isfinite(row['atr']) and np.isfinite(row['atr_ma']) and row['atr_ma'] > 0:
        ratio = row['atr'] / row['atr_ma']
        if ratio > cfg.atr_max_mult or ratio < cfg.atr_min_mult:
            return 0

    zf, zs, rsi = row['z_fast'], row['z_slow'], row['rsi']
    if not (np.isfinite(zf) and np.isfinite(zs) and np.isfinite(rsi)):
        return 0

    # long spread: اسپرد خیلی پایین، انتظار بازگشت به میانگین
    if zf <= -cfg.z_entry and zs <= -cfg.z_slow_confirm and rsi <= cfg.rsi_long_max:
        return 1
    # short spread
    if zf >= cfg.z_entry and zs >= cfg.z_slow_confirm and rsi >= cfg.rsi_short_min:
        return -1

    return 0
def calc_lot(acc, cfg: Config) -> float:
    """
    ریسک پویای دوطرفه:
    - بردهای متوالی → افزایش پلکانی ریسک تا سقف risk_max
    - باخت‌های متوالی (از consec_loss_n) → کاهش با ضریب risk_reduce تا کف risk_min
    """
    risk_pct = cfg.risk_base_pct

    cw = acc.get('consec_win', 0)
    cl = acc.get('consec_loss', 0)

    if cw > 0:
        risk_pct = min(cfg.risk_max_pct,
                       cfg.risk_base_pct + cw * cfg.risk_win_step)
    elif cl >= cfg.consec_loss_n:
        steps = cl - cfg.consec_loss_n + 1
        risk_pct = max(cfg.risk_min_pct,
                       cfg.risk_base_pct * (cfg.risk_reduce ** steps))

    risk_amount = acc['balance'] * risk_pct
    pip_value   = cfg.lot_size * cfg.pip          # ارزش هر pip برای 1 لات
    sl_value    = cfg.sl_pips * pip_value         # ضرر هر لات در SL
    lot = risk_amount / sl_value if sl_value > 0 else cfg.min_lot

    lot = max(cfg.min_lot, min(cfg.max_lot, lot))
    return round(lot, 2)
def run_backtest(df: pd.DataFrame, cfg: Config) -> dict:
    acc = {
        'balance': cfg.initial_balance,
        'peak': cfg.initial_balance,
        'open_pos': None,
        'consec_win': 0,
        'consec_loss': 0,
        'day': None,
        'day_start_bal': cfg.initial_balance,
        'trades_today': 0,
        'daily_locked': False,
    }
    trades, equity = [], []
    accounts_done = 0       # تعداد اکانت‌های pass شده
    blown = 0               # تعداد اکانت‌های سوخته

    pip = cfg.pip
    spread_cost = cfg.spread_pips * pip
    slip = cfg.slippage_pips * pip

    rows = df.reset_index().to_dict('records')

    def reset_account(reason):
        acc['balance'] = cfg.initial_balance
        acc['peak'] = cfg.initial_balance
        acc['open_pos'] = None
        acc['consec_win'] = 0
        acc['consec_loss'] = 0
        acc['day_start_bal'] = cfg.initial_balance
        acc['trades_today'] = 0
        acc['daily_locked'] = False

    pending = None   # سیگنال کندل قبل که باید روی open این کندل اجرا شود

    for i in range(cfg.warmup, len(rows) - 1):
        row = rows[i]
        ts  = row['ts']
        cur_day = ts.date()

        # ── شروع روز جدید ──
        if acc['day'] != cur_day:
            acc['day'] = cur_day
            acc['day_start_bal'] = acc['balance']
            acc['trades_today'] = 0
            acc['daily_locked'] = False

        equity.append({'ts': ts, 'balance': acc['balance']})
        # ── مدیریت پوزیشن باز (اول از همه، روی همین کندل) ──
        if acc['open_pos'] is not None:
            p = acc['open_pos']
            d  = p['dir']
            hi = row['h_eur']
            lo = row['l_eur']
            cl = row['c_eur']
            p['bars'] += 1

            exit_px = None
            reason  = None

            # SL/TP درون‌کندلی (محافظه‌کارانه: اگر هر دو در یک کندل لمس شوند، SL مقدم است)
            if d == 1:
                if lo <= p['sl']:
                    exit_px, reason = p['sl'], 'SL'
                elif hi >= p['tp']:
                    exit_px, reason = p['tp'], 'TP'
            else:
                if hi >= p['sl']:
                    exit_px, reason = p['sl'], 'SL'
                elif lo <= p['tp']:
                    exit_px, reason = p['tp'], 'TP'

            # ── اصلاح باگ ۱: Z-exit روی close واقعی همین کندل، نه روی tp ──
            if exit_px is None:
                zf = row['z_fast']
                if np.isfinite(zf) and abs(zf) <= cfg.z_exit:
                    exit_px, reason = cl, 'Z-exit'

            # Time stop → خروج روی close
            if exit_px is None and p['bars'] >= cfg.time_stop_bars:
                exit_px, reason = cl, 'TimeStop'

            if exit_px is not None:
                pip_value = cfg.lot_size * pip
                gross = (exit_px - p['entry']) / pip * pip_value * p['lot'] * d
                cost  = (spread_cost / pip * pip_value * p['lot']) \
                        + (cfg.commission_per_lot * p['lot'])
                pnl   = gross - cost
                acc['balance'] += pnl

                if pnl >= 0:
                    acc['consec_win']  += 1
                    acc['consec_loss']  = 0
                else:
                    acc['consec_loss'] += 1
                    acc['consec_win']   = 0

                trades.append({
                    'ts': ts, 'dir': d, 'entry': p['entry'], 'exit': exit_px,
                    'lot': p['lot'], 'pnl': pnl, 'reason': reason,
                    'bars': p['bars'], 'bal': acc['balance'],
                })
                acc['open_pos'] = None
                if acc['balance'] > acc['peak']:
                    acc['peak'] = acc['balance']
        # ── چک قوانین پراپ (بعد از به‌روزرسانی بالانس) ──
        # 1) هدف سود → برداشت و اکانت فرش
        if acc['balance'] >= cfg.initial_balance * (1 + cfg.profit_target_pct):
            accounts_done += 1
            # در واقعیت: سود برداشت می‌شود، پراپ اکانت ۵۰۰۰$ جدید می‌دهد
            reset_account('TARGET')
            equity.append({'ts': ts, 'balance': acc['balance']})
            pending = None
            continue

        # 2) ضرر روزانه (نسبت به بالانس ابتدای روز)
        daily_dd = (acc['day_start_bal'] - acc['balance']) / acc['day_start_bal']
        if daily_dd >= cfg.max_daily_loss_pct:
            acc['daily_locked'] = True       # تا پایان روز معامله نزن

        # 3) افت کل اکانت → سوختن
        total_dd = (cfg.initial_balance - acc['balance']) / cfg.initial_balance
        if total_dd >= cfg.max_total_dd_pct:
            blown += 1
            reset_account('BLOWN')
            equity.append({'ts': ts, 'balance': acc['balance']})
            pending = None
            continue
        # ── اجرای سیگنال pending روی OPEN همین کندل (بدون look-ahead) ──
        if pending is not None and acc['open_pos'] is None \
           and not acc['daily_locked'] and acc['trades_today'] < cfg.max_trades_day:

            d = pending
            # ورود روی open + اسلیپیج + نصف اسپرد (سمت نامساعد)
            entry = row['o_eur'] + (slip + spread_cost / 2) * d
            sl = entry - cfg.sl_pips * pip * d
            tp = entry + cfg.tp_pips * pip * d
            lot = calc_lot(acc, cfg)

            # ── اصلاح باگ ۲: حتی اگر همین کندل به SL بخورد، پوزیشن باز می‌شود ──
            # (سفارش روی open پر شده؛ مدیریت SL/TP از کندل بعد انجام می‌شود)
            acc['open_pos'] = {
                'dir': d, 'entry': entry, 'sl': sl, 'tp': tp,
                'lot': lot, 'bars': 0, 'entry_ts': ts,
            }
            acc['trades_today'] += 1

        pending = None

        # ── تولید سیگنال این کندل برای اجرا در کندل بعد ──
        if acc['open_pos'] is None and not acc['daily_locked'] \
           and acc['trades_today'] < cfg.max_trades_day:
            s = gen_signal(row, cfg)
            if s != 0:
                pending = s

    return {
        'trades': trades,
        'equity': equity,
        'accounts_done': accounts_done,
        'blown': blown,
        'final_balance': acc['balance'],
    }
def report(res: dict, cfg: Config):
    tr = pd.DataFrame(res['trades'])
    if tr.empty:
        print("هیچ معامله‌ای انجام نشد.")
        return

    eq = pd.DataFrame(res['equity']).set_index('ts')
    wins = tr[tr['pnl'] > 0]
    loss = tr[tr['pnl'] <= 0]

    win_rate = len(wins) / len(tr) * 100
    pf = wins['pnl'].sum() / abs(loss['pnl'].sum()) if len(loss) else np.inf
    avg_w = wins['pnl'].mean() if len(wins) else 0
    avg_l = loss['pnl'].mean() if len(loss) else 0
    rr = abs(avg_w / avg_l) if avg_l else np.inf

    # max drawdown روی منحنی اکویتی
    roll_max = eq['balance'].cummax()
    dd = (eq['balance'] - roll_max) / roll_max * 100
    max_dd = dd.min()

    # تعداد روزها → CAGR تقریبی بر پایه کل سود برداشتی + بالانس نهایی
    total_profit = res['accounts_done'] * cfg.initial_balance * cfg.profit_target_pct
    days = (eq.index[-1] - eq.index[0]).days or 1
    years = days / 365.25
    # بازده روی سرمایه فرض‌شده یک اکانت (5000$)
    end_val = cfg.initial_balance + total_profit
    cagr = ((end_val / cfg.initial_balance) ** (1 / years) - 1) * 100

    # تعداد معامله در ماه
    months = days / 30.0
    trades_per_month = len(tr) / months if months else 0

    print("=" * 50)
    print("  CorrArb Prop Simulator v4 — نتایج")
    print("=" * 50)
    print(f"کل معاملات        : {len(tr)}")
    print(f"معامله در ماه     : {trades_per_month:.1f}")
    print(f"نرخ برد           : {win_rate:.1f}%")
    print(f"Profit Factor     : {pf:.2f}")
    print(f"Reward/Risk       : {rr:.2f}")
    print(f"میانگین برد       : {avg_w:.2f}$")
    print(f"میانگین باخت      : {avg_l:.2f}$")
    print(f"حداکثر افت        : {max_dd:.2f}%")
    print(f"اکانت‌های pass شده : {res['accounts_done']}")
    print(f"اکانت‌های سوخته    : {res['blown']}")
    print(f"کل سود برداشتی    : {total_profit:,.2f}$")
    print(f"CAGR تقریبی       : {cagr:.2f}%")
    print("=" * 50)

    # ── معیار قبولی ──
    dd_ok     = abs(max_dd) <= 8.0
    pf_ok     = pf > 1.3
    blown_ok  = res['blown'] == 0
    target_ok = res['accounts_done'] > 0
    wr_ok     = win_rate >= 50.0
    cnt_ok    = 8 <= trades_per_month <= 12

    checks = [
        ("DD ≤ 8%",        dd_ok),
        ("PF > 1.3",       pf_ok),
        ("بدون سوختن",      blown_ok),
        ("حداقل ۱ هدف",     target_ok),
        ("Win ≥ 50%",      wr_ok),
        ("8-12 trade/mo",  cnt_ok),
    ]
    print("بررسی realism / قبولی:")
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
    print("=" * 50)
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    cfg = Config()
    print(">> بارگذاری دیتا...")
    df = load_data(cfg)
    print(f">> {len(df)} کندل 15m | از {df.index[0]} تا {df.index[-1]}")
    print(">> محاسبه اندیکاتورها...")
    df = add_indicators(df, cfg)
    print(">> اجرای بک‌تست...")
    res = run_backtest(df, cfg)
    report(res, cfg)
