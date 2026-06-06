"""
CorrArb MTF v7 — رفع کامل باگ تایید
"""

import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class Config:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.04
    max_total_dd_pct   = 0.08

    risk_base_pct = 0.008
    risk_min_pct  = 0.004

    spread_pips        = 1.2
    commission_per_lot = 7.0
    slippage_pips      = 0.3

    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 2.0
    min_lot  = 0.01
    warmup   = 500

    # 15min
    z_fast_period   = 96
    z_slow_period   = 384
    z_entry         = 1.8
    z_exit          = 0.5
    z_slow_confirm  = 0.6
    adx_max         = 28
    rsi_long_max    = 45
    rsi_short_min   = 55
    atr_period      = 14
    atr_ma_period   = 96
    atr_max_mult    = 2.5
    atr_min_mult    = 0.4
    corr_window     = 48
    corr_min        = 0.65
    std_min_pct     = 0.20
    hour_start      = 7
    hour_end        = 18
    trade_days      = [0, 1, 2, 3]

    # 1min تایید — ساده‌شده
    confirm_bars        = 15     # حداکثر ۱۵ کندل 1min
    confirm_rsi_lo      = 30     # برای Long: RSI > این
    confirm_rsi_hi      = 70     # برای Short: RSI < این
    confirm_mom_bars    = 3      # momentum ۳ کندل
    confirm_vol_ma      = 20     # میانگین حجم
    confirm_vol_mult    = 1.0    # حجم > ۱× میانگین (خیلی ساده)
    confirm_body_min    = 0.20   # body حداقل ۲۰٪

    # SL/TP
    sl_lookback       = 8
    sl_buffer_pips    = 2.5
    sl_min_pips       = 8.0
    sl_max_pips       = 20.0
    tp_rr             = 2.2

    max_trades_day    = 2
    time_stop_bars    = 160
    trail_be_prog     = 0.50
    trail_be_pct      = 0.08
    trail_lock_prog   = 0.75
    trail_lock_pct    = 0.45
    consec_loss_n     = 3
    risk_reduce       = 0.65


# ═══════════════════════════════════════════════════════
def load_data():
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))

    def read_raw(paths, suffix):
        frames = []
        for p in paths:
            df = pd.read_csv(p, sep=';', header=None,
                             names=['ts','o','h','l','c','v'])
            df['ts'] = pd.to_datetime(df['ts'],
                                       format='%Y%m%d %H%M%S',
                                       errors='coerce')
            df = df.dropna(subset=['ts']).set_index('ts')
            df = df[~df.index.duplicated(keep='last')]
            for col in ['o','h','l','c','v']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            df.columns = [f'{c}_{suffix}' for c in df.columns]
            frames.append(df)
            print(f"    ✓ {p.split('/')[-1]}: {len(df):,}")
        return pd.concat(frames).sort_index()

    print("  EURUSD...")
    eur = read_raw(files_eur, 'eur')
    print("  GBPUSD...")
    gbp = read_raw(files_gbp, 'gbp')
    raw = eur.join(gbp, how='inner').dropna()
    print(f"  مشترک: {len(raw):,} | "
          f"{raw.index[0].date()} → {raw.index[-1].date()}")

    def resample(raw_df, freq):
        df = pd.DataFrame({
            'o_eur': raw_df['o_eur'].resample(freq).first(),
            'h_eur': raw_df['h_eur'].resample(freq).max(),
            'l_eur': raw_df['l_eur'].resample(freq).min(),
            'c_eur': raw_df['c_eur'].resample(freq).last(),
            'v_eur': raw_df['v_eur'].resample(freq).sum(),
            'o_gbp': raw_df['o_gbp'].resample(freq).first(),
            'h_gbp': raw_df['h_gbp'].resample(freq).max(),
            'l_gbp': raw_df['l_gbp'].resample(freq).min(),
            'c_gbp': raw_df['c_gbp'].resample(freq).last(),
            'v_gbp': raw_df['v_gbp'].resample(freq).sum(),
        }).dropna()
        df = df[(df.index.weekday < 5) & (df['c_eur'] > 0)]
        return df

    df15 = resample(raw, '15min')
    df1  = resample(raw, '1min')

    print(f"✅ 15min: {len(df15):,} | {df15.index[0].date()} → {df15.index[-1].date()}")
    print(f"✅  1min: {len(df1):,}  | {df1.index[0].date()} → {df1.index[-1].date()}")
    return df15, df1


# ═══════════════════════════════════════════════════════
def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    ls= (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100/(1 + g/ls.replace(0, np.nan))

def calc_adx(h, l, c, p=14):
    up  = h.diff(); dn = -l.diff()
    dmp = up.where((up>dn)&(up>0), 0.0)
    dmn = dn.where((dn>up)&(dn>0), 0.0)
    s   = calc_atr(h,l,c,1).rolling(p).sum().replace(0,np.nan)
    dip = 100*dmp.rolling(p).sum()/s
    din = 100*dmn.rolling(p).sum()/s
    dx  = (abs(dip-din)/(dip+din).replace(0,np.nan))*100
    return dx.rolling(p).mean()


# ═══════════════════════════════════════════════════════
def compute_15min_signals(df15):
    print("  [15min] سیگنال‌ها...", end="", flush=True)
    C   = Config
    c_e = df15['c_eur']; h_e = df15['h_eur']
    l_e = df15['l_eur']; c_g = df15['c_gbp']

    rsi_e = calc_rsi(c_e, 14)
    rsi_g = calc_rsi(c_g, 14)
    adx   = calc_adx(h_e, l_e, c_e, 14)
    atr   = calc_atr(h_e, l_e, c_e, C.atr_period)
    atr_ma= atr.rolling(C.atr_ma_period).mean()

    ratio  = c_e/c_g
    z_mf   = ratio.rolling(C.z_fast_period).mean()
    z_sf   = ratio.rolling(C.z_fast_period).std()
    z_fast = (ratio-z_mf)/z_sf.replace(0,np.nan)
    z_ms   = ratio.rolling(C.z_slow_period).mean()
    z_ss   = ratio.rolling(C.z_slow_period).std()
    z_slow = (ratio-z_ms)/z_ss.replace(0,np.nan)

    corr   = c_e.pct_change().rolling(C.corr_window).corr(c_g.pct_change())
    std_ok = z_sf > z_sf.rolling(C.z_slow_period).mean()*C.std_min_pct
    vol_ok = (atr>atr_ma*C.atr_min_mult)&(atr<atr_ma*C.atr_max_mult)
    hour   = pd.Series(df15.index.hour, index=df15.index)
    dow    = pd.Series(df15.index.dayofweek, index=df15.index)
    tok    = hour.between(C.hour_start,C.hour_end)&dow.isin(C.trade_days)
    div12  = c_e.pct_change(48)-c_g.pct_change(48)

    sig = pd.Series(0, index=df15.index)
    sig[(z_fast<-C.z_entry)&(z_slow<-C.z_slow_confirm)&
        (div12<-0.0005)&std_ok&vol_ok&tok&
        (adx<C.adx_max)&(corr>C.corr_min)&
        (rsi_e<C.rsi_long_max)&(rsi_e<rsi_g-5)] = 1
    sig[(z_fast>C.z_entry)&(z_slow>C.z_slow_confirm)&
        (div12>0.0005)&std_ok&vol_ok&tok&
        (adx<C.adx_max)&(corr>C.corr_min)&
        (rsi_e>C.rsi_short_min)&(rsi_e>rsi_g+5)] = -1
    sig = sig.where(sig!=sig.shift(), 0)

    print(f" ✓  {int((sig!=0).sum())} "
          f"(L:{int((sig==1).sum())}, S:{int((sig==-1).sum())})")
    return {'sig': sig, 'z_fast': z_fast, 'atr15': atr}


# ═══════════════════════════════════════════════════════
def build_1min_index(df1: pd.DataFrame) -> pd.DataFrame:
    """
    پیش‌محاسبه همه اندیکاتورهای 1min.
    ✅ causal: rolling از گذشته
    """
    print("  [ 1min] اندیکاتورها...", end="", flush=True)
    C   = Config
    c   = df1['c_eur']
    h   = df1['h_eur']
    l   = df1['l_eur']
    o   = df1['o_eur']
    v   = df1['v_eur']

    atr    = calc_atr(h, l, c, 14)
    rsi    = calc_rsi(c, C.confirm_rsi_lo)   # RSI سریع
    mom    = c.pct_change(C.confirm_mom_bars)
    vol_ma = v.rolling(C.confirm_vol_ma).mean()
    vol_r  = v / vol_ma.replace(0, np.nan)
    rng    = (h - l).replace(0, np.nan)
    body   = (c - o) / rng

    out = pd.DataFrame({
        'c': c, 'h': h, 'l': l, 'o': o,
        'atr': atr, 'rsi': rsi, 'mom': mom,
        'vol_r': vol_r, 'body': body,
    }, index=df1.index)

    print(f" ✓  {len(out):,}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  تابع تایید 1min — ساده و مستقیم
#
#  ✅ بدون look-ahead:
#    - ts_15close: زمان بسته شدن کندل 15min سیگنال
#    - ts_15next:  زمان بسته شدن کندل 15min بعدی
#    - فقط کندل‌های 1min در بازه (ts_15close, ts_15next) بررسی می‌شوند
#    - ورود: open کندل 1min بعد از کندل تایید
#    - چون ورود روی open کندل بعدی است، هیچ look-ahead وجود ندارد
# ═══════════════════════════════════════════════════════════════════════════
def find_entry_1min(
        direction:   int,
        ts_15close:  pd.Timestamp,   # بسته شدن کندل 15min سیگنال
        ts_15next:   pd.Timestamp,   # بسته شدن کندل 15min بعدی
        ind1:        pd.DataFrame,
) -> tuple:
    """
    Returns: (confirmed, sl_pips, entry_price, entry_ts)

    entry_price: قیمت open کندل 1min بعد از تایید
    entry_ts:    timestamp کندل 1min که روی آن وارد می‌شویم
    """
    C = Config

    # ── کندل‌های 1min بین دو کندل 15min ──
    # ts_15close < index <= ts_15next
    mask   = (ind1.index > ts_15close) & (ind1.index <= ts_15next)
    window = ind1[mask]

    if len(window) < 2:
        return False, C.sl_min_pips, 0.0, None

    # بررسی هر کندل (به جز آخری که برای ورود است)
    for i in range(len(window) - 1):
        row      = window.iloc[i]
        next_row = window.iloc[i + 1]

        # skip اگر NaN
        if pd.isna(row['rsi']) or pd.isna(row['mom']) or pd.isna(row['body']):
            continue

        # ── شرط تایید ──
        if direction == 1:    # Long: کندل صعودی با momentum مثبت
            ok = (
                row['mom']  > 0 and
                row['body'] > C.confirm_body_min and
                row['rsi']  > C.confirm_rsi_lo
            )
        else:                  # Short: کندل نزولی با momentum منفی
            ok = (
                row['mom']  < 0 and
                row['body'] < -C.confirm_body_min and
                row['rsi']  < C.confirm_rsi_hi
            )

        if ok:
            # ── محاسبه SL از swing ──
            # N کندل 1min قبل از تایید (causal)
            prev = ind1[ind1.index <= row.name].tail(C.sl_lookback)

            entry_price = float(next_row['o'])   # open کندل بعدی

            if len(prev) > 0 and entry_price > 0:
                if direction == 1:
                    swing   = prev['l'].min()
                    sl_dist = (entry_price - swing) / C.pip
                else:
                    swing   = prev['h'].max()
                    sl_dist = (swing - entry_price) / C.pip

                sl_pips = float(np.clip(
                    sl_dist + C.sl_buffer_pips,
                    C.sl_min_pips, C.sl_max_pips
                ))
            else:
                sl_pips = C.sl_min_pips

            return True, sl_pips, entry_price, next_row.name

    return False, C.sl_min_pips, 0.0, None


# ═══════════════════════════════════════════════════════
def trade_cost(lot):
    C = Config
    return C.spread_pips*2*C.pip*lot*C.lot_size + C.commission_per_lot*lot

def calc_lot(equity, sl_pips, consec_loss):
    C    = Config
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n:
        risk = max(risk*C.risk_reduce, C.risk_min_pct)
    if sl_pips <= 0: return C.min_lot
    return round(float(np.clip(
        equity*risk/(sl_pips*C.pip*C.lot_size),
        C.min_lot, C.max_lot)), 2)

def check_prop(equity, day_start, floor):
    C = Config
    if day_start > 0:
        dd = (equity-day_start)/day_start
        if dd <= -C.max_daily_loss_pct:
            return True, f"DailyDD {dd*100:.2f}%"
    if equity <= floor:
        dd = (equity-C.initial_balance)/C.initial_balance
        return True, f"TotalDD {dd*100:.2f}%"
    return False, ""

def new_acc(ts):
    C = Config
    return dict(equity=C.initial_balance, start_ts=ts, trades=[],
                open_pos=None, blown=False, blown_rsn="",
                peak=C.initial_balance, min_eq=C.initial_balance,
                max_dd_pct=0.0, consec_loss=0, consec_win=0)

def upd_dd(acc):
    eq = acc['equity']
    if eq > acc['peak']:   acc['peak']   = eq
    if eq < acc['min_eq']: acc['min_eq'] = eq
    if acc['peak'] > 0:
        dd = (eq-acc['peak'])/acc['peak']*100
        if dd < acc['max_dd_pct']: acc['max_dd_pct'] = dd

def reg_acc(acc, end_ts, tw, num, reason, logs):
    C   = Config
    pnl = acc['equity']-C.initial_balance
    w   = sum(1 for t in acc['trades'] if t.get('pnl',0)>0)
    wr  = w/len(acc['trades'])*100 if acc['trades'] else 0
    logs.append(dict(account=num, start_ts=acc['start_ts'], end_ts=end_ts,
                     final=round(acc['equity'],2), pnl=round(pnl,2),
                     ret_pct=round(pnl/C.initial_balance*100,2),
                     trades=len(acc['trades']), wins=w, wr=round(wr,1),
                     reason=reason, total_withdrawn=round(tw,2),
                     max_dd_pct=round(acc['max_dd_pct'],4)))


# ═══════════════════════════════════════════════════════════════════════════
#  موتور بک‌تست MTF
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df15, signals15, ind1):
    """
    ✅ جریان صحیح بدون look-ahead:

    bar i (15min بسته می‌شود):
        ts_close = ts15[i]        ← زمان بسته شدن
        ts_next  = ts15[i+1]      ← زمان بسته شدن 15min بعدی

        کندل‌های 1min در (ts_close, ts_next] بررسی می‌شوند
        ورود روی open کندل 1min بعد از تایید

    bar i+1 (15min بعدی):
        اگر ورود در بازه قبلی انجام شده → position فعال است
        مدیریت پوزیشن روی این کندل
    """
    C   = Config
    pip = C.pip
    ls  = C.lot_size

    ts15_arr   = df15.index
    open15     = df15['o_eur'].values
    close15    = df15['c_eur'].values
    high15     = df15['h_eur'].values
    low15      = df15['l_eur'].values
    sig_arr    = signals15['sig'].values
    z_arr      = signals15['z_fast'].values

    FLOOR  = C.initial_balance*(1-C.max_total_dd_pct)
    TARGET = C.initial_balance*(1+C.profit_target_pct)

    tw=0.0; acc_num=1; logs=[]; trades=[]; eq_cv=[]; ts_cv=[]; tot_cv=[]
    acc=new_acc(ts15_arr[C.warmup])
    cur_day=None; day_eq=C.initial_balance; trades_today=0

    # pending: وقتی سیگنال داریم، در bar بعدی تایید 1min را بررسی می‌کنیم
    # ساختار: {'dir', 'ts_close', 'ts_next', 'bar'}
    pending = None

    n_conf=0; n_rej=0

    # دیکشنری سیگنال‌ها برای سرعت
    sig_d = {i: int(sig_arr[i])
             for i in range(C.warmup, len(ts15_arr)-1)
             if sig_arr[i] != 0}

    print(f"\n  FLOOR=${FLOOR:,.0f} | TARGET=${TARGET:,.0f}")

    for bar in range(C.warmup, len(ts15_arr)):
        ts  = ts15_arr[bar]
        day = ts.date()

        eq_cv.append(round(acc['equity'],4))
        ts_cv.append(ts)
        tot_cv.append(round(acc['equity']+tw,4))
        upd_dd(acc)

        if day != cur_day:
            cur_day=day; day_eq=acc['equity']; trades_today=0

        # ── Blown ──
        if acc['blown']:
            if acc['open_pos']:
                pos=acc['open_pos']
                cp=close15[bar]
                pnl=pos['dir']*(cp-pos['entry'])*pos['lot']*ls-trade_cost(pos['lot'])
                acc['equity']+=pnl
                acc['trades'].append({**pos,'exit':cp,'exit_ts':ts,
                                      'pnl':pnl,'status':'blown_close'})
                trades.append(acc['trades'][-1])
                acc['open_pos']=None
            reg_acc(acc,ts,tw,acc_num,acc['blown_rsn'],logs)
            print(f"    💥 #{acc_num:>3} | {ts.date()} | "
                  f"{acc['blown_rsn']}")
            acc_num+=1; acc=new_acc(ts); day_eq=acc['equity']
            trades_today=0; pending=None
            FLOOR=C.initial_balance*(1-C.max_total_dd_pct)
            TARGET=C.initial_balance*(1+C.profit_target_pct)
            continue

        # ══════════════════════════════════════════════════════
        #  بررسی pending signal — تایید 1min
        #
        #  pending از bar قبل ثبت شده:
        #    ts_close = زمان بسته شدن کندل 15min سیگنال
        #    ts_next  = ts (کندل 15min جاری = بعدی)
        #
        #  کندل‌های 1min در (ts_close, ts] بررسی می‌شوند
        # ══════════════════════════════════════════════════════
        if (pending is not None
                and acc['open_pos'] is None
                and not acc['blown']
                and trades_today < C.max_trades_day):

            sv       = pending['dir']
            ts_close = pending['ts_close']
            ts_next  = ts   # کندل 15min جاری = بسته شدن 15min بعدی

            confirmed, sl_pips, ep_raw, entry_ts = find_entry_1min(
                sv, ts_close, ts_next, ind1
            )

            if confirmed and entry_ts is not None and ep_raw > 0:
                # ✅ ep_raw: open کندل 1min که ورود می‌کنیم
                # slippage در جهت بدبینانه
                ep = ep_raw + sv*(C.slippage_pips+C.spread_pips/2)*pip
                sl = ep - sv*sl_pips*pip
                tp = ep + sv*sl_pips*C.tp_rr*pip
                lot= calc_lot(acc['equity'], sl_pips, acc['consec_loss'])

                # immediate SL check روی کندل 15min جاری
                imm = (sv==1 and low15[bar]<=sl) or (sv==-1 and high15[bar]>=sl)
                if not imm:
                    acc['open_pos'] = dict(
                        account=acc_num, dir=sv, lot=lot,
                        entry=ep, sl=sl, tp=tp,
                        sl_pips=round(sl_pips,1),
                        tp_pips=round(sl_pips*C.tp_rr,1),
                        entry_ts=entry_ts, entry_bar=bar,
                    )
                    trades_today+=1; n_conf+=1
                    print(f"    ✅ ورود | {entry_ts} | "
                          f"{'BUY' if sv==1 else 'SELL'} | "
                          f"SL={sl_pips:.1f}pip | lot={lot}")
                else:
                    n_rej+=1
            else:
                n_rej+=1

            pending = None

        # ── مدیریت پوزیشن (روی 15min) ──
        pos = acc['open_pos']
        if pos:
            hi=high15[bar]; lo=low15[bar]; cp=close15[bar]
            d=pos['dir']; ep=pos['entry']; sl=pos['sl']; tp=pos['tp']

            hit_sl=(d==1 and lo<=sl) or (d==-1 and hi>=sl)
            hit_tp=(d==1 and hi>=tp) or (d==-1 and lo<=tp)

            zn=z_arr[bar]
            if not np.isnan(zn) and abs(zn)<C.z_exit: hit_tp=True
            if hit_sl and hit_tp: hit_tp=False

            # intra-candle blown
            if not hit_sl:
                w_pnl=d*(sl-ep)*pos['lot']*ls-trade_cost(pos['lot'])
                blown,rsn=check_prop(acc['equity']+w_pnl,day_eq,FLOOR)
                if blown:
                    pnl=d*(sl-ep)*pos['lot']*ls-trade_cost(pos['lot'])
                    acc['equity']+=pnl
                    acc['trades'].append({**pos,'exit':sl,'exit_ts':ts,
                                          'pnl':pnl,'status':'blown_SL'})
                    trades.append(acc['trades'][-1])
                    acc['open_pos']=None
                    acc['blown']=True; acc['blown_rsn']=rsn
                    acc['consec_loss']+=1; acc['consec_win']=0
                    upd_dd(acc); continue

            # trailing stop
            td=abs(tp-ep)
            if td>0:
                prog=d*(cp-ep)/td
                if prog>=C.trail_be_prog:
                    be=ep+d*td*C.trail_be_pct
                    if d==1 and be>pos['sl']: pos['sl']=be
                    elif d==-1 and be<pos['sl']: pos['sl']=be
                if prog>=C.trail_lock_prog:
                    lk=ep+d*td*C.trail_lock_pct
                    if d==1 and lk>pos['sl']: pos['sl']=lk
                    elif d==-1 and lk<pos['sl']: pos['sl']=lk

            # time stop
            if bar-pos['entry_bar']>=C.time_stop_bars and not hit_tp and not hit_sl:
                pnl=d*(cp-ep)*pos['lot']*ls-trade_cost(pos['lot'])
                acc['equity']+=pnl
                st='TP_time' if pnl>0 else 'SL_time'
                acc['trades'].append({**pos,'exit':cp,'exit_ts':ts,
                                      'pnl':pnl,'status':st})
                trades.append(acc['trades'][-1])
                acc['open_pos']=None
                if pnl>0: acc['consec_win']+=1; acc['consec_loss']=0
                else: acc['consec_loss']+=1; acc['consec_win']=0
                upd_dd(acc)
                b,r=check_prop(acc['equity'],day_eq,FLOOR)
                acc['blown']=b; acc['blown_rsn']=r
                continue

            # sl/tp
            if hit_sl or hit_tp:
                xp=sl if hit_sl else tp
                st='SL' if hit_sl else 'TP'
                pnl=d*(xp-ep)*pos['lot']*ls-trade_cost(pos['lot'])
                acc['equity']+=pnl
                acc['trades'].append({**pos,'exit':xp,'exit_ts':ts,
                                      'pnl':pnl,'status':st})
                trades.append(acc['trades'][-1])
                acc['open_pos']=None
                if pnl>0: acc['consec_win']+=1; acc['consec_loss']=0
                else: acc['consec_loss']+=1; acc['consec_win']=0
                upd_dd(acc)
                b,r=check_prop(acc['equity'],day_eq,FLOOR)
                acc['blown']=b; acc['blown_rsn']=r

        # ── هدف ──
        if acc['equity']>=TARGET and acc['open_pos'] is None and not acc['blown']:
            w=acc['equity']-C.initial_balance; tw+=w
            reg_acc(acc,ts,tw,acc_num,"TARGET_HIT",logs)
            print(f"    💰 #{acc_num:>3} | {ts.date()} | "
                  f"${w:>7.2f} | کل: ${tw:>9.2f}")
            acc_num+=1; acc=new_acc(ts); day_eq=acc['equity']
            trades_today=0; pending=None
            FLOOR=C.initial_balance*(1-C.max_total_dd_pct)
            TARGET=C.initial_balance*(1+C.profit_target_pct)
            continue

        # ── سیگنال جدید → pending ──
        # ✅ ts_next = ts15_arr[bar+1] اگر وجود داشته باشد
        if (acc['open_pos'] is None and pending is None
                and not acc['blown']
                and bar in sig_d
                and trades_today < C.max_trades_day
                and bar+1 < len(ts15_arr)):
            pending = {
                'dir':      sig_d[bar],
                'ts_close': ts,                  # بسته شدن کندل جاری
                'ts_next':  ts15_arr[bar+1],     # کندل بعدی (برای محدوده 1min)
                'bar':      bar,
            }

    # پایان
    if acc['open_pos']:
        pos=acc['open_pos']; cp=close15[-1]
        pnl=pos['dir']*(cp-pos['entry'])*pos['lot']*ls-trade_cost(pos['lot'])
        acc['equity']+=pnl
        acc['trades'].append({**pos,'exit':cp,'exit_ts':ts15_arr[-1],
                              'pnl':pnl,'status':'EndOfData'})
        trades.append(acc['trades'][-1])
        acc['open_pos']=None

    reg_acc(acc,ts15_arr[-1],tw,acc_num,"ACTIVE/END",logs)

    tot=n_conf+n_rej
    rate=f"{n_conf/tot*100:.1f}%" if tot>0 else "N/A"
    print(f"\n  تایید: {n_conf} | رد: {n_rej} | نرخ: {rate}")

    return dict(all_trades=trades, account_logs=logs,
                eq_curve=eq_cv, eq_ts=ts_cv, total_curve=tot_cv,
                total_withdrawn=tw, final_equity=acc['equity'],
                total_accounts=acc_num, n_conf=n_conf, n_rej=n_rej)


# ═══════════════════════════════════════════════════════
def compute_stats(res):
    if not res['all_trades']: return None
    C  = Config
    t  = pd.DataFrame(res['all_trades'])
    t['pnl']      = pd.to_numeric(t['pnl'],errors='coerce').fillna(0)
    t['entry_ts'] = pd.to_datetime(t['entry_ts'])
    t['exit_ts']  = pd.to_datetime(t['exit_ts'])
    t['dur_min']  = (t['exit_ts']-t['entry_ts']).dt.total_seconds()/60
    al = pd.DataFrame(res['account_logs'])

    tw=res['total_withdrawn']; feq=res['final_equity']
    tv=tw+feq; tp_=tv-C.initial_balance; tr=tp_/C.initial_balance*100
    sd=t['entry_ts'].min(); ed=t['exit_ts'].max()
    td=max((ed-sd).days,1)
    ar=((tv/C.initial_balance)**(365.25/td)-1)*100

    wt=t[t['pnl']>0]; lt=t[t['pnl']<0]
    wr=len(wt)/len(t)*100 if len(t) else 0
    aw=wt['pnl'].mean() if len(wt) else 0
    al_=lt['pnl'].mean() if len(lt) else 0
    pf=wt['pnl'].sum()/abs(lt['pnl'].sum()) if lt['pnl'].sum()!=0 else np.inf
    rr=abs(aw/al_) if al_!=0 else 0

    mdd=al['max_dd_pct'].min() if 'max_dd_pct' in al.columns else 0.0
    rc=pd.Series(res['total_curve']).pct_change().dropna()
    sh=rc.mean()/rc.std()*np.sqrt(252*96) if rc.std()>0 else 0
    neg=rc[rc<0]
    so=rc.mean()/neg.std()*np.sqrt(252*96) if len(neg)>1 else 0

    nt=int((al['reason']=='TARGET_HIT').sum())
    nb=int(al['reason'].str.contains('DailyDD|TotalDD|blown',
                                      case=False,na=False).sum())
    na=int((al['reason']=='ACTIVE/END').sum())

    sg=t['pnl'].apply(lambda x:1 if x>0 else -1 if x<0 else 0)
    cw=cl=mcw=mcl=0
    for s in sg:
        if s>0:   cw+=1;cl=0;mcw=max(mcw,cw)
        elif s<0: cl+=1;cw=0;mcl=max(mcl,cl)
        else:     cw=cl=0

    t['ym']=t['entry_ts'].dt.to_period('M')
    mg=t.groupby('ym').agg(n=('pnl','count'),pnl=('pnl','sum'),
                            wins=('pnl',lambda x:(x>0).sum())).reset_index()
    mg['wr']=mg['wins']/mg['n']*100
    mg['ret']=mg['pnl']/C.initial_balance*100

    return dict(
        trades=t, acc_logs=al, monthly=mg,
        eq_curve=res['eq_curve'], eq_ts=res['eq_ts'],
        total_curve=res['total_curve'],
        tw=tw, feq=feq, tv=tv, tp_=tp_, tr=tr, ar=ar, td=td,
        wr=wr, aw=aw, al_=al_, pf=pf, rr=rr,
        exp=t['pnl'].mean(), mdd=mdd, sh=sh, so=so,
        mcw=mcw, mcl=mcl, na_=res['total_accounts'],
        nt=nt, nb=nb, na=na,
        avg_dur=t['dur_min'].mean(),
        nc=res['n_conf'], nr=res['n_rej'],
        avg_m=mg['ret'].mean(), std_m=mg['ret'].std(),
        best_m=mg['ret'].max(), worst_m=mg['ret'].min(),
        pos_m=int((mg['pnl']>0).sum()), neg_m=int((mg['pnl']<=0).sum()),
    )


# ═══════════════════════════════════════════════════════
def print_report(s):
    C=Config; W=84; SEP="═"*W

    def rw(l,v,ok=None):
        lb=f"  {l}"; vl=str(v)
        m="" if ok is None else (" ✅" if ok else " ❌")
        d="·"*max(2,W-len(lb)-len(vl)-len(m)-2)
        return f"{lb} {d} {vl}{m}"
    def box(t):
        i=f"─ {t} "; return "┌"+i+"─"*(W-len(i)-1)+"┐"
    bot="└"+"─"*(W-1)+"┘"

    passed=all([abs(s['mdd'])<=8,s['pf']>1.3,s['nb']==0,s['nt']>0,s['ar']>10])
    flag="✅ PROP PASS" if passed else "⚠️  NEEDS REVIEW"
    nc=s['nc']; nr=s['nr']
    rate=f"{nc/(nc+nr)*100:.1f}%" if nc+nr>0 else "N/A"

    lines=["",SEP,
           f"  ▌  CorrArb MTF v7  [15min+1min]  —  {flag}  ▐",
           f"  ▌  {s['trades']['entry_ts'].min().date()} → "
           f"{s['trades']['exit_ts'].max().date()}  ({s['td']} روز)  ▐",
           f"  ▌  تایید 1min: {nc} | رد: {nr} | نرخ: {rate}  ▐",
           SEP,"",
           box("نتایج مالی"),
           rw("بالانس هر اکانت",    f"${C.initial_balance:>12,.2f}"),
           rw("کل سود برداشت‌شده",  f"${s['tw']:>+12,.2f}"),
           rw("موجودی فعلی",        f"${s['feq']:>12,.2f}"),
           rw("ارزش کل",            f"${s['tv']:>12,.2f}"),
           rw("سود خالص",           f"${s['tp_']:>+12,.2f}"),
           rw("بازده کل",           f"{s['tr']:>+.2f}%"),
           rw("CAGR",f"{s['ar']:>+.2f}%", ok=s['ar']>10),
           bot,"",
           box("ریسک"),
           rw("Max DD per Account", f"{s['mdd']:.2f}%", ok=abs(s['mdd'])<=8),
           rw("Sharpe",             f"{s['sh']:.2f}"),
           rw("Sortino",            f"{s['so']:.2f}"),
           rw("Profit Factor",      f"{s['pf']:.2f}", ok=s['pf']>1.3),
           bot,"",
           box("پایداری ماهانه"),
           rw("میانگین ماهانه",     f"{s['avg_m']:>+.2f}%"),
           rw("انحراف معیار",       f"{s['std_m']:.2f}%"),
           rw("سودده / زیان‌ده",
              f"{s['pos_m']} / {s['neg_m']}"),
           rw("بهترین ماه",         f"{s['best_m']:>+.2f}%"),
           rw("بدترین ماه",         f"{s['worst_m']:>+.2f}%",
              ok=s['worst_m']>-5),
           bot,"",
           box("پراپ"),
           rw("کل اکانت‌ها",        f"{s['na_']}"),
           rw("✅ Target Hit",      f"{s['nt']}", ok=s['nt']>0),
           rw("💥 Blown",           f"{s['nb']}", ok=s['nb']==0),
           rw("نرخ موفقیت",
              f"{s['nt']/max(s['na_'],1)*100:.1f}%"),
           bot,"",
           box("معاملات"),
           rw("تعداد کل",           f"{len(s['trades']):,}"),
           rw("Win Rate",           f"{s['wr']:.1f}%", ok=s['wr']>=50),
           rw("Avg Win",            f"${s['aw']:>+.2f}"),
           rw("Avg Loss",           f"${s['al_']:>+.2f}"),
           rw("RR واقعی",           f"{s['rr']:.2f}"),
           rw("Expectancy",         f"${s['exp']:>+.2f}"),
           rw("Max Cons. Wins",     f"{s['mcw']}"),
           rw("Max Cons. Losses",   f"{s['mcl']}"),
           rw("میانگین مدت",        f"{s['avg_dur']:.0f} min"),
           bot,"",
           ]

    # اکانت‌ها
    lines.append(box("جزئیات اکانت‌ها"))
    lines.append(f"  {'#':>4}  {'شروع':>10}  {'پایان':>10}  "
                 f"{'PnL':>9}  {'Ret%':>6}  {'#T':>3}  "
                 f"{'WR%':>5}  {'MaxDD':>7}  وضعیت")
    lines.append("  "+"─"*(W-3))
    for _,r in s['acc_logs'].iterrows():
        rn=r['reason']
        ic=("💰 WITHDRAW" if rn=='TARGET_HIT' else
            "🔄 ACTIVE"  if rn=='ACTIVE/END'  else f"💥 {rn[:20]}")
        mdd=r.get('max_dd_pct',0.0)
        wn=" ⚠️" if abs(mdd)>5 else ""
        lines.append(f"  {int(r['account']):>4}  "
                     f"{str(r['start_ts'])[:10]:>10}  "
                     f"{str(r['end_ts'])[:10]:>10}  "
                     f"${r['pnl']:>+8.2f}  {r['ret_pct']:>+5.1f}%  "
                     f"{r['trades']:>3}  {r['wr']:>4.0f}%  "
                     f"{mdd:>+6.2f}%{wn}  {ic}")
    lines+=[bot,""]

    # ماهانه
    lines.append(box("ماهانه"))
    lines.append(f"  {'ماه':>7}  {'#T':>3}  {'WR%':>5}  "
                 f"{'PnL':>9}  {'Ret%':>7}  نتیجه")
    lines.append("  "+"─"*(W-3))
    for _,mr in s['monthly'].iterrows():
        ic="🟢" if mr['ret']>0 else "🔴"
        wn=" ⚠️" if mr['ret']<-4 else ""
        lines.append(f"  {str(mr['ym']):>7}  {int(mr['n']):>3}  "
                     f"{mr['wr']:>4.1f}%  ${mr['pnl']:>+8.2f}  "
                     f"{mr['ret']:>+6.2f}%  {ic}{wn}")
    lines+=[bot,""]

    # سالانه
    s['trades']['yr']=s['trades']['entry_ts'].dt.year
    yg=(s['trades'].groupby('yr')
        .agg(n=('pnl','count'),pnl=('pnl','sum'),
             wins=('pnl',lambda x:(x>0).sum())).reset_index())
    yg['wr']=yg['wins']/yg['n']*100
    yg['ret']=yg['pnl']/C.initial_balance*100
    lines.append(box("سالانه"))
    lines.append(f"  {'سال':>5}  {'#T':>5}  {'WR%':>5}  "
                 f"{'PnL':>11}  {'Ret%':>7}  نتیجه")
    lines.append("  "+"─"*(W-3))
    for _,yr in yg.iterrows():
        ic="🟢" if yr['ret']>0 else "🔴"
        lines.append(f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
                     f"{yr['wr']:>4.1f}%  ${yr['pnl']:>10.2f}  "
                     f"{yr['ret']:>+6.1f}%  {ic}")
    lines+=[bot,""]

    out="\n".join(lines)
    print(out)
    return out


# ═══════════════════════════════════════════════════════
def save_outputs(s, report_txt):
    C=Config
    with open("Report_MTF_v7.txt","w",encoding="utf-8") as f:
        f.write(report_txt)

    rows=[["CorrArb MTF v7 (15min+1min)"],
          [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
          [],["=== Summary ==="],
          ["CAGR%",round(s['ar'],2)],["PF",round(s['pf'],2)],
          ["WR%",round(s['wr'],1)],["MaxDD%",round(s['mdd'],2)],
          ["Confirmed",s['nc']],["Rejected",s['nr']],
          [],["=== Trades ==="],
          ["Acc","EntryTS","ExitTS","Side","Lot","Entry","SL","TP",
           "Exit","SL_pip","TP_pip","PnL","Status","DurMin"]]
    for _,tr in s['trades'].iterrows():
        rows.append([
            tr.get('account',''),
            str(tr['entry_ts'])[:16],str(tr['exit_ts'])[:16],
            'BUY' if tr.get('dir',0)==1 else 'SELL',
            tr.get('lot',''),
            round(float(tr.get('entry',0)),5),
            round(float(tr.get('sl',0)),5),
            round(float(tr.get('tp',0)),5),
            round(float(tr.get('exit',0)),5),
            round(float(tr.get('sl_pips',0)),1),
            round(float(tr.get('tp_pips',0)),1),
            round(float(tr['pnl']),2),
            tr.get('status',''),
            round(float(tr.get('dur_min',0)),0),
        ])
    pd.DataFrame(rows).to_csv("Report_MTF_v7.csv",
                               index=False,header=False,encoding="utf-8-sig")

    wc=[round(tv-ae,2) for tv,ae in zip(s['total_curve'],s['eq_curve'])]
    pd.DataFrame({'ts':s['eq_ts'],'equity':s['eq_curve'],
                  'withdrawn':wc,'total':s['total_curve']}
                 ).to_csv("eq_MTF_v7.csv",index=False,encoding="utf-8-sig")
    print("✅ Report_MTF_v7.txt | Report_MTF_v7.csv | eq_MTF_v7.csv")


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═"*84)
    print("  CorrArb MTF v7  —  15min فرصت + 1min تایید دقیق")
    print("═"*84)
    C=Config
    print(f"  Risk={C.risk_base_pct*100:.1f}%  |  "
          f"Target=+{C.profit_target_pct*100:.0f}%  |  "
          f"DailyDD=-{C.max_daily_loss_pct*100:.0f}%  |  "
          f"TotalDD=-{C.max_total_dd_pct*100:.0f}%")
    print(f"  Confirm: mom>{C.confirm_mom_bars}bar | "
          f"body>{C.confirm_body_min} | "
          f"window={C.confirm_bars}×1min")
    print("═"*84)

    t0=datetime.now()
    print("\n  ▶ بارگذاری...")
    df15,df1=load_data()

    print("\n  ▶ سیگنال‌ها...")
    sig15=compute_15min_signals(df15)
    ind1 =build_1min_index(df1)

    print("\n  ▶ شبیه‌سازی MTF...")
    t1=datetime.now()
    res=run_backtest(df15,sig15,ind1)
    dt=(datetime.now()-t1).total_seconds()
    print(f"\n  ⏱ {dt:.1f}s | معاملات:{len(res['all_trades'])} | "
          f"اکانت‌ها:{res['total_accounts']}")

    if not res['all_trades']:
        print("\n❌ معامله‌ای نشد.")
    else:
        st=compute_stats(res)
        if st:
            rpt=print_report(st)
            save_outputs(st,rpt)
    print(f"\n  ✅ کل: {(datetime.now()-t0).total_seconds():.1f}s")
