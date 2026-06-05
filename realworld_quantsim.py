import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ================================================================== #
#                         CONFIG                                     #
# ================================================================== #
class Config:
    initial_balance    = 5_000.0
    risk_per_trade_pct = 0.012      # 1.2% - کمی بالاتر برای سود بیشتر
    max_daily_loss_pct = 0.04
    max_total_dd_pct   = 0.08
    monthly_target_pct = 0.10
    spread_eur_pips    = 1.0
    spread_gbp_pips    = 1.2
    commission_per_lot = 6.0
    pip                = 0.0001
    lot_size           = 100_000
    max_lot            = 2.0
    warmup             = 300


# ================================================================== #
#                     داده                                          #
# ================================================================== #
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur or not files_gbp:
        raise FileNotFoundError("فایل CSV پیدا نشد در data/")

    def read(paths, suffix):
        frames = []
        for p in paths:
            df = pd.read_csv(p, sep=';', header=None,
                             names=['ts','o','h','l','c','v'])
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            df = df.set_index('ts')
            df = df[~df.index.duplicated(keep='last')]
            df.columns = [f'{c}_{suffix}' for c in df.columns]
            frames.append(df)
        return pd.concat(frames).sort_index()

    eur = read(files_eur, 'eur')
    gbp = read(files_gbp, 'gbp')
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


# ── اندیکاتورها ──
def calc_atr(h, l, c, p=14):
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/p,adjust=False).mean()
    lo= (-d.clip(upper=0)).ewm(alpha=1/p,adjust=False).mean()
    return 100 - 100/(1+g/lo.replace(0,np.nan))

def calc_adx(h, l, c, p=14):
    up=h.diff(); dn=-l.diff()
    dmp=up.where((up>dn)&(up>0),0.0)
    dmn=dn.where((dn>up)&(dn>0),0.0)
    tr=calc_atr(h,l,c,1)
    s=tr.rolling(p).sum()
    dip=100*dmp.rolling(p).sum()/s.replace(0,np.nan)
    din=100*dmn.rolling(p).sum()/s.replace(0,np.nan)
    dx=(abs(dip-din)/(dip+din).replace(0,np.nan))*100
    return dx.rolling(p).mean()

def calc_macd(c,f=12,s=26,sg=9):
    m=c.ewm(span=f,adjust=False).mean()-c.ewm(span=s,adjust=False).mean()
    sig=m.ewm(span=sg,adjust=False).mean()
    return m, sig, m-sig

def lot_size_calc(equity, sl_pips):
    if sl_pips<=0: return 0.01
    lot=equity*Config.risk_per_trade_pct/(sl_pips*Config.pip*Config.lot_size)
    return round(float(np.clip(lot,0.01,Config.max_lot)),2)


# ================================================================== #
#   سیگنال‌سازی - رویکرد جدید: بدون Regime Filter                  #
#   ─────────────────────────────────────────────────────────────── #
#   درس گرفته شده: Regime filter خیلی محدودکننده بود               #
#   راه‌حل: هر استراتژی فیلترهای خودش را دارد                      #
#   و همه در Mixed regime هم می‌توانند کار کنند                     #
# ================================================================== #
def compute_signals(df: pd.DataFrame) -> dict:
    print("  محاسبه سیگنال‌ها...", end="", flush=True)

    c_e=df['c_eur']; h_e=df['h_eur']
    l_e=df['l_eur']; c_g=df['c_gbp']

    atr    = calc_atr(h_e,l_e,c_e,14)
    rsi    = calc_rsi(c_e,14)
    adx    = calc_adx(h_e,l_e,c_e,14)
    ema21  = c_e.ewm(span=21, adjust=False).mean()
    ema50  = c_e.ewm(span=50, adjust=False).mean()
    ema200 = c_e.ewm(span=200,adjust=False).mean()
    _,_,macd_h = calc_macd(c_e)

    hour    = pd.Series(df.index.hour,   index=df.index)
    weekday = pd.Series(df.index.weekday,index=df.index)

    # ══════════════════════════════════════════════════════
    #  S1: CorrArb — بدون Regime Filter
    #  منطق: Z-score بالا/پایین + ADX متوسط (نه خیلی بالا)
    #  فقط شرط ADX < 35 (نه 25 که خیلی سخت بود)
    # ══════════════════════════════════════════════════════
    eurgbp = c_e / c_g
    z_mean = eurgbp.rolling(96).mean()
    z_std  = eurgbp.rolling(96).std()
    z      = (eurgbp-z_mean)/z_std.replace(0,np.nan)

    # Z از بازه بلندمدت‌تر
    z_mean_slow = eurgbp.rolling(480).mean()   # 5 روز
    z_std_slow  = eurgbp.rolling(480).std()
    z_slow      = (eurgbp-z_mean_slow)/z_std_slow.replace(0,np.nan)

    time_ok = hour.between(7,19)
    # ADX زیر ۳۵ (نه زیر ۲۵ که خیلی محدود بود)
    adx_ok  = adx < 35
    # تایید: هم Z کوتاه هم Z بلند در یک جهت
    std_ok  = z_std > z_std.rolling(480).mean()*0.2

    sig_arb = pd.Series(0,index=df.index)
    # Long EUR (Z خیلی منفی = EUR خیلی ارزان نسبت به GBP)
    sig_arb[(z < -2.0) & (z_slow < -0.5) &
            std_ok & adx_ok & time_ok & (rsi < 48)] =  1
    # Short EUR
    sig_arb[(z >  2.0) & (z_slow >  0.5) &
            std_ok & adx_ok & time_ok & (rsi > 52)] = -1
    sig_arb = sig_arb.where(sig_arb!=sig_arb.shift(),0)

    sl_arb = np.where(sig_arb!=0, 22.0, 0.0)
    tp_arb = np.where(sig_arb!=0, 40.0, 0.0)

    # ══════════════════════════════════════════════════════
    #  S2: Trend Follow — EMA Pullback
    #  مشکل قبلی: فقط 9 سیگنال در 5 سال! (crossover خیلی کم)
    #  راه‌حل: Pullback به EMA21 در جهت ترند
    #  این سیگنال‌های بیشتری می‌دهد
    # ══════════════════════════════════════════════════════
    active = hour.between(7,18) & weekday.between(0,3)

    dist21 = (c_e-ema21)/atr.replace(0,np.nan)  # فاصله از EMA21 به واحد ATR

    sig_tr = pd.Series(0,index=df.index)
    # Long: ترند صعودی + pullback به EMA21 + RSI در pullback zone
    sig_tr[active &
           (ema21>ema50) & (ema50>ema200) &  # ترند صعودی کامل
           (adx>20) &
           dist21.between(-1.5,0.1) &          # pullback به EMA21
           rsi.between(35,55) &
           (macd_h>macd_h.shift(2))] =  1      # MACD برگشته

    # Short: ترند نزولی + pullback
    sig_tr[active &
           (ema21<ema50) & (ema50<ema200) &
           (adx>20) &
           dist21.between(-0.1,1.5) &
           rsi.between(45,65) &
           (macd_h<macd_h.shift(2))] = -1

    sig_tr = sig_tr.where(sig_tr!=sig_tr.shift(),0)

    # SL زیر EMA21 + buffer
    sl_tr = np.where(sig_tr!=0,
                     np.maximum(18, (c_e-ema21).abs().values/Config.pip
                                + atr.values/Config.pip*0.8), 0.0)
    tp_tr = np.where(sig_tr!=0, sl_tr*2.5, 0.0)

    # ══════════════════════════════════════════════════════
    #  S3: London Open Range Breakout
    #  جایگزین AsianBreak که DD=-49% داشت
    #  منطق: رنج ۳۰ دقیقه قبل از بازگشایی لندن (۶:۳۰-۶:۵۹)
    #  شکست در ۷:۰۰-۸:۳۰ → ورود
    #  این استراتژی در فارکس بسیار اثبات‌شده است
    # ══════════════════════════════════════════════════════
    d_temp = df.copy()
    d_temp['date'] = d_temp.index.date

    # رنج پیش از لندن: ۶:۰۰-۶:۵۹
    pre_london = d_temp[hour==6]
    pre_rng = pre_london.groupby('date').agg(
        ph=('h_eur','max'),
        pl=('l_eur','min'),
    )
    pre_rng['prng'] = (pre_rng['ph']-pre_rng['pl'])/Config.pip
    d_temp = d_temp.join(pre_rng,on='date')

    # رنج معقول: ۵ تا ۳۰ پیپ (نه خیلی کوچک، نه خیلی بزرگ)
    rng_ok2  = d_temp['prng'].between(5,30)
    # ساعت شکست: ۷:۰۰-۸:۳۰
    break_t  = hour.between(7,8)
    day_ok2  = weekday.between(0,3)

    # شکست: قیمت بالای/پایین رنج با حداقل ۲ پیپ
    brk_up  = (c_e > d_temp['ph']+2*Config.pip) & \
               (c_e.shift(1) <= d_temp['ph']+2*Config.pip)
    brk_dn  = (c_e < d_temp['pl']-2*Config.pip) & \
               (c_e.shift(1) >= d_temp['pl']-2*Config.pip)

    sig_lb = pd.Series(0,index=df.index)
    sig_lb[break_t & rng_ok2 & day_ok2 & brk_up & (adx>15)] =  1
    sig_lb[break_t & rng_ok2 & day_ok2 & brk_dn & (adx>15)] = -1

    # اولین سیگنال هر روز
    nz_lb = sig_lb[sig_lb!=0]
    fi_lb = nz_lb.groupby(nz_lb.index.date).head(1).index
    s_lb  = pd.Series(0,index=df.index)
    s_lb[fi_lb] = sig_lb[fi_lb]

    prng_arr = d_temp['prng'].fillna(15).values
    # SL: پشت طرف مقابل رنج + ۲ پیپ
    sl_lb = np.where(s_lb!=0,
                     np.maximum(10, prng_arr+2), 0.0)
    tp_lb = np.where(s_lb!=0, sl_lb*2.0, 0.0)

    # ══════════════════════════════════════════════════════
    #  S4: RSI Mean Reversion (بدون Regime Filter)
    #  مشکل قبلی: 0 سیگنال چون Regime=Range فقط 16%
    #  راه‌حل: فیلتر ساده ADX + ساعت فعال
    # ══════════════════════════════════════════════════════
    bb_mid = c_e.rolling(20).mean()
    bb_std = c_e.rolling(20).std()
    bb_up2 = bb_mid + 2.0*bb_std
    bb_lo2 = bb_mid - 2.0*bb_std

    active4 = hour.between(9,17) & weekday.between(0,3)

    sig_mr = pd.Series(0,index=df.index)
    # خرید: RSI oversold + زیر BB پایین + ADX رنج
    sig_mr[active4 &
           (rsi<28) & (c_e<bb_lo2) &
           (adx<30) &
           (macd_h>macd_h.shift(1))] =  1
    # فروش: RSI overbought + بالای BB بالا + ADX رنج
    sig_mr[active4 &
           (rsi>72) & (c_e>bb_up2) &
           (adx<30) &
           (macd_h<macd_h.shift(1))] = -1

    nz_mr = sig_mr[sig_mr!=0]
    fi_mr = nz_mr.groupby(nz_mr.index.date).head(1).index
    s_mr  = pd.Series(0,index=df.index)
    s_mr[fi_mr] = sig_mr[fi_mr]

    sl_mr = np.where(s_mr!=0,
                     np.maximum(14,atr.values/Config.pip*1.8), 0.0)
    tp_mr = np.where(s_mr!=0, sl_mr*1.6, 0.0)

    print(" ✓")
    print(f"  CorrArb={int((sig_arb!=0).sum())} | "
          f"TrendPB={int((sig_tr!=0).sum())} | "
          f"LondonBreak={int((s_lb!=0).sum())} | "
          f"MeanRev={int((s_mr!=0).sum())}")

    return {
        'CorrArb':     (sig_arb, sl_arb, tp_arb, z),
        'TrendPB':     (sig_tr,  sl_tr,  tp_tr,  None),
        'LondonBreak': (s_lb,    sl_lb,  tp_lb,  None),
        'MeanRev':     (s_mr,    sl_mr,  tp_mr,  None),
        'atr': atr, 'ema21': ema21,
    }


# ================================================================== #
#   موتور بک‌تست ماهانه                                             #
# ================================================================== #
def run_monthly_backtest(
    df, strategy_name, sig_series,
    sl_arr, tp_arr, z_series=None,
):
    pip  = Config.pip
    ls   = Config.lot_size
    sp   = Config.spread_eur_pips
    comm = Config.commission_per_lot

    close_a = df['c_eur'].values
    high_a  = df['h_eur'].values
    low_a   = df['l_eur'].values
    sig_a   = sig_series.values
    ts_a    = df.index
    z_a     = z_series.values if z_series is not None else None

    periods  = pd.Series(ts_a).dt.to_period('M')
    months   = periods.unique()

    equity      = Config.initial_balance
    all_trades  = []
    monthly_log = []
    eq_curve    = [equity]
    eq_curve_ts = [None]
    open_pos    = None

    for month_period in months:
        mask      = (periods==month_period).values
        m_bars    = np.where(mask)[0]
        if len(m_bars)==0: continue
        bar_start = m_bars[0]; bar_end=m_bars[-1]
        if bar_end < Config.warmup: continue

        m_start  = equity
        m_peak   = equity
        m_day_eq = equity
        m_halted = False
        m_halt_r = ""
        m_trades = []
        cur_day  = None

        sig_set = {x for x in np.where(sig_a!=0)[0]
                   if bar_start<=x<=bar_end and x>=Config.warmup}

        for bar in range(max(bar_start,Config.warmup), bar_end+1):
            day = ts_a[bar].date()
            if day != cur_day:
                cur_day=day; m_day_eq=equity
                if m_halted and "Daily" in m_halt_r:
                    m_halted=False; m_halt_r=""

            if m_halted:
                if open_pos is not None:
                    cp=close_a[bar]
                    raw=open_pos['dir']*(cp-open_pos['entry'])*open_pos['lot']*ls
                    pnl=raw-(sp*pip*open_pos['lot']*ls+comm*open_pos['lot'])
                    equity+=pnl; m_peak=max(m_peak,equity)
                    eq_curve.append(round(equity,4))
                    eq_curve_ts.append(ts_a[bar])
                    rec={**open_pos,'exit':cp,'exit_ts':ts_a[bar],
                         'pnl':pnl,'status':'month_halt'}
                    m_trades.append(rec); all_trades.append(rec)
                    open_pos=None
                break

            # ── مدیریت پوزیشن ──
            if open_pos is not None:
                hi=high_a[bar]; lo=low_a[bar]; cp=close_a[bar]
                d=open_pos['dir']; ep=open_pos['entry']
                sl=open_pos['sl']; tp=open_pos['tp']

                hit_sl=(d==1 and lo<=sl) or (d==-1 and hi>=sl)
                hit_tp=(d==1 and hi>=tp) or (d==-1 and lo<=tp)

                # Z-exit برای CorrArb
                if z_a is not None:
                    zn=z_a[bar]
                    if not np.isnan(zn) and abs(zn)<0.3:
                        hit_tp=True

                # Trailing SL
                move=d*(cp-ep); td=abs(tp-ep)
                if td>0:
                    pct=move/td
                    if pct>0.5:
                        be=ep+d*td*0.1
                        open_pos['sl']=(max(sl,be) if d==1 else min(sl,be))
                    if pct>0.8:
                        lock=ep+d*td*0.5
                        open_pos['sl']=(max(open_pos['sl'],lock)
                                       if d==1 else min(open_pos['sl'],lock))

                # Time stop
                if (bar-open_pos['entry_bar'])>=384 and not hit_tp:
                    raw=d*(cp-ep)*open_pos['lot']*ls
                    pnl=raw-(sp*pip*open_pos['lot']*ls+comm*open_pos['lot'])
                    equity+=pnl; m_peak=max(m_peak,equity)
                    eq_curve.append(round(equity,4))
                    eq_curve_ts.append(ts_a[bar])
                    rec={**open_pos,'exit':cp,'exit_ts':ts_a[bar],
                         'pnl':pnl,'status':'TimeStop'}
                    m_trades.append(rec); all_trades.append(rec)
                    open_pos=None
                    dd_d=(equity-m_day_eq)/m_day_eq
                    dd_p=(equity-m_peak)/m_peak
                    if dd_d<=-Config.max_daily_loss_pct:
                        m_halted=True;m_halt_r=f"Daily {dd_d*100:.1f}%"
                    elif dd_p<=-Config.max_total_dd_pct:
                        m_halted=True;m_halt_r=f"MaxDD {dd_p*100:.1f}%"
                    continue

                er=exp=None
                if hit_sl: er,exp='SL',open_pos['sl']
                elif hit_tp: er,exp='TP',open_pos['tp']
                if er:
                    raw=d*(exp-ep)*open_pos['lot']*ls
                    pnl=raw-(sp*pip*open_pos['lot']*ls+comm*open_pos['lot'])
                    equity+=pnl; m_peak=max(m_peak,equity)
                    eq_curve.append(round(equity,4))
                    eq_curve_ts.append(ts_a[bar])
                    rec={**open_pos,'exit':exp,'exit_ts':ts_a[bar],
                         'pnl':pnl,'status':er}
                    m_trades.append(rec); all_trades.append(rec)
                    open_pos=None
                    dd_d=(equity-m_day_eq)/m_day_eq
                    dd_p=(equity-m_peak)/m_peak
                    if dd_d<=-Config.max_daily_loss_pct:
                        m_halted=True;m_halt_r=f"Daily {dd_d*100:.1f}%"
                    elif dd_p<=-Config.max_total_dd_pct:
                        m_halted=True;m_halt_r=f"MaxDD {dd_p*100:.1f}%"

            # ── ورود ──
            if open_pos is None and not m_halted and bar in sig_set:
                sv=int(sig_a[bar])
                slp=float(sl_arr[bar]); tpp=float(tp_arr[bar])
                if slp>0 and tpp>0 and not np.isnan(slp) and not np.isnan(tpp):
                    lot=lot_size_calc(equity,slp)
                    ep2=close_a[bar]+sv*sp*pip/2
                    open_pos=dict(
                        strategy=strategy_name,symbol='EUR',
                        dir=sv,lot=lot,entry=ep2,
                        sl=ep2-sv*slp*pip, tp=ep2+sv*tpp*pip,
                        entry_ts=ts_a[bar], entry_bar=bar,
                    )

        # ── ثبت ماه ──
        m_pnl = equity-m_start
        m_ret = m_pnl/m_start*100
        wins  = sum(1 for t in m_trades if t.get('pnl',0)>0)
        wr    = wins/len(m_trades)*100 if m_trades else 0
        dd_m  = min(0,(equity-m_peak)/m_peak*100)

        if m_halted:       st=f"🛑 HALTED"
        elif m_ret>=10:    st=f"🎯 TARGET"
        elif m_ret>0:      st=f"✅"
        elif len(m_trades)==0: st="⏸"
        else:              st=f"❌"

        monthly_log.append(dict(
            period=str(month_period),
            start_eq=round(m_start,2),
            end_eq=round(equity,2),
            pnl=round(m_pnl,2),
            ret_pct=round(m_ret,2),
            trades=len(m_trades),
            wins=wins, wr=round(wr,1),
            max_dd=round(dd_m,2),
            halted=m_halted,
            status=st,
        ))

    return all_trades, monthly_log, eq_curve, eq_curve_ts


# ================================================================== #
#   آمار                                                             #
# ================================================================== #
def compute_stats(trades, monthly_log, eq_curve, eq_curve_ts, name):
    if not trades: return None
    t=pd.DataFrame(trades)
    t['pnl']=(pd.to_numeric(t['pnl'],errors='coerce').fillna(0))
    t['entry_ts']=pd.to_datetime(t['entry_ts'])
    t['exit_ts']=pd.to_datetime(t['exit_ts'])
    t['duration_min']=(t['exit_ts']-t['entry_ts']).dt.total_seconds()/60
    ml=pd.DataFrame(monthly_log)

    final_eq=eq_curve[-1]
    total_pnl=final_eq-Config.initial_balance
    total_ret=total_pnl/Config.initial_balance*100
    sd=t['entry_ts'].min(); ed=t['exit_ts'].max()
    total_days=max((ed-sd).days,1)
    ann_ret=((final_eq/Config.initial_balance)**(365.25/total_days)-1)*100

    win_t=t[t['pnl']>0]; loss_t=t[t['pnl']<0]
    win_r=len(win_t)/len(t)*100 if len(t)>0 else 0
    avg_w=win_t['pnl'].mean() if len(win_t)>0 else 0
    avg_l=loss_t['pnl'].mean() if len(loss_t)>0 else 0
    gw=win_t['pnl'].sum(); gl=abs(loss_t['pnl'].sum())
    pf=gw/gl if gl>0 else float('inf')
    exp_v=t['pnl'].mean()
    rr=abs(avg_w/avg_l) if avg_l!=0 else 0

    eq_s=pd.Series(eq_curve)
    max_dd=((eq_s-eq_s.cummax())/eq_s.cummax()*100).min()
    r=eq_s.pct_change().dropna()
    sharpe=(r.mean()/r.std()*np.sqrt(252*96)) if r.std()>0 else 0
    neg=r[r<0]; ds=neg.std() if len(neg)>0 else 1e-10
    sortino=r.mean()/ds*np.sqrt(252*96)
    calmar=(final_eq/Config.initial_balance-1)/abs(max_dd/100) if max_dd!=0 else 0

    sign=t['pnl'].apply(lambda x:1 if x>0 else(-1 if x<0 else 0))
    cw=cl=mcw=mcl=0
    for s in sign:
        if s>0: cw+=1;cl=0;mcw=max(mcw,cw)
        elif s<0: cl+=1;cw=0;mcl=max(mcl,cl)
        else: cw=cl=0

    active_m  = ml[ml['trades']>0]
    prof_m    = (ml['pnl']>0).sum()
    loss_m    = (ml['pnl']<0).sum()
    no_trade  = (ml['trades']==0).sum()
    halt_m    = ml['halted'].sum()
    target_m  = (ml['ret_pct']>=10).sum()
    avg_mret  = active_m['ret_pct'].mean() if len(active_m)>0 else 0

    return dict(
        name=name,trades=t,monthly=ml,
        eq_curve=eq_curve,eq_curve_ts=eq_curve_ts,
        final_eq=final_eq,total_pnl=total_pnl,
        total_ret=total_ret,ann_ret=ann_ret,total_days=total_days,
        win_r=win_r,avg_w=avg_w,avg_l=avg_l,pf=pf,
        exp=exp_v,rr=rr,mcw=mcw,mcl=mcl,
        max_dd=max_dd,sharpe=sharpe,sortino=sortino,calmar=calmar,
        prof_m=int(prof_m),loss_m=int(loss_m),
        no_trade_m=int(no_trade),halt_m=int(halt_m),target_m=int(target_m),
        avg_mret=round(avg_mret,2),
        best_m=ml['pnl'].max(),worst_m=ml['pnl'].min(),
        best_t=t['pnl'].max(),worst_t=t['pnl'].min(),
        avg_dur=t['duration_min'].mean(),
    )


# ================================================================== #
#   گزارش                                                           #
# ================================================================== #
def print_report(s:dict)->str:
    W=74; SEP="═"*W
    def rw(lb,v,ok=None):
        l=f"  {lb}"; vs=str(v)
        mk="" if ok is None else(" ✅" if ok else" ❌")
        d="·"*max(2,W-len(l)-len(vs)-len(mk)-2)
        return f"{l} {d} {vs}{mk}"
    def box(t):
        i=f"─ {t} "; return"┌"+i+"─"*(W-len(i)-1)+"┐"
    bot="└"+"─"*(W-1)+"┘"

    ppm=s['avg_mret']
    ok=(s['total_ret']>0 and s['pf']>1.3
        and abs(s['max_dd'])<8 and ppm>8
        and s['prof_m']>s['loss_m'])
    flag="✅ PROP READY" if ok else "⚠️  در حال بهینه‌سازی"

    lines=[
        "",SEP,
        f"  ▌  {s['name']}   {flag}  ▐",
        f"  ▌  {s['trades']['entry_ts'].min().date()} → "
        f"{s['trades']['exit_ts'].max().date()}  ({s['total_days']} روز)  ▐",
        SEP,"",
        box("نتایج مالی"),
        rw("موجودی اولیه",  f"${Config.initial_balance:>12,.2f}"),
        rw("موجودی نهایی",  f"${s['final_eq']:>12,.2f}"),
        rw("سود کل",       f"${s['total_pnl']:>+12,.2f}"),
        rw("بازده کل",     f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه", f"{s['ann_ret']:>+.2f}%"),
        rw("ماهانه avg",   f"{ppm:>+.2f}%",ok=(ppm>8)),
        rw("بهترین ماه",   f"${s['best_m']:>+.2f}"),
        rw("بدترین ماه",   f"${s['worst_m']:>+.2f}"),
        bot,"",
        box("ریسک"),
        rw("Max Drawdown", f"{s['max_dd']:.2f}%",ok=(abs(s['max_dd'])<8)),
        rw("Sharpe",       f"{s['sharpe']:.2f}"),
        rw("Sortino",      f"{s['sortino']:.2f}"),
        rw("Calmar",       f"{s['calmar']:.2f}"),
        rw("Profit Factor",f"{s['pf']:.2f}",ok=(s['pf']>1.3)),
        bot,"",
        box("معاملات"),
        rw("تعداد کل",     f"{len(s['trades']):,}"),
        rw("Win Rate",      f"{s['win_r']:.1f}%"),
        rw("Avg Win",       f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",      f"${s['avg_l']:>+.2f}"),
        rw("RR",            f"{s['rr']:.2f}"),
        rw("Expectancy",    f"${s['exp']:>+.2f}"),
        rw("Max Cons Win",  f"{s['mcw']}"),
        rw("Max Cons Loss", f"{s['mcl']}"),
        rw("مدت میانگین",  f"{s['avg_dur']:.0f} min"),
        bot,"",
        box("آمار ماهانه"),
        rw("ماه‌های سودده",f"{s['prof_m']}",ok=(s['prof_m']>s['loss_m'])),
        rw("ماه‌های ضررده",f"{s['loss_m']}"),
        rw("بدون معامله",  f"{s['no_trade_m']}"),
        rw("Halted",       f"{s['halt_m']}",ok=(s['halt_m']<3)),
        rw("رسیده به ۱۰%",f"{s['target_m']}"),
        bot,"",
    ]

    # ── جدول ماه به ماه ──
    lines.append(box("جدول ماه به ماه"))
    lines.append(
        f"  {'ماه':>7}  {'موجودی':>10}  {'PnL':>9}  "
        f"{'Ret%':>6}  {'#T':>3}  {'WR%':>5}  "
        f"{'DD%':>6}  وضعیت")
    lines.append("  "+"─"*(W-3))
    for _,mr in s['monthly'].iterrows():
        lines.append(
            f"  {mr['period']:>7}  ${mr['start_eq']:>9,.0f}  "
            f"${mr['pnl']:>+8,.2f}  {mr['ret_pct']:>+5.1f}%  "
            f"{mr['trades']:>3}  {mr['wr']:>4.0f}%  "
            f"{mr['max_dd']:>5.1f}%  {mr['status']}")
    lines+=[bot,""]

    # ── سالانه ──
    s['trades']['yr']=s['trades']['entry_ts'].dt.year
    yr_g=(s['trades'].groupby('yr')
          .agg(n=('pnl','count'),pnl=('pnl','sum'),
               wins=('pnl',lambda x:(x>0).sum()))
          .reset_index())
    yr_g['wr']=yr_g['wins']/yr_g['n']*100
    yr_g['ret']=yr_g['pnl']/Config.initial_balance*100

    # موجودی شروع هر سال
    eq_yr={}; pv=None
    for _,mr in s['monthly'].iterrows():
        yr=int(mr['period'][:4])
        if yr!=pv: eq_yr[yr]=mr['start_eq']; pv=yr

    lines.append(box("گزارش سالانه"))
    lines.append(f"  {'سال':>5}  {'#':>5}  {'WR%':>5}  "
                 f"{'PnL':>10}  {'بازده واقعی%':>13}")
    lines.append("  "+"─"*(W-3))
    for _,yr in yr_g.iterrows():
        y=int(yr['yr'])
        base=eq_yr.get(y,Config.initial_balance)
        real=yr['pnl']/base*100
        lines.append(
            f"  {y:>5}  {int(yr['n']):>5}  {yr['wr']:>4.1f}%  "
            f"${yr['pnl']:>9.2f}  {real:>+12.1f}%")
    lines.append(bot)
    out="\n".join(lines); print(out); return out


def print_comparison(results)->str:
    W=74; SEP="═"*W
    lines=["",SEP,
           "  ▌  STRATEGY COMPARISON  ▐",SEP,
           f"  {'نام':<13} {'Ann%':>7} {'AvgM%':>6} {'DD%':>7} "
           f"{'PF':>5} {'WR%':>5} {'M+':>4} {'M-':>4} "
           f"{'Halt':>5} {'10%':>4}  نتیجه",
           "  "+"─"*(W-3)]
    for s in results:
        ppm=s['avg_mret']
        ok=(s['total_ret']>0 and s['pf']>1.3
            and abs(s['max_dd'])<8 and ppm>8
            and s['prof_m']>s['loss_m'])
        flag="✅ PASS" if ok else"❌ FAIL"
        pf_s=f"{s['pf']:.2f}" if s['pf']!=float('inf') else"  ∞"
        lines.append(
            f"  {s['name']:<13} {s['ann_ret']:>+6.1f}% {ppm:>+5.1f}% "
            f"{s['max_dd']:>6.1f}% {pf_s:>5} {s['win_r']:>4.1f}% "
            f"{s['prof_m']:>4} {s['loss_m']:>4} "
            f"{s['halt_m']:>5} {s['target_m']:>4}  {flag}")

    lines+=["  "+"─"*(W-3),""]
    good=[s for s in results
          if s['total_ret']>0 and s['pf']>1.3
          and abs(s['max_dd'])<8 and s['avg_mret']>8
          and s['prof_m']>s['loss_m']]
    if good:
        lines.append("  🏆 PROP READY:")
        for s in sorted(good,key=lambda x:x['ann_ret'],reverse=True):
            lines.append(
                f"     ✅ {s['name']:<13}  "
                f"سالانه={s['ann_ret']:>+.1f}%  "
                f"ماهانه={s['avg_mret']:>+.1f}%  "
                f"DD={s['max_dd']:.1f}%  "
                f"M+={s['prof_m']}")
    else:
        sr=sorted(results,key=lambda x:x['ann_ret'],reverse=True)
        lines.append("  📊 رتبه‌بندی:")
        for i,s in enumerate(sr,1):
            lines.append(
                f"  {i}. {s['name']:<13}  "
                f"Ann={s['ann_ret']:>+.1f}%  "
                f"AvgM={s['avg_mret']:>+.1f}%  "
                f"DD={s['max_dd']:.1f}%  "
                f"PF={s['pf']:.2f}")
    lines+=["",SEP]
    out="\n".join(lines); print(out); return out


def save_outputs(results):
    rows=[["BACKTEST REPORT"],
          [f"Generated:{datetime.now().strftime('%Y-%m-%d %H:%M')}"],
          [f"Risk={Config.risk_per_trade_pct*100}%  "
           f"MaxDD={Config.max_total_dd_pct*100}%"],[""],
          ["Strategy","FinalEq","TotalRet%","AnnRet%","AvgMonth%",
           "MaxDD%","PF","WinRate%","RR","Trades",
           "ProfM","LossM","HaltM","TargetM","Status"]]
    for s in results:
        ok=(s['total_ret']>0 and s['pf']>1.3
            and abs(s['max_dd'])<8 and s['avg_mret']>8)
        pf_v=round(s['pf'],2) if s['pf']!=float('inf') else 999
        rows.append([s['name'],round(s['final_eq'],2),
                     round(s['total_ret'],2),round(s['ann_ret'],2),
                     round(s['avg_mret'],2),round(s['max_dd'],2),pf_v,
                     round(s['win_r'],1),round(s['rr'],2),
                     len(s['trades']),s['prof_m'],s['loss_m'],
                     s['halt_m'],s['target_m'],
                     "PASS" if ok else"FAIL"])
    for s in results:
        rows+=[[""],
               [f"=== MONTHLY: {s['name']} ==="],
               ["Month","StartEq","EndEq","PnL","Ret%",
                "Trades","WinRate%","MaxDD%","Status"]]
        for _,mr in s['monthly'].iterrows():
            rows.append([mr['period'],round(mr['start_eq'],2),
                         round(mr['end_eq'],2),round(mr['pnl'],2),
                         round(mr['ret_pct'],2),mr['trades'],
                         round(mr['wr'],1),round(mr['max_dd'],2),
                         mr['status']])
    pd.DataFrame(rows).to_csv("Report.csv",index=False,
                               header=False,encoding="utf-8-sig")
    for s in results:
        eq_df=pd.DataFrame({'ts':s['eq_curve_ts'],'equity':s['eq_curve']})
        eq_df['dd']=((eq_df['equity']-eq_df['equity'].cummax())
                     /eq_df['equity'].cummax()*100).round(4)
        eq_df.to_csv(f"eq_{s['name']}.csv",index=False,encoding="utf-8-sig")
    print(f"\n✅ فایل‌ها: Report.csv + " +
          " + ".join(f"eq_{s['name']}.csv" for s in results))


# ================================================================== #
if __name__=="__main__":
    df=load_data()
    print("\n"+"═"*74)
    print("  MONTHLY SIMULATION — بدون Regime Filter")
    print("═"*74)

    sigs=compute_signals(df)
    strats=[
        ('CorrArb',     *sigs['CorrArb']),
        ('TrendPB',     *sigs['TrendPB'],),
        ('LondonBreak', *sigs['LondonBreak']),
        ('MeanRev',     *sigs['MeanRev']),
    ]

    all_res=[]; all_txt=[]
    for row in strats:
        name=row[0]; sig=row[1]; sl=row[2]; tp=row[3]; z=row[4]
        t0=datetime.now()
        print(f"\n  ▶ {name}...",end="",flush=True)
        trades,ml,eqc,eqts=run_monthly_backtest(df,name,sig,sl,tp,z)
        dt=(datetime.now()-t0).total_seconds()
        print(f" {dt:.1f}s | {len(trades)} معامله")
        if not trades: print("  ❌ بدون معامله"); continue
        st=compute_stats(trades,ml,eqc,eqts,name)
        if st: all_res.append(st); all_txt.append(print_report(st))

    if all_res:
        all_txt.append(print_comparison(all_res))
        with open("Report.txt","w",encoding="utf-8") as f:
            f.write("\n".join(all_txt))
        save_outputs(all_res)
