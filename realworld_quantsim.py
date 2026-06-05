import pandas as pd, numpy as np, glob, warnings
warnings.filterwarnings('ignore')

class Config:
    init_bal = 5000.0; risk_pct = 0.015; max_dd_day = 0.045; max_dd_tot = 0.09
    sp_eur = 1.0; sp_gbp = 1.2; comm = 6.0; pip = 0.0001; lot_sz = 100000; max_lot = 3.0

def load_data(tf):
    def read(pths, sfx):
        dfs = []
        for p in pths:
            d = pd.read_csv(p, sep=';', header=None, names=['ts','o','h','l','c','v'])
            d['ts'] = pd.to_datetime(d['ts'], format='%Y%m%d %H%M%S')
            d = d.set_index('ts'); d = d[~d.index.duplicated(keep='last')]
            d.columns = [f'{c}_{sfx}' for c in d.columns]; dfs.append(d)
        return pd.concat(dfs).sort_index()
    raw = read(sorted(glob.glob('data/*EURUSD*.csv')), 'eur').join(read(sorted(glob.glob('data/*GBPUSD*.csv')), 'gbp'), how='inner').dropna()
    df = pd.DataFrame({
        'o_eur': raw['o_eur'].resample(tf).first(), 'h_eur': raw['h_eur'].resample(tf).max(),
        'l_eur': raw['l_eur'].resample(tf).min(),   'c_eur': raw['c_eur'].resample(tf).last(),
        'c_gbp': raw['c_gbp'].resample(tf).last()
    }).dropna()
    return df[df.index.weekday < 5]

def calc_atr(h, l, c, p=14): return pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).rolling(p).mean()
def calc_rsi(c, p=14):
    d = c.diff(); g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100 / (1 + g / (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean().replace(0, np.nan))
def make_sigs(idx): return pd.DataFrame({'signal': 0, 'sl_pips': 0.0, 'tp_pips': 0.0}, index=idx, dtype=float)
def throt(sigs, gap=3):
    nz = sigs[sigs['signal']!=0]
    if len(nz)<=1: return sigs
    kp = [nz.index[0]]
    for i in nz.index[1:]:
        if (i - kp[-1]).total_seconds() >= gap*3600: kp.append(i)
    sigs.loc[[i for i in nz.index if i not in kp], 'signal'] = 0
    return sigs
def sz(eq, slp): return round(np.clip((eq*Config.risk_pct)/(max(slp,0.1)*Config.pip*Config.lot_sz), 0.01, Config.max_lot), 2)
def pnl(d, lot, en, ex, sym): return (d*(ex-en)*lot*Config.lot_sz) - ((Config.sp_eur if sym=='EUR' else Config.sp_gbp)*Config.pip*lot*Config.lot_sz + Config.comm*lot)

def strat_rsi_div(df):
    c, h, l, o = df['c_eur'], df['h_eur'], df['l_eur'], df['o_eur']
    rsi, atr, act = calc_rsi(c, 14), calc_atr(h, l, c, 14), pd.Series(df.index.hour, index=df.index).between(7, 18)
    
    # 100% Realistic Causal Swings (No Lookahead Bias)
    sw_lo, sw_hi = l.rolling(10).min(), h.rolling(10).max()
    is_lo = (l.shift(2) == sw_lo.shift(2)) & (l.shift(1) > l.shift(2)) & (l > l.shift(1))
    is_hi = (h.shift(2) == sw_hi.shift(2)) & (h.shift(1) < h.shift(2)) & (h < h.shift(1))
    
    lp, lr, hp, hr_ = l.shift(2).where(is_lo).ffill(), rsi.shift(2).where(is_lo).ffill(), h.shift(2).where(is_hi).ffill(), rsi.shift(2).where(is_hi).ffill()
    bull = (lp < lp.shift(10)) & (lr > lr.shift(10)+3) & (rsi < 40) & (rsi > rsi.shift(1)) & (c > o) & act
    bear = (hp > hp.shift(10)) & (hr_ < hr_.shift(10)-3) & (rsi > 60) & (rsi < rsi.shift(1)) & (c < o) & act
    
    sigs = make_sigs(df.index)
    for i in range(250, len(df)):
        if pd.notna(atr.iloc[i]) and atr.iloc[i]>0 and (bull.iloc[i] or bear.iloc[i]):
            sl = max(12, min(atr.iloc[i]/Config.pip*1.2, 25))
            sigs.at[df.index[i], 'signal'] = 1 if bull.iloc[i] else -1
            sigs.at[df.index[i], 'sl_pips'], sigs.at[df.index[i], 'tp_pips'] = sl, sl*2.0
    return throt(sigs, 4)

def strat_corr_arb_v3(df):
    # Restored to proven V2 Settings
    ratio = df['c_eur']/df['c_gbp']
    mu, sd = ratio.rolling(96).mean(), ratio.rolling(96).std()
    z = (ratio - mu) / sd.replace(0, np.nan)
    std_ok = sd > sd.rolling(384).mean() * 0.3
    sigs, act = make_sigs(df.index), pd.Series(df.index.hour, index=df.index).between(7, 19)
    bull, bear = (z < -2.0) & std_ok & act, (z > 2.0) & std_ok & act
    
    for i in range(250, len(df)):
        if bull.iloc[i] or bear.iloc[i]:
            sigs.at[df.index[i], 'signal'] = 1 if bull.iloc[i] else -1
            sigs.at[df.index[i], 'sl_pips'], sigs.at[df.index[i], 'tp_pips'] = 20.0, 35.0 # V2 Fixed Targets
    return throt(sigs, 4)

class Risk:
    def __init__(self): self.eq = self.pk = self.ds = Config.init_bal; self.cd = None; self.hlt = False
    def chk(self, ts, amt=0):
        # FIX: Reset Halt logic at midnight
        if ts.date() != self.cd:
            self.cd = ts.date(); self.ds = self.eq
            if (self.pk - self.eq)/self.pk < Config.max_dd_tot: self.hlt = False
            
        self.eq += amt; self.pk = max(self.pk, self.eq)
        dd_day, dd_tot = (self.ds - self.eq)/self.ds, (self.pk - self.eq)/self.pk
        if dd_day >= Config.max_dd_day or dd_tot >= Config.max_dd_tot: self.hlt = True
        return not self.hlt and dd_day < Config.max_dd_day*0.7

def run_combo(df, strats):
    rsk = Risk(); trds = []; pos = {}; atr_e = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    for i in range(300, len(df)):
        ts = df.index[i]; rsk.chk(ts)
        if rsk.hlt:
            for k, p in list(pos.items()):
                c = df[f"c_{p['sym'].lower()}"].iloc[i]; pn = pnl(p['d'], p['l'], p['en'], c, p['sym'])
                rsk.chk(ts, pn); trds.append({**p, 'ex': c, 'pnl': pn}); del pos[k]
            continue
            
        for k, p in list(pos.items()):
            s = p['sym'].lower(); h, l, c, d, av = df[f'h_{s}'].iloc[i], df[f'l_{s}'].iloc[i], df[f'c_{s}'].iloc[i], p['d'], atr_e.iloc[i]
            
            # Trailing stop only for RSI (Arbitrage uses fixed SL/TP)
            if p['st'] == 'RSI_Div' and pd.notna(av) and av>0 and (d*(c-p['en'])) > av*1.2:
                p['sl'] = max(p['sl'], p['en']+d*av*0.3) if d==1 else min(p['sl'], p['en']+d*av*0.3)
                
            hit_s, hit_t = (d==1 and l<=p['sl']) or (d==-1 and h>=p['sl']), (d==1 and h>=p['tp']) or (d==-1 and l<=p['tp'])
            elo = (ts - p['ts']).total_seconds()/3600 >= (72 if p['st']=='CorrArb_v3' else 48)
            
            if hit_s or hit_t or elo or (ts.weekday()==4 and ts.hour>=20):
                xp = p['sl'] if hit_s else (p['tp'] if hit_t else c); pn = pnl(d, p['l'], p['en'], xp, p['sym'])
                rsk.chk(ts, pn); trds.append({**p, 'ex': xp, 'pnl': pn}); del pos[k]
                
        if rsk.chk(ts) and len(pos)<2:
            for sn, sg in strats.items():
                if sn in pos or sg.iloc[i]['signal']==0: continue
                v, slp, tpp, c = int(sg.iloc[i]['signal']), sg.iloc[i]['sl_pips'], sg.iloc[i]['tp_pips'], df['c_eur'].iloc[i]
                en = c + v*Config.sp_eur*Config.pip/2
                pos[sn] = {'st': sn, 'sym': 'EUR', 'd': v, 'l': sz(rsk.eq, slp), 'en': en, 'ts': ts, 'sl': en-v*slp*Config.pip, 'tp': en+v*tpp*Config.pip}
    return trds, rsk.eq

if __name__ == "__main__":
    print("="*60 + "\n 🚀 MULTI-TIMEFRAME PROP BACKTEST (5 YEARS)\n" + "="*60)
    for tf in ['5min', '15min', '1h']:
        print(f"\n⏳ Processing {tf} timeframe (Realistic Execution)...")
        try: df = load_data(tf)
        except Exception as e: print(f" Error loading data: {e}"); continue
        if df.empty: continue
        
        sigs = {'RSI_Div': strat_rsi_div(df), 'CorrArb_v3': strat_corr_arb_v3(df)}
        trds, eq = run_combo(df, sigs)
        t = pd.DataFrame(trds)
        
        if not t.empty:
            t['pnl'] = pd.to_numeric(t['pnl'])
            t['peak'] = (t['pnl'].cumsum() + Config.init_bal).cummax()
            max_dd = abs(((t['pnl'].cumsum() + Config.init_bal - t['peak']) / t['peak'] * 100).min())
            wr = len(t[t['pnl']>0]) / len(t) * 100
            pf = t[t['pnl']>0]['pnl'].sum() / abs(t[t['pnl']<0]['pnl'].sum()) if len(t[t['pnl']<0])>0 else float('inf')
            ret = ((eq - Config.init_bal) / Config.init_bal) * 100
            mo_ret = ret / max(((t['ts'].max() - t['ts'].min()).days / 30.44), 1)
            
            print(f" ► TIMEFRAME: {tf.upper()}")
            print(f" 💼 Trades: {len(t):<5} | 🎯 WR: {wr:.1f}% | ⚖️ PF: {pf:.2f}")
            print(f" 💰 Profit: ${eq-Config.init_bal:,.2f} ({ret:+.1f}%) | 📅 Mo: {mo_ret:+.2f}% | 📉 Max DD: {max_dd:.2f}%")
            if max_dd < 9.0 and mo_ret > 3.0: print(" ✅ VERDICT: PROP READY")
            else: print(" ⚠️ VERDICT: NEEDS TWEAKING")
        else:
            print(f" ► TIMEFRAME: {tf.upper()} - ❌ No trades generated.")
    print("\n" + "="*60 + "\n ✅ DONE\n" + "="*60)
