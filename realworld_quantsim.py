import pandas as pd
import numpy as np

# ============================================================
# 1) CORR ARB ORIGINAL SIGNALS
# ============================================================
def strategy_corr_arb_original(df):
    eurgbp = df['c_eur'] / df['c_gbp']
    period = 96  # 1 trading day

    mean = eurgbp.rolling(period).mean()
    std  = eurgbp.rolling(period).std()
    z    = (eurgbp - mean) / std.replace(0, np.nan)

    std_ok  = std > std.rolling(period * 4).mean() * 0.3
    adx_eur = calc_adx_full(df['h_eur'], df['l_eur'], df['c_eur'], 14)[0]
    adx_ok  = adx_eur < 28

    hour    = pd.Series(df.index.hour, index=df.index)
    time_ok = hour.between(7, 19)

    sig = pd.DataFrame(index=df.index)
    sig['signal']  = 0
    sig['sl_pips'] = 0.0
    sig['tp_pips'] = 0.0

    long_cond  = (z < -2.0) & std_ok & adx_ok & time_ok
    short_cond = (z >  2.0) & std_ok & adx_ok & time_ok

    sig.loc[long_cond,  'signal'] =  1
    sig.loc[short_cond, 'signal'] = -1
    sig.loc[sig['signal'] != 0, 'sl_pips'] = 20.0
    sig.loc[sig['signal'] != 0, 'tp_pips'] = 35.0

    # فقط تغییر سیگنال
    sig['signal'] = sig['signal'].where(sig['signal'] != sig['signal'].shift(), 0)

    return sig, z


# ============================================================
# 2) RSI DIV - همان نسخه سودده فعلی
# ============================================================
def strategy_rsi_div_proven(df):
    c = df['c_eur']; h = df['h_eur']
    l = df['l_eur']; o = df['o_eur']

    rsi = calc_rsi(c, 14)
    atr = calc_atr(h, l, c, 14)
    hour = pd.Series(df.index.hour, index=df.index)
    active = hour.between(7, 18)

    lb = 10
    sw_lo = l.rolling(lb * 2 + 1, center=True).min()
    sw_hi = h.rolling(lb * 2 + 1, center=True).max()

    is_lo = (l == sw_lo) & (l < l.shift(1)) & (l < l.shift(-1))
    is_hi = (h == sw_hi) & (h > h.shift(1)) & (h > h.shift(-1))

    lp = l.where(is_lo).ffill()
    lr = rsi.where(is_lo).ffill()
    pp = lp.shift(lb)
    pr = lr.shift(lb)

    hp = h.where(is_hi).ffill()
    hr = rsi.where(is_hi).ffill()
    php = hp.shift(lb)
    phr = hr.shift(lb)

    bull = (
        (lp < pp) &
        (lr > pr + 3) &
        (rsi < 40) &
        (rsi > rsi.shift(1)) &
        (c > o) &
        active
    )

    bear = (
        (hp > php) &
        (hr < phr - 3) &
        (rsi > 60) &
        (rsi < rsi.shift(1)) &
        (c < o) &
        active
    )

    sig = pd.DataFrame(index=df.index)
    sig['signal']  = 0
    sig['sl_pips'] = 0.0
    sig['tp_pips'] = 0.0

    for i in range(250, len(df)):
        atr_v = atr.iloc[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue

        atr_pips = atr_v / Config.pip
        idx = df.index[i]

        if bull.iloc[i]:
            sl = max(12, min(atr_pips * 1.2, 25))
            sig.at[idx, 'signal']  =  1
            sig.at[idx, 'sl_pips'] = sl
            sig.at[idx, 'tp_pips'] = sl * 2.0

        elif bear.iloc[i]:
            sl = max(12, min(atr_pips * 1.2, 25))
            sig.at[idx, 'signal']  = -1
            sig.at[idx, 'sl_pips'] = sl
            sig.at[idx, 'tp_pips'] = sl * 2.0

    # فاصله حداقل 4 ساعت بین سیگنال‌ها
    nz = sig[sig['signal'] != 0]
    if len(nz) > 1:
        keep = [nz.index[0]]
        for idx in nz.index[1:]:
            if (idx - keep[-1]).total_seconds() >= 4 * 3600:
                keep.append(idx)
        drop = [x for x in nz.index if x not in keep]
        sig.loc[drop, 'signal'] = 0

    return sig


# ============================================================
# 3) MONTHLY RETURNS - دقیق
# ============================================================
def build_precise_monthly_returns(risk, start_ts, end_ts):
    eq = pd.DataFrame({
        'ts': risk.curve_ts,
        'equity': risk.curve
    }).dropna()

    if eq.empty:
        return pd.DataFrame(columns=['equity_end', 'pnl', 'ret_pct'])

    eq = eq.sort_values('ts').set_index('ts')

    # daily equity with forward fill
    daily_idx = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq='D')
    daily_eq = eq['equity'].resample('D').last().reindex(daily_idx).ffill()
    daily_eq = daily_eq.fillna(Config.initial_balance)

    monthly_eq = daily_eq.resample('M').last()

    monthly = pd.DataFrame(index=monthly_eq.index)
    monthly['equity_end'] = monthly_eq
    monthly['equity_start'] = monthly['equity_end'].shift(1)
    monthly.iloc[0, monthly.columns.get_loc('equity_start')] = Config.initial_balance

    monthly['pnl'] = monthly['equity_end'] - monthly['equity_start']
    monthly['ret_pct'] = monthly['pnl'] / monthly['equity_start'] * 100

    monthly.index = monthly.index.to_period('M').astype(str)
    return monthly[['equity_end', 'pnl', 'ret_pct']]


# ============================================================
# 4) BACKTEST SINGLE - تمیز و درست
#    نکته مهم:
#    - CorrArb trailing ندارد
#    - CorrArb وقتی |z|<0.3 شد با قیمت فعلی close می‌بندد
#      نه اینکه fake روی TP ببندیم
# ============================================================
def run_single_clean(df, name, signals, symbol='EUR',
                     zscore=None, time_stop_h=48, trailing=False):
    risk   = RiskManager(name)
    trades = []
    pos    = None
    warmup = 300

    s_ = symbol.lower()
    hc, lc, cc = f'h_{s_}', f'l_{s_}', f'c_{s_}'
    atr = calc_atr(df[hc], df[lc], df[cc], 14)

    for i in range(warmup, len(df)):
        ts = df.index[i]
        hi = df[hc].iloc[i]
        lo = df[lc].iloc[i]
        cp = df[cc].iloc[i]

        risk.new_bar(ts)

        if risk.halted:
            if pos is not None:
                pnl = calc_pnl(pos['dir'], pos['lot'], pos['entry'], cp, symbol)
                trades.append({**pos, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'halt_close'})
                risk.add_pnl(pnl, ts)
                pos = None
            continue

        # =========================
        # EXIT
        # =========================
        if pos is not None:
            d  = pos['dir']
            ep = pos['entry']

            # trailing فقط برای RSI_Div
            if trailing:
                atr_v = atr.iloc[i]
                if pd.notna(atr_v) and atr_v > 0:
                    move = d * (cp - ep)
                    if move > atr_v * 1.2:
                        be = ep + d * atr_v * 0.3
                        if d == 1:
                            pos['sl'] = max(pos['sl'], be)
                        else:
                            pos['sl'] = min(pos['sl'], be)

                    if move > atr_v * 2.0:
                        lock = ep + d * atr_v * 1.0
                        if d == 1:
                            pos['sl'] = max(pos['sl'], lock)
                        else:
                            pos['sl'] = min(pos['sl'], lock)

            sl = pos['sl']
            tp = pos['tp']

            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            exit_reason = None
            exit_price  = None

            # اولویت درست: SL/TP اول
            if hit_sl and hit_tp:
                exit_reason = 'SL'      # محافظه‌کارانه
                exit_price  = sl
            elif hit_sl:
                exit_reason = 'SL'
                exit_price  = sl
            elif hit_tp:
                exit_reason = 'TP'
                exit_price  = tp

            # CorrArb special exit
            if exit_reason is None and name == 'CorrArb' and zscore is not None:
                z_now = zscore.iloc[i]
                if pd.notna(z_now) and abs(z_now) < 0.3:
                    exit_reason = 'ZExit'
                    exit_price  = cp   # مهم: قیمت واقعی فعلی، نه TP فیک

            # Time stop
            if exit_reason is None:
                elapsed_h = (ts - pos['entry_ts']).total_seconds() / 3600
                if elapsed_h >= time_stop_h:
                    exit_reason = 'TimeStop'
                    exit_price  = cp

            # Weekend close
            if exit_reason is None and ts.weekday() == 4 and ts.hour >= 20:
                exit_reason = 'WeekEnd'
                exit_price  = cp

            if exit_reason is not None:
                pnl = calc_pnl(d, pos['lot'], ep, exit_price, symbol)
                trades.append({**pos, 'exit': exit_price, 'exit_ts': ts,
                               'pnl': pnl, 'status': exit_reason})
                risk.add_pnl(pnl, ts)
                pos = None

        # =========================
        # ENTRY
        # =========================
        if pos is None and risk.can_trade():
            sv = int(signals['signal'].iloc[i])
            if sv != 0:
                sl_pips = float(signals['sl_pips'].iloc[i])
                tp_pips = float(signals['tp_pips'].iloc[i])

                if sl_pips <= 0 or tp_pips <= 0:
                    continue
                if tp_pips / sl_pips < Config.min_rr:
                    continue

                lot = lot_size_calc(risk.equity, sl_pips)
                spread = Config.spread_eur_pips if symbol == 'EUR' else Config.spread_gbp_pips
                entry = cp + sv * spread * Config.pip / 2

                pos = dict(
                    strategy=name,
                    symbol=symbol,
                    dir=sv,
                    lot=lot,
                    entry=entry,
                    sl=entry - sv * sl_pips * Config.pip,
                    tp=entry + sv * tp_pips * Config.pip,
                    entry_ts=ts,
                )

    # close remaining
    if pos is not None:
        last_p = df[cc].iloc[-1]
        last_ts = df.index[-1]
        pnl = calc_pnl(pos['dir'], pos['lot'], pos['entry'], last_p, symbol)
        trades.append({**pos, 'exit': last_p, 'exit_ts': last_ts,
                       'pnl': pnl, 'status': 'eod_close'})
        risk.add_pnl(pnl, last_ts)

    return trades, risk


# ============================================================
# 5) BACKTEST COMBINED - همان منطق single
#    مهم: اگر فقط یک استراتژی باشد، باید با single برابر شود
# ============================================================
def run_combined_clean(df, strategy_map):
    """
    strategy_map:
    {
      'CorrArb': {
          'signals': sig_corr,
          'symbol': 'EUR',
          'zscore': z_corr,
          'time_stop_h': 96,
          'trailing': False,
      },
      'RSI_Div': {
          'signals': sig_rsi,
          'symbol': 'EUR',
          'zscore': None,
          'time_stop_h': 48,
          'trailing': True,
      }
    }
    """
    risk     = RiskManager("Combined")
    trades   = []
    open_pos = {}
    warmup   = 300
    max_open = len(strategy_map)

    atr_eur = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    atr_gbp = calc_atr(df['h_gbp'], df['l_gbp'], df['c_gbp'], 14)

    for i in range(warmup, len(df)):
        ts = df.index[i]
        risk.new_bar(ts)

        if risk.halted:
            for key in list(open_pos.keys()):
                p = open_pos.pop(key)
                s_ = p['symbol'].lower()
                cp = df[f'c_{s_}'].iloc[i]
                pnl = calc_pnl(p['dir'], p['lot'], p['entry'], cp, p['symbol'])
                trades.append({**p, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'halt_close'})
                risk.add_pnl(pnl, ts)
            continue

        # =========================
        # EXIT
        # =========================
        for key in list(open_pos.keys()):
            p    = open_pos[key]
            meta = strategy_map[p['strategy']]

            s_ = p['symbol'].lower()
            hi = df[f'h_{s_}'].iloc[i]
            lo = df[f'l_{s_}'].iloc[i]
            cp = df[f'c_{s_}'].iloc[i]
            d  = p['dir']
            ep = p['entry']

            atr_series = atr_eur if p['symbol'] == 'EUR' else atr_gbp

            if meta['trailing']:
                atr_v = atr_series.iloc[i]
                if pd.notna(atr_v) and atr_v > 0:
                    move = d * (cp - ep)
                    if move > atr_v * 1.2:
                        be = ep + d * atr_v * 0.3
                        if d == 1:
                            p['sl'] = max(p['sl'], be)
                        else:
                            p['sl'] = min(p['sl'], be)

                    if move > atr_v * 2.0:
                        lock = ep + d * atr_v * 1.0
                        if d == 1:
                            p['sl'] = max(p['sl'], lock)
                        else:
                            p['sl'] = min(p['sl'], lock)

            sl = p['sl']
            tp = p['tp']

            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            exit_reason = None
            exit_price  = None

            if hit_sl and hit_tp:
                exit_reason = 'SL'
                exit_price  = sl
            elif hit_sl:
                exit_reason = 'SL'
                exit_price  = sl
            elif hit_tp:
                exit_reason = 'TP'
                exit_price  = tp

            if exit_reason is None and p['strategy'] == 'CorrArb' and meta['zscore'] is not None:
                z_now = meta['zscore'].iloc[i]
                if pd.notna(z_now) and abs(z_now) < 0.3:
                    exit_reason = 'ZExit'
                    exit_price  = cp

            if exit_reason is None:
                elapsed_h = (ts - p['entry_ts']).total_seconds() / 3600
                if elapsed_h >= meta['time_stop_h']:
                    exit_reason = 'TimeStop'
                    exit_price  = cp

            if exit_reason is None and ts.weekday() == 4 and ts.hour >= 20:
                exit_reason = 'WeekEnd'
                exit_price  = cp

            if exit_reason is not None:
                pnl = calc_pnl(d, p['lot'], ep, exit_price, p['symbol'])
                trades.append({**p, 'exit': exit_price, 'exit_ts': ts,
                               'pnl': pnl, 'status': exit_reason})
                risk.add_pnl(pnl, ts)
                del open_pos[key]

        # =========================
        # ENTRY
        # =========================
        if not risk.can_trade():
            continue

        if len(open_pos) >= max_open:
            continue

        for strat_name, meta in strategy_map.items():
            if strat_name in open_pos:
                continue

            sigs = meta['signals']
            sym  = meta['symbol']
            sv   = int(sigs['signal'].iloc[i])

            if sv == 0:
                continue

            sl_pips = float(sigs['sl_pips'].iloc[i])
            tp_pips = float(sigs['tp_pips'].iloc[i])

            if sl_pips <= 0 or tp_pips <= 0:
                continue
            if tp_pips / sl_pips < Config.min_rr:
                continue

            cp = df[f'c_{sym.lower()}'].iloc[i]
            lot = lot_size_calc(risk.equity, sl_pips)
            spread = Config.spread_eur_pips if sym == 'EUR' else Config.spread_gbp_pips
            entry = cp + sv * spread * Config.pip / 2

            open_pos[strat_name] = dict(
                strategy=strat_name,
                symbol=sym,
                dir=sv,
                lot=lot,
                entry=entry,
                sl=entry - sv * sl_pips * Config.pip,
                tp=entry + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

    # close remaining
    for key, p in open_pos.items():
        cp = df[f"c_{p['symbol'].lower()}"].iloc[-1]
        last_ts = df.index[-1]
        pnl = calc_pnl(p['dir'], p['lot'], p['entry'], cp, p['symbol'])
        trades.append({**p, 'exit': cp, 'exit_ts': last_ts,
                       'pnl': pnl, 'status': 'eod_close'})
        risk.add_pnl(pnl, last_ts)

    return trades, risk


# ============================================================
# 6) REPORT FOCUSED - دقیق روی ماهانه و DD
# ============================================================
def report_focus(name, trades, risk, start_ts, end_ts):
    monthly = build_precise_monthly_returns(risk, start_ts, end_ts)

    if len(monthly) == 0:
        print(f"\n❌ {name}: no monthly data")
        return None

    avg_m  = monthly['ret_pct'].mean()
    med_m  = monthly['ret_pct'].median()
    best_m = monthly['ret_pct'].max()
    worst_m= monthly['ret_pct'].min()
    pos_m  = (monthly['ret_pct'] > 0).sum()
    total_m= len(monthly)

    total_pnl = risk.equity - Config.initial_balance
    total_ret = total_pnl / Config.initial_balance * 100

    t = pd.DataFrame(trades) if len(trades) else pd.DataFrame(columns=['pnl'])
    win_rate = (t['pnl'] > 0).mean() * 100 if len(t) else 0
    pf = (
        t.loc[t['pnl'] > 0, 'pnl'].sum() /
        abs(t.loc[t['pnl'] < 0, 'pnl'].sum())
    ) if len(t) and abs(t.loc[t['pnl'] < 0, 'pnl'].sum()) > 0 else np.nan

    print("\n" + "=" * 72)
    print(f"{name}")
    print("=" * 72)
    print(f"Final Equity        : ${risk.equity:,.2f}")
    print(f"Total PnL           : ${total_pnl:,.2f}")
    print(f"Total Return        : {total_ret:.2f}%")
    print(f"Max Drawdown        : {risk.max_dd:.2f}%")
    print(f"Max DD $            : ${risk.max_dd_abs:,.2f}")
    print(f"Trades              : {len(t)}")
    print(f"Win Rate            : {win_rate:.1f}%")
    print(f"Profit Factor       : {pf:.2f}" if pd.notna(pf) else "Profit Factor       : N/A")
    print("-" * 72)
    print(f"Avg Monthly Return  : {avg_m:.2f}%")
    print(f"Median Monthly Ret  : {med_m:.2f}%")
    print(f"Best Month          : {best_m:.2f}%")
    print(f"Worst Month         : {worst_m:.2f}%")
    print(f"Positive Months     : {pos_m}/{total_m} ({pos_m/total_m*100:.1f}%)")
    print("-" * 72)
    print(monthly.to_string())
    print("=" * 72)

    monthly.to_csv(f"monthly_{name}.csv", encoding="utf-8-sig")

    return {
        'name': name,
        'equity': risk.equity,
        'total_pnl': total_pnl,
        'total_ret': total_ret,
        'max_dd': risk.max_dd,
        'max_dd_abs': risk.max_dd_abs,
        'avg_monthly': avg_m,
        'median_monthly': med_m,
        'best_month': best_m,
        'worst_month': worst_m,
        'positive_months': pos_m,
        'total_months': total_m,
        'monthly_df': monthly,
        'trades': len(t),
        'win_rate': win_rate,
        'pf': pf,
    }


# ============================================================
# 7) MAIN - فقط CorrArb + RSI_Div
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  CLEAN TEST: CorrArb + RSI_Div")
    print("=" * 72)

    df = load_data()

    print("\n⚙️ Building CorrArb original...")
    sig_corr, z_corr = strategy_corr_arb_original(df)
    print(f"   → {(sig_corr['signal'] != 0).sum()} signals")

    print("⚙️ Building RSI_Div proven...")
    sig_rsi = strategy_rsi_div_proven(df)
    print(f"   → {(sig_rsi['signal'] != 0).sum()} signals")

    # =========================
    # Single: CorrArb
    # =========================
    tr_corr, rk_corr = run_single_clean(
        df=df,
        name='CorrArb',
        signals=sig_corr,
        symbol='EUR',
        zscore=z_corr,
        time_stop_h=96,
        trailing=False
    )
    rep_corr = report_focus("CorrArb", tr_corr, rk_corr, df.index[0], df.index[-1])

    # =========================
    # Single: RSI_Div
    # =========================
    tr_rsi, rk_rsi = run_single_clean(
        df=df,
        name='RSI_Div',
        signals=sig_rsi,
        symbol='EUR',
        zscore=None,
        time_stop_h=48,
        trailing=True
    )
    rep_rsi = report_focus("RSI_Div", tr_rsi, rk_rsi, df.index[0], df.index[-1])

    # =========================
    # Combined: CorrArb + RSI_Div
    # =========================
    strategy_map = {
        'CorrArb': {
            'signals': sig_corr,
            'symbol': 'EUR',
            'zscore': z_corr,
            'time_stop_h': 96,
            'trailing': False,
        },
        'RSI_Div': {
            'signals': sig_rsi,
            'symbol': 'EUR',
            'zscore': None,
            'time_stop_h': 48,
            'trailing': True,
        }
    }

    tr_comb, rk_comb = run_combined_clean(df, strategy_map)
    rep_comb = report_focus("Combined_CorrArb_RSI", tr_comb, rk_comb, df.index[0], df.index[-1])

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for rep in [rep_corr, rep_rsi, rep_comb]:
        if rep is None:
            continue
        print(
            f"{rep['name']:<24} | "
            f"AvgMo={rep['avg_monthly']:>6.2f}% | "
            f"MedMo={rep['median_monthly']:>6.2f}% | "
            f"Best={rep['best_month']:>6.2f}% | "
            f"Worst={rep['worst_month']:>6.2f}% | "
            f"DD={rep['max_dd']:>6.2f}% | "
            f"PF={rep['pf'] if pd.notna(rep['pf']) else np.nan:>5.2f}"
        )

    print("\n✅ Saved monthly CSVs:")
    print("   monthly_CorrArb.csv")
    print("   monthly_RSI_Div.csv")
    print("   monthly_Combined_CorrArb_RSI.csv")
