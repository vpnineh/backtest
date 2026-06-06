"""
meta_trainer.py — Meta-Learner + Walk-Forward Training
═══════════════════════════════════════════════════════
لایه ۴: ترکیب Model A + B + C با Meta-Learner
لایه ۵: Prop Risk Engine
لایه ۶: Walk-Forward Evaluation
"""

import numpy as np
import pandas as pd
import warnings, os, json
from datetime import datetime
warnings.filterwarnings('ignore')

from features import build_features, walk_forward_splits, get_feature_cols
from models   import ModelA_GBM, ModelB_LSTM, print_feature_importance
from rl_agent import PropTradingEnv, SimplePPOAgent


# ═══════════════════════════════════════════════════════════════════════════
#  Meta-Learner
# ═══════════════════════════════════════════════════════════════════════════

class MetaLearner:
    """
    ترکیب سیگنال‌های سه مدل:
      - Model A (GBM): prob_long_A, prob_short_A, prob_none_A
      - Model B (LSTM): prob_long_B, prob_short_B
      - Model C (RL): action_C
    
    خروجی نهایی: signal + confidence
    """
    
    def __init__(self):
        self.weights_long  = {'A': 0.50, 'B': 0.30, 'C': 0.20}
        self.weights_short = {'A': 0.50, 'B': 0.30, 'C': 0.20}
        self.thresh_long   = 0.55
        self.thresh_short  = 0.55
        self._calibrated   = False
    
    def combine(self,
                proba_A: np.ndarray,  # [N, 3]: none, long, short
                proba_B: np.ndarray,  # [N, 3]: none, long, short
                rl_actions: np.ndarray,  # [N]: 0=hold, 1=long, 2=short, 3=close
                ) -> tuple:
        """
        ترکیب احتمالات سه مدل.
        خروجی: (signals, confidence)
          signals: -1, 0, +1
          confidence: 0..1
        """
        N = len(proba_A)
        
        # RL رو به probability تبدیل کن
        rl_long  = (rl_actions == 1).astype(float)
        rl_short = (rl_actions == 2).astype(float)
        
        # weighted combination
        score_long = (
            self.weights_long['A']  * proba_A[:, 1] +
            self.weights_long['B']  * proba_B[:, 1] +
            self.weights_long['C']  * rl_long
        )
        score_short = (
            self.weights_short['A'] * proba_A[:, 2] +
            self.weights_short['B'] * proba_B[:, 2] +
            self.weights_short['C'] * rl_short
        )
        
        signals    = np.zeros(N, dtype=int)
        confidence = np.maximum(score_long, score_short)
        
        signals[score_long  > self.thresh_long]  =  1
        signals[score_short > self.thresh_short] = -1
        
        # conflict resolution: هر دو high → بزرگتر برنده
        conflict = (score_long > self.thresh_long) & (score_short > self.thresh_short)
        signals[conflict & (score_long >= score_short)] =  1
        signals[conflict & (score_long <  score_short)] = -1
        
        return signals, confidence
    
    def calibrate(self, val_signals: np.ndarray, val_returns: np.ndarray,
                  thresholds: list = None):
        """
        بهینه‌سازی threshold روی validation set.
        معیار: Sharpe-like metric
        """
        if thresholds is None:
            thresholds = np.arange(0.45, 0.75, 0.025)
        
        best_thresh = self.thresh_long
        best_score  = -np.inf
        
        for t in thresholds:
            sig = np.zeros_like(val_signals)
            sig[val_signals >  t] =  1
            sig[val_signals < -t] = -1
            
            rets = sig * val_returns
            if rets.std() > 0:
                score = rets.mean() / rets.std() * np.sqrt(252*96)
                if score > best_score:
                    best_score  = score
                    best_thresh = t
        
        self.thresh_long  = best_thresh
        self.thresh_short = best_thresh
        self._calibrated  = True
        print(f"  Meta threshold calibrated: {best_thresh:.3f} (score={best_score:.2f})")


# ═══════════════════════════════════════════════════════════════════════════
#  Prop Risk Engine
# ═══════════════════════════════════════════════════════════════════════════

class PropRiskEngine:
    """
    قوانین سخت پراپ + dynamic position sizing.
    این لایه هیچوقت ML رو override نمیکنه — فقط محدود میکنه.
    """
    
    INITIAL_BALANCE    = 5_000.0
    PROFIT_TARGET_PCT  = 0.05
    MAX_DAILY_DD_PCT   = 0.04
    MAX_TOTAL_DD_PCT   = 0.08
    BASE_RISK          = 0.008
    MIN_RISK           = 0.004
    SPREAD_PIPS        = 1.2
    COMMISSION         = 7.0
    SLIPPAGE_PIPS      = 0.3
    PIP                = 0.0001
    LOT_SIZE           = 100_000
    MAX_LOT            = 2.0
    MIN_LOT            = 0.01
    SL_PIPS            = 20.0
    TP_PIPS            = 44.0
    MAX_TRADES_PER_DAY = 2
    TIME_STOP_BARS     = 160
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.equity       = self.INITIAL_BALANCE
        self.peak         = self.INITIAL_BALANCE
        self.day_start    = self.INITIAL_BALANCE
        self.cur_day      = None
        self.trades_today = 0
        self.open_pos     = None
        self.consec_loss  = 0
        self.consec_win   = 0
        self.blown        = False
        self.blown_reason = ""
        self.trades       = []
    
    def _cost(self, lot): 
        return (self.SPREAD_PIPS*2*self.PIP*lot*self.LOT_SIZE +
                self.COMMISSION * lot)
    
    def _lot(self) -> float:
        risk = self.BASE_RISK
        if self.consec_loss >= 3:
            risk = max(risk * 0.65, self.MIN_RISK)
        raw = self.equity * risk / (self.SL_PIPS * self.PIP * self.LOT_SIZE)
        return round(float(np.clip(raw, self.MIN_LOT, self.MAX_LOT)), 2)
    
    def _check_dd(self) -> tuple:
        dd_day = (self.equity - self.day_start) / max(self.day_start, 1)
        if dd_day <= -self.MAX_DAILY_DD_PCT:
            return True, f"DailyDD {dd_day*100:.2f}%"
        dd_tot = (self.equity - self.INITIAL_BALANCE) / self.INITIAL_BALANCE
        if dd_tot <= -self.MAX_TOTAL_DD_PCT:
            return True, f"TotalDD {dd_tot*100:.2f}%"
        return False, ""
    
    def allowed_to_trade(self, ts, signal: int, confidence: float,
                         min_confidence: float = 0.0) -> bool:
        """آیا مجاز به ورود هستیم؟"""
        if self.blown or self.open_pos is not None:
            return False
        
        day = ts.date() if hasattr(ts, 'date') else ts
        if day != self.cur_day:
            self.cur_day      = day
            self.day_start    = self.equity
            self.trades_today = 0
        
        if self.trades_today >= self.MAX_TRADES_PER_DAY:
            return False
        
        if confidence < min_confidence:
            return False
        
        return signal != 0
    
    def open_trade(self, ts, signal: int, open_price: float, bar: int):
        if self.open_pos is not None:
            return
        d   = signal
        lot = self._lot()
        ep  = open_price + d * (self.SLIPPAGE_PIPS + self.SPREAD_PIPS/2) * self.PIP
        sl  = ep - d * self.SL_PIPS * self.PIP
        tp  = ep + d * self.TP_PIPS * self.PIP
        
        self.open_pos = dict(
            dir=d, lot=lot, entry=ep, sl=sl, tp=tp,
            entry_ts=ts, entry_bar=bar,
        )
        self.trades_today += 1
    
    def update(self, ts, bar: int, high: float, low: float,
               close: float, z_val: float = None) -> dict:
        """
        بروزرسانی per-bar.
        خروجی: dict با اطلاعات closed trade (یا None)
        """
        day = ts.date() if hasattr(ts, 'date') else ts
        if day != self.cur_day:
            self.cur_day   = day
            self.day_start = self.equity
            self.trades_today = 0
        
        if self.equity > self.peak:
            self.peak = self.equity
        
        if self.open_pos is None:
            return None
        
        pos = self.open_pos
        d   = pos['dir']
        ep  = pos['entry']
        sl  = pos['sl']
        tp  = pos['tp']
        
        hit_sl = (d ==  1 and low  <= sl) or (d == -1 and high >= sl)
        hit_tp = (d ==  1 and high >= tp) or (d == -1 and low  <= tp)
        
        # Z-exit
        if z_val is not None and not np.isnan(z_val) and abs(z_val) < 0.5:
            hit_tp = True
        
        if hit_sl and hit_tp:
            hit_tp = False
        
        # Trailing stop
        tp_dist = abs(tp - ep)
        if tp_dist > 0:
            prog = d * (close - ep) / tp_dist
            if prog >= 0.50:
                be = ep + d * tp_dist * 0.08
                if d == 1 and be > sl: pos['sl'] = be; sl = be
                elif d == -1 and be < sl: pos['sl'] = be; sl = be
            if prog >= 0.75:
                lock = ep + d * tp_dist * 0.45
                if d == 1 and lock > sl: pos['sl'] = lock; sl = lock
                elif d == -1 and lock < sl: pos['sl'] = lock; sl = lock
        
        # Time stop
        if bar - pos['entry_bar'] >= self.TIME_STOP_BARS and not hit_tp and not hit_sl:
            pnl = d*(close-ep)*pos['lot']*self.LOT_SIZE - self._cost(pos['lot'])
            return self._close(pos, ts, close, pnl, 'TP_time' if pnl > 0 else 'SL_time')
        
        if hit_sl or hit_tp:
            exit_px = sl if hit_sl else tp
            status  = 'SL' if hit_sl else 'TP'
            pnl = d*(exit_px-ep)*pos['lot']*self.LOT_SIZE - self._cost(pos['lot'])
            return self._close(pos, ts, exit_px, pnl, status)
        
        return None
    
    def _close(self, pos, ts, exit_px, pnl, status) -> dict:
        self.equity += pnl
        rec = {**pos, 'exit': exit_px, 'exit_ts': ts,
               'pnl': pnl, 'status': status}
        self.trades.append(rec)
        self.open_pos = None
        
        if pnl > 0:
            self.consec_win += 1; self.consec_loss = 0
        else:
            self.consec_loss += 1; self.consec_win = 0
        
        blown, rsn = self._check_dd()
        self.blown = blown; self.blown_reason = rsn
        
        return rec
    
    @property
    def target_hit(self) -> bool:
        return self.equity >= self.INITIAL_BALANCE * (1 + self.PROFIT_TARGET_PCT)
    
    @property
    def max_dd_pct(self) -> float:
        if self.peak <= 0: return 0.0
        return (self.equity - self.peak) / self.peak * 100


# ═══════════════════════════════════════════════════════════════════════════
#  Walk-Forward Trainer & Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardTrainer:
    
    def __init__(self, output_dir: str = 'ml_models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.split_results = []
    
    def train_split(self, ft: pd.DataFrame, df_prices: pd.DataFrame,
                    split: dict, split_idx: int) -> dict:
        """
        آموزش روی یه split و ارزیابی روی test.
        """
        print(f"\n{'─'*60}")
        print(f"  Split {split_idx}: test={split['test_start'].date()}→{split['test_end'].date()}")
        
        feat_cols = get_feature_cols(ft)
        
        # ── داده train/val ──
        tr_ft = ft.loc[split['train']].dropna()
        vl_ft = ft.loc[split['val']].dropna()
        ts_ft = ft.loc[split['test']].dropna()
        
        if len(tr_ft) < 200 or len(ts_ft) < 50:
            print("  ⚠️ داده کافی نیست — skip")
            return None
        
        X_tr = tr_ft[feat_cols]; y_tr = tr_ft['target'].values.astype(int)
        X_vl = vl_ft[feat_cols]; y_vl = vl_ft['target'].values.astype(int)
        X_ts = ts_ft[feat_cols]; y_ts = ts_ft['target'].values.astype(int)
        
        # ── Model A ──
        model_a = ModelA_GBM(use_lightgbm=True)
        model_a.fit(X_tr, y_tr, X_vl, y_vl)
        proba_A_test = model_a.predict_proba(X_ts)
        
        # Feature importance (فقط split اول)
        if split_idx == 0:
            print_feature_importance(model_a, feat_cols)
        
        # ── Model B (LSTM) ──
        model_b = ModelB_LSTM(lookback=48, n_features=30)
        try:
            model_b.fit(X_tr, y_tr, X_vl, y_vl)
            proba_B_test = model_b.predict_proba_batch(X_ts)
        except Exception as e:
            print(f"  ⚠️ LSTM خطا: {e} — از uniform استفاده میشه")
            proba_B_test = np.ones((len(X_ts), 3)) / 3
        
        # ── Model C (RL) ──
        tr_prices  = df_prices.loc[split['train']]
        tr_ft_rl   = tr_ft[feat_cols].fillna(0)
        
        rl_env = PropTradingEnv(tr_ft_rl, tr_prices, lookback=24, training=True)
        rl_agent = SimplePPOAgent(rl_env.obs_dim)
        
        try:
            rl_agent.train(rl_env, total_timesteps=50_000)
        except Exception as e:
            print(f"  ⚠️ RL خطا: {e} — هیچ action نمیده")
        
        # RL actions روی test
        rl_actions_test = self._run_rl_on_test(rl_agent, ts_ft[feat_cols], df_prices.loc[split['test']])
        
        # ── Meta-Learner ──
        meta = MetaLearner()
        
        # Calibration روی val
        proba_A_val = model_a.predict_proba(X_vl)
        proba_B_val = model_b.predict_proba_batch(X_vl) if model_b._fitted else np.ones((len(X_vl), 3))/3
        rl_val_actions = np.zeros(len(X_vl), dtype=int)
        
        raw_signals_val, conf_val = meta.combine(proba_A_val, proba_B_val, rl_val_actions)
        vl_returns = vl_ft['target_ret'].values
        meta.calibrate(raw_signals_val.astype(float), vl_returns)
        
        # Final signals روی test
        signals_test, conf_test = meta.combine(proba_A_test, proba_B_test, rl_actions_test)
        
        # ── Prop Backtest روی test ──
        result = self._run_prop_backtest(
            signals_test, conf_test,
            ts_ft, df_prices.loc[split['test']],
            split_idx
        )
        
        result['split_idx']    = split_idx
        result['test_start']   = split['test_start']
        result['test_end']     = split['test_end']
        result['n_signals']    = int((signals_test != 0).sum())
        result['meta_thresh']  = meta.thresh_long
        
        self.split_results.append(result)
        
        # ذخیره مدل‌ها
        model_a.save(f"{self.output_dir}/model_a_split{split_idx}.pkl")
        
        return result
    
    def _run_rl_on_test(self, agent, X_test, prices_test) -> np.ndarray:
        """اجرای RL agent روی test برای گرفتن actions"""
        actions = np.zeros(len(X_test), dtype=int)
        
        if not agent._fitted:
            return actions
        
        try:
            env = PropTradingEnv(X_test, prices_test, lookback=24, training=False)
            obs = env.reset()
            for i in range(len(X_test)):
                action = agent.predict(obs)
                actions[i] = action
                obs, _, done, _ = env.step(action)
                if done:
                    break
        except Exception:
            pass
        
        return actions
    
    def _run_prop_backtest(self, signals: np.ndarray, confidence: np.ndarray,
                           ft_test: pd.DataFrame, prices_test: pd.DataFrame,
                           split_idx: int) -> dict:
        """
        بک‌تست کامل پراپ روی test period.
        """
        engine = PropRiskEngine()
        
        z_vals = ft_test['z_ratio_96'].values if 'z_ratio_96' in ft_test.columns else None
        opens  = prices_test['o_eur'].values
        highs  = prices_test['h_eur'].values
        lows   = prices_test['l_eur'].values
        closes = prices_test['c_eur'].values
        times  = ft_test.index
        
        all_trades  = []
        eq_curve    = [engine.INITIAL_BALANCE]
        n_acc       = 1
        n_target    = 0
        n_blown     = 0
        withdrawals = 0.0
        
        for i in range(len(signals)):
            ts   = times[i]
            sig  = int(signals[i])
            conf = float(confidence[i])
            z    = float(z_vals[i]) if z_vals is not None else np.nan
            
            # بروزرسانی پوزیشن
            if i > 0:
                trade_rec = engine.update(ts, i, highs[i], lows[i], closes[i], z)
                if trade_rec:
                    all_trades.append(trade_rec)
            
            # بررسی blown/target
            if engine.blown:
                n_blown += 1
                n_acc   += 1
                engine.reset()
            
            if engine.target_hit and engine.open_pos is None:
                w = engine.equity - engine.INITIAL_BALANCE
                withdrawals += w
                n_target    += 1
                n_acc       += 1
                engine.reset()
            
            # ورود trade جدید
            if engine.allowed_to_trade(ts, sig, conf, min_confidence=0.0):
                engine.open_trade(ts, sig, opens[min(i+1, len(opens)-1)], i)
            
            eq_curve.append(engine.equity)
        
        # محاسبه آمار
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        
        if len(trades_df) == 0:
            return {
                'trades': 0, 'win_rate': 0, 'pf': 0, 'sharpe': 0,
                'max_dd': 0, 'total_ret': 0, 'n_blown': n_blown,
                'n_target': n_target, 'withdrawals': withdrawals,
            }
        
        pnls = trades_df['pnl'].values
        wins = pnls[pnls > 0]
        loss = pnls[pnls < 0]
        
        wr = len(wins)/len(pnls)*100 if len(pnls) else 0
        pf = wins.sum()/abs(loss.sum()) if len(loss) and loss.sum() != 0 else 99
        
        eq = np.array(eq_curve)
        dd = self._max_dd(eq)
        
        rets = pd.Series(eq).pct_change().dropna()
        sharpe = rets.mean()/rets.std()*np.sqrt(252*96) if rets.std() > 0 else 0
        
        total_val = engine.equity + withdrawals
        total_ret = (total_val - engine.INITIAL_BALANCE) / engine.INITIAL_BALANCE * 100
        
        n_months = max((times[-1] - times[0]).days / 30, 1)
        monthly_ret = total_ret / n_months
        
        print(f"  Test: trades={len(pnls)} | WR={wr:.1f}% | PF={pf:.2f} | "
              f"DD={dd:.1f}% | Sharpe={sharpe:.2f} | Ret={total_ret:+.1f}%")
        
        return {
            'trades': len(pnls), 'win_rate': wr, 'pf': pf,
            'sharpe': sharpe, 'max_dd': dd, 'total_ret': total_ret,
            'monthly_ret': monthly_ret,
            'n_blown': n_blown, 'n_target': n_target,
            'withdrawals': withdrawals, 'pnls': pnls.tolist(),
        }
    
    def _max_dd(self, eq: np.ndarray) -> float:
        peak = eq[0]; max_dd = 0.0
        for e in eq:
            if e > peak: peak = e
            dd = (e - peak) / peak * 100
            if dd < max_dd: max_dd = dd
        return max_dd
    
    def print_summary(self):
        if not self.split_results:
            print("هیچ نتیجه‌ای نیست")
            return
        
        W = 70
        print("\n" + "═"*W)
        print("  ML CorrArb — Walk-Forward Summary")
        print("═"*W)
        
        metrics = {
            'trades':      [r.get('trades', 0)      for r in self.split_results],
            'win_rate':    [r.get('win_rate', 0)     for r in self.split_results],
            'pf':          [r.get('pf', 0)           for r in self.split_results],
            'sharpe':      [r.get('sharpe', 0)       for r in self.split_results],
            'max_dd':      [r.get('max_dd', 0)       for r in self.split_results],
            'monthly_ret': [r.get('monthly_ret', 0)  for r in self.split_results],
            'n_blown':     [r.get('n_blown', 0)      for r in self.split_results],
            'n_target':    [r.get('n_target', 0)     for r in self.split_results],
        }
        
        def rw(lbl, vals, fmt='.2f', ok=None):
            avg = np.mean(vals); med = np.median(vals)
            v = f"avg={avg:{fmt}}  med={med:{fmt}}"
            m = "" if ok is None else (" ✅" if ok else " ❌")
            d = "·" * max(2, W - len(lbl) - len(v) - len(m) - 4)
            return f"  {lbl} {d} {v}{m}"
        
        print(rw("معاملات per split",  metrics['trades'],      '.0f'))
        print(rw("Win Rate %",          metrics['win_rate'],    '.1f',  np.mean(metrics['win_rate']) >= 52))
        print(rw("Profit Factor",       metrics['pf'],          '.2f',  np.mean(metrics['pf']) > 1.3))
        print(rw("Sharpe",             metrics['sharpe'],       '.2f',  np.mean(metrics['sharpe']) > 1.0))
        print(rw("Max DD %",           [abs(d) for d in metrics['max_dd']], '.1f', np.mean([abs(d) for d in metrics['max_dd']]) < 8.0))
        print(rw("Monthly Ret %",      metrics['monthly_ret'], '.2f',  np.mean(metrics['monthly_ret']) > 3.0))
        print(rw("Blown accounts",     metrics['n_blown'],     '.0f',  np.mean(metrics['n_blown']) == 0))
        print(rw("Target hits",        metrics['n_target'],    '.0f',  np.mean(metrics['n_target']) > 0))
        print("═"*W)
        
        # جدول per-split
        print(f"\n  {'Split':>5}  {'Period':>21}  {'Trades':>6}  {'WR%':>5}  "
              f"{'PF':>4}  {'Sharpe':>6}  {'DD%':>6}  {'Ret%':>6}")
        print("  " + "─"*(W-2))
        for r in self.split_results:
            print(f"  {r['split_idx']:>5}  "
                  f"{str(r['test_start'].date()):>10}→{str(r['test_end'].date()):>10}  "
                  f"{r.get('trades',0):>6}  "
                  f"{r.get('win_rate',0):>4.1f}%  "
                  f"{r.get('pf',0):>4.2f}  "
                  f"{r.get('sharpe',0):>6.2f}  "
                  f"{r.get('max_dd',0):>5.1f}%  "
                  f"{r.get('total_ret',0):>+5.1f}%")
        
        # ذخیره نتایج
        with open(f"{self.output_dir}/wf_results.json", 'w') as f:
            clean = [{k: v for k, v in r.items() if k != 'pnls'} for r in self.split_results]
            for c in clean:
                for k, v in c.items():
                    if isinstance(v, (np.integer, np.floating)):
                        c[k] = float(v)
                    elif hasattr(v, 'isoformat'):
                        c[k] = str(v)
            json.dump(clean, f, indent=2)
        print(f"\n  ✅ نتایج ذخیره شد: {self.output_dir}/wf_results.json")


# ═══════════════════════════════════════════════════════════════════════════
#  تست
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("meta_trainer.py — تست Meta-Learner")
    
    np.random.seed(42)
    N = 500
    
    pA = np.random.dirichlet([1,1,1], N)
    pB = np.random.dirichlet([1,1,1], N)
    rl = np.random.randint(0, 4, N)
    
    meta = MetaLearner()
    sigs, conf = meta.combine(pA, pB, rl)
    print(f"  Signals: long={( sigs==1).sum()} | short={(sigs==-1).sum()} | none={(sigs==0).sum()}")
    print("✅ Meta-Learner OK")
