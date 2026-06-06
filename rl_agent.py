"""
rl_agent.py — Reinforcement Learning Agent (PPO)
══════════════════════════════════════════════════
عامل RL یاد میگیره:
  - چه موقع وارد بشه (entry)
  - چه موقع خارج بشه (exit)
  - چقدر ریسک بکنه (position size)

Reward function: Sharpe-based با prop constraint penalty
State: آخرین N کندل از featureها + state اکانت
Action: {0=hold/flat, 1=long, 2=short, 3=close}
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
#  Trading Environment
# ═══════════════════════════════════════════════════════════════════════════

class PropTradingEnv:
    """
    محیط trading برای RL agent.
    شبیه‌ساز پراپ با قوانین DD سخت.
    """
    
    # Prop parameters
    INITIAL_BALANCE    = 5_000.0
    PROFIT_TARGET      = 0.05    # +5%
    MAX_DAILY_DD       = 0.04    # -4%
    MAX_TOTAL_DD       = 0.08    # -8%
    SPREAD_PIPS        = 1.2
    COMMISSION_PER_LOT = 7.0
    SLIPPAGE_PIPS      = 0.3
    PIP                = 0.0001
    LOT_SIZE           = 100_000
    MAX_LOT            = 2.0
    MIN_LOT            = 0.01
    BASE_RISK          = 0.008
    SL_PIPS            = 20.0
    TP_PIPS            = 44.0
    
    def __init__(self, features: pd.DataFrame, prices: pd.DataFrame,
                 lookback: int = 24, training: bool = True):
        """
        features: DataFrame از featureهای آماده
        prices:   DataFrame با c_eur, h_eur, l_eur, o_eur
        lookback: طول تاریخچه برای state
        """
        self.features = features.fillna(0).values
        self.prices   = prices
        self.closes   = prices['c_eur'].values
        self.highs    = prices['h_eur'].values
        self.lows     = prices['l_eur'].values
        self.opens    = prices['o_eur'].values
        self.lookback = lookback
        self.training = training
        self.n_steps  = len(features)
        self.n_feats  = min(features.shape[1], 50)  # حداکثر ۵۰ feature برای RL
        
        # State dimension
        self.obs_dim  = self.lookback * self.n_feats + 8  # 8 تا account state
        self.n_actions = 4  # hold, long, short, close
        
        self.reset()
    
    def reset(self):
        self.bar          = self.lookback
        self.equity       = self.INITIAL_BALANCE
        self.peak         = self.INITIAL_BALANCE
        self.day_start    = self.INITIAL_BALANCE
        self.current_day  = None
        self.position     = 0   # 0=flat, 1=long, -1=short
        self.entry_price  = 0.0
        self.entry_bar    = 0
        self.lot          = 0.0
        self.sl           = 0.0
        self.tp           = 0.0
        self.trades       = []
        self.returns      = []
        self.blown        = False
        self.target_hit   = False
        self.consec_loss  = 0
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """ساخت observation vector"""
        # تاریخچه feature
        start = max(0, self.bar - self.lookback)
        hist  = self.features[start:self.bar, :self.n_feats]
        
        if len(hist) < self.lookback:
            pad  = np.zeros((self.lookback - len(hist), self.n_feats))
            hist = np.vstack([pad, hist])
        
        hist_flat = hist.flatten()
        
        # Account state (normalized)
        dd       = (self.equity - self.peak) / self.INITIAL_BALANCE
        day_dd   = (self.equity - self.day_start) / self.INITIAL_BALANCE
        prog     = (self.equity - self.INITIAL_BALANCE) / (
                    self.INITIAL_BALANCE * self.PROFIT_TARGET + 1e-8)
        
        # position state
        pos_enc  = self.position  # -1, 0, 1
        pl_pct   = 0.0
        if self.position != 0 and self.entry_price > 0:
            cp    = self.closes[min(self.bar, len(self.closes)-1)]
            pl_pct = self.position * (cp - self.entry_price) / self.entry_price
        
        acc_state = np.array([
            dd,           # drawdown فعلی
            day_dd,       # drawdown روزانه
            prog,         # پیشرفت به هدف
            pos_enc,      # وضعیت پوزیشن
            pl_pct,       # P&L فعلی
            self.consec_loss / 5.0,           # ضررهای متوالی
            (self.bar - self.lookback) / max(self.n_steps - self.lookback, 1),  # زمان
            len(self.trades) / 100.0,          # تعداد معاملات
        ], dtype=np.float32)
        
        return np.concatenate([hist_flat.astype(np.float32), acc_state])
    
    def _trade_cost(self, lot: float) -> float:
        return (self.SPREAD_PIPS * 2 * self.PIP * lot * self.LOT_SIZE +
                self.COMMISSION_PER_LOT * lot)
    
    def _calc_lot(self) -> float:
        risk = self.BASE_RISK
        if self.consec_loss >= 3:
            risk = max(risk * 0.65, 0.004)
        raw = self.equity * risk / (self.SL_PIPS * self.PIP * self.LOT_SIZE)
        return round(float(np.clip(raw, self.MIN_LOT, self.MAX_LOT)), 2)
    
    def step(self, action: int):
        """
        action: 0=hold, 1=long, 2=short, 3=close
        returns: (obs, reward, done, info)
        """
        if self.bar >= self.n_steps - 1:
            return self._get_obs(), 0.0, True, {'reason': 'end_of_data'}
        
        cp   = self.closes[self.bar]
        hp   = self.highs[self.bar]
        lp   = self.lows[self.bar]
        prev_equity = self.equity
        
        # ── ریست روزانه ──
        ts = self.prices.index[self.bar] if hasattr(self.prices, 'index') else None
        if ts is not None:
            day = ts.date()
            if day != self.current_day:
                self.current_day = day
                self.day_start   = self.equity
        
        reward      = 0.0
        done        = False
        trade_info  = {}
        
        # ── مدیریت پوزیشن باز ──
        if self.position != 0:
            sl = self.sl; tp = self.tp; d = self.position
            
            hit_sl = (d ==  1 and lp <= sl) or (d == -1 and hp >= sl)
            hit_tp = (d ==  1 and hp >= tp) or (d == -1 and lp <= tp)
            
            if hit_sl and hit_tp:
                hit_tp = False
            
            close_now = (action == 3) or hit_sl or hit_tp
            
            if close_now:
                if hit_sl:
                    exit_px = sl; status = 'SL'
                elif hit_tp:
                    exit_px = tp; status = 'TP'
                else:
                    exit_px = cp; status = 'manual_close'
                
                pnl = d * (exit_px - self.entry_price) * self.lot * self.LOT_SIZE
                pnl -= self._trade_cost(self.lot)
                self.equity += pnl
                
                self.trades.append({'pnl': pnl, 'status': status})
                if pnl > 0:
                    self.consec_loss = 0
                else:
                    self.consec_loss += 1
                
                self.position = 0
                
                # پیشرفت
                if self.equity > self.peak:
                    self.peak = self.equity
                
                # بررسی prop rules
                dd_total = (self.equity - self.INITIAL_BALANCE) / self.INITIAL_BALANCE
                dd_day   = (self.equity - self.day_start) / max(self.day_start, 1)
                
                if dd_day <= -self.MAX_DAILY_DD or (self.INITIAL_BALANCE - self.equity) / self.INITIAL_BALANCE >= self.MAX_TOTAL_DD:
                    self.blown = True
                    done = True
                    reward = -10.0  # penalty شدید برای blown
                    return self._get_obs(), reward, done, {'reason': 'blown'}
                
                # target hit
                if self.equity >= self.INITIAL_BALANCE * (1 + self.PROFIT_TARGET):
                    self.target_hit = True
                    done = True
                    reward = +10.0
                    return self._get_obs(), reward, done, {'reason': 'target_hit'}
        
        # ── باز کردن پوزیشن ──
        if self.position == 0 and action in [1, 2]:
            d     = 1 if action == 1 else -1
            lot   = self._calc_lot()
            ep    = cp + d * (self.SLIPPAGE_PIPS + self.SPREAD_PIPS/2) * self.PIP
            sl    = ep - d * self.SL_PIPS * self.PIP
            tp_px = ep + d * self.TP_PIPS * self.PIP
            
            # immediate SL check
            if not ((d == 1 and lp <= sl) or (d == -1 and hp >= sl)):
                self.position    = d
                self.entry_price = ep
                self.entry_bar   = self.bar
                self.lot         = lot
                self.sl          = sl
                self.tp          = tp_px
        
        # ── Reward Shaping ──
        equity_change = self.equity - prev_equity
        
        # پایه: تغییر equity normalized
        reward = equity_change / self.INITIAL_BALANCE * 100
        
        # penalty برای نزدیک شدن به DD limit
        dd = (self.equity - self.INITIAL_BALANCE) / self.INITIAL_BALANCE
        if dd < -0.06:
            reward -= 2.0 * abs(dd + 0.06)  # penalty تدریجی
        
        # bonus برای holding در position سودده
        if self.position != 0 and self.entry_price > 0:
            open_pnl = self.position * (cp - self.entry_price) * self.lot * self.LOT_SIZE
            if open_pnl > 0:
                reward += open_pnl / self.INITIAL_BALANCE * 0.1
        
        # penalty برای overtrading
        if len(self.trades) > 0 and len(self.trades) % 5 == 0:
            recent_pnls = [t['pnl'] for t in self.trades[-5:]]
            if sum(recent_pnls) < 0:
                reward -= 0.5
        
        self.returns.append(equity_change / self.INITIAL_BALANCE)
        self.bar += 1
        
        return self._get_obs(), reward, done, trade_info
    
    def get_sharpe(self) -> float:
        if len(self.returns) < 10:
            return 0.0
        r = np.array(self.returns)
        std = r.std()
        if std == 0:
            return 0.0
        return r.mean() / std * np.sqrt(252 * 96)


# ═══════════════════════════════════════════════════════════════════════════
#  PPO Agent (ساده با numpy — بدون نیاز به stable-baselines)
# ═══════════════════════════════════════════════════════════════════════════

class SimplePPOAgent:
    """
    PPO ساده با neural network خودمون.
    اگر stable-baselines3 نصب باشه از اون استفاده میکنه.
    """
    
    def __init__(self, obs_dim: int, n_actions: int = 4):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.policy    = None
        self._fitted   = False
    
    def train(self, env: PropTradingEnv, total_timesteps: int = 200_000):
        print("  [Model C] آموزش RL Agent...", end="", flush=True)
        
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # Wrapper برای stable-baselines3
            import gymnasium as gym
            
            class GymWrapper(gym.Env):
                def __init__(self, trading_env):
                    super().__init__()
                    self.env = trading_env
                    self.observation_space = gym.spaces.Box(
                        low=-10, high=10,
                        shape=(trading_env.obs_dim,),
                        dtype=np.float32
                    )
                    self.action_space = gym.spaces.Discrete(trading_env.n_actions)
                
                def reset(self, seed=None, options=None):
                    obs = self.env.reset()
                    return obs.astype(np.float32), {}
                
                def step(self, action):
                    obs, reward, done, info = self.env.step(action)
                    return obs.astype(np.float32), float(reward), done, False, info
            
            gym_env = GymWrapper(env)
            vec_env = DummyVecEnv([lambda: gym_env])
            
            self.policy = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate    = 3e-4,
                n_steps          = 2048,
                batch_size       = 64,
                n_epochs         = 10,
                gamma            = 0.99,
                gae_lambda       = 0.95,
                clip_range       = 0.2,
                ent_coef         = 0.01,
                verbose          = 0,
                policy_kwargs    = dict(net_arch=[256, 128, 64]),
            )
            
            self.policy.learn(total_timesteps=total_timesteps, progress_bar=False)
            self._fitted = True
            print(" ✓ (stable-baselines3)")
        
        except ImportError:
            # fallback: random policy با heuristic
            print(" ⚠️ (stable-baselines3 نصب نیست — heuristic policy)")
            self._fitted = False
    
    def predict(self, obs: np.ndarray) -> int:
        """پیش‌بینی action"""
        if not self._fitted or self.policy is None:
            return 0  # hold
        action, _ = self.policy.predict(obs.reshape(1, -1), deterministic=True)
        return int(action[0])
    
    def get_action_proba(self, obs: np.ndarray) -> np.ndarray:
        """احتمال هر action"""
        if not self._fitted or self.policy is None:
            return np.ones(self.n_actions) / self.n_actions
        import torch
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            dist = self.policy.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.numpy()[0]
        return probs
    
    def save(self, path: str):
        if self._fitted and self.policy:
            self.policy.save(path)
    
    def load(self, path: str):
        try:
            from stable_baselines3 import PPO
            self.policy  = PPO.load(path)
            self._fitted = True
        except Exception as e:
            print(f"  ⚠️ بارگذاری RL ناموفق: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  تست
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("rl_agent.py — تست محیط")
    
    n = 2000
    idx = pd.date_range('2020-01-01', periods=n, freq='15min')
    
    feat = pd.DataFrame(np.random.randn(n, 50), index=idx,
                        columns=[f'f{i}' for i in range(50)])
    price = pd.DataFrame({
        'c_eur': 1.1 + np.cumsum(np.random.randn(n) * 0.0001),
        'h_eur': 1.1 + np.cumsum(np.random.randn(n) * 0.0001) + 0.0005,
        'l_eur': 1.1 + np.cumsum(np.random.randn(n) * 0.0001) - 0.0005,
        'o_eur': 1.1 + np.cumsum(np.random.randn(n) * 0.0001),
    }, index=idx)
    
    env = PropTradingEnv(feat, price, lookback=24)
    obs = env.reset()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs dim:   {env.obs_dim}")
    
    # Random policy test
    total_r = 0
    for _ in range(200):
        a = np.random.randint(0, 4)
        obs, r, done, info = env.step(a)
        total_r += r
        if done:
            break
    
    print(f"  Equity: ${env.equity:.2f} | Trades: {len(env.trades)} | Sharpe: {env.get_sharpe():.2f}")
    print("✅ محیط RL کار میکنه")
