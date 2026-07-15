# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path

# ==========================================
# 🎯 USER CONFIGURATION (CHANGE THESE)
# ==========================================
SYMBOL = "EURGBP"
TIMEFRAME = "M1"
START_YEAR = 2012
END_YEAR = 2025
# ==========================================

class TradingCosts(BaseModel):
    """
    Realistic trading costs for AUDUSD (major pair, USD is quote currency).
    🔥 FIXED: قبلاً این مقادیر برای AUDNZD کالیبره شده بودند ولی روی
    AUDUSD اعمال می‌شدند (ناسازگاری کامل بین کامنت‌ها و نماد واقعی).
    الان مقادیر برای AUDUSD واقعی است.
    """
    spread_pips: float = Field(default=0.9, description="AUDUSD typical retail/ECN spread")
    slippage_pips: float = Field(default=0.2, description="Average slippage per execution")
    commission_per_lot_usd: float = Field(default=5.0, description="Commission per 1 Lot per side (ECN-style account)")

    # 🔥 FIX: چون quote currency خود AUDUSD همان USD است:
    # 1 pip (0.0001) روی 1 لات استاندارد (100,000 واحد) = دقیقاً 10 دلار.
    # (برخلاف AUDNZD که نیاز به تبدیل نرخ NZD/USD داشت)
    pip_value_usd_per_lot: float = Field(default=10.0, description="Value of 1 pip for 1 Lot in USD (AUDUSD)")
    pip_size: float = Field(default=0.0001, description="0.0001 for 4-digit pairs like AUDUSD")


class StrategyParams(BaseModel):
    """
    Trend Following Strategy Parameters.

    ⚠️ هشدار Overfitting:
    این پارامترها عیناً از تست روی EURGBP کپی شده‌اند تا robustness
    استراتژی سنجیده شود. این رویکرد فقط زمانی معتبر است که این اعداد
    هرگز بر اساس نتیجه‌ی بک‌تست روی AUDUSD یا همین بازه‌ی زمانی دوباره
    tune نشوند. اگر بعداً این مقادیر را برای بهتر شدن نتیجه تغییر دادید،
    دیگر این یک تست robustness نیست بلکه curve-fitting روی داده‌ی
    تاریخی است.
    """
    ema_trend_period: int = 200
    donchian_period: int = 20

    atr_period: int = 14
    initial_sl_atr_mult: float = 2.0
    trail_atr_mult: float = 2.5

    # Time filters (UTC) - فقط برای مجاز بودن ورود جدید استفاده می‌شود،
    # نه برای فیلتر کردن کل داده (نگاه کن به strategy.py)
    london_start_hour: int = 7
    london_end_hour: int = 16


class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    risk_per_trade_percent: float = 0.01

    # 🔥 NEW: کنترل صریح روش تعیین حجم معامله
    # False -> لات ثابت (fixed_lot_size)، برای سنجش expectancy خام
    #          استراتژی مستقل از اثر money management.
    #          ⚠️ در این حالت total_return_pct و max_drawdown_pct معیار
    #          ریسک واقعی حساب حقیقی نیستند.
    # True  -> position sizing واقعی بر اساس risk_per_trade_percent و ATR
    #          لحظه‌ی ورود. این حالت باید برای گزارش نهایی و تصمیم واقعی
    #          استفاده شود.
    use_dynamic_position_sizing: bool = False
    fixed_lot_size: float = 0.1

    data_dir: Path = Path("data")

    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    start_year: int = START_YEAR
    end_year: int = END_YEAR

    @property
    def parquet_filename(self) -> str:
        # Will generate: AUDUSD_M15_2012_2025.parquet
        return f"{self.symbol}_{self.timeframe}_{self.start_year}_{self.end_year}.parquet"
