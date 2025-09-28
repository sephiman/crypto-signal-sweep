import os
import re

PAIRS = os.getenv("PAIRS", "BTC/USDT").split(",")
TIMEFRAMES = os.getenv("TIMEFRAMES", "15m").split(",")
RUN_AT_START = os.getenv("RUN_AT_START", "false").lower() == "true"
USE_HIGHER_TF_CONFIRM = os.getenv("USE_HIGHER_TF_CONFIRM", "true").lower() == "true"
USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "true").lower() == "true"
TREND_MA_PERIOD = int(os.getenv("TREND_MA_PERIOD", 21))
REQUIRED_MA_BARS = int(os.getenv("REQUIRED_MA_BARS", 2))
SEND_UNCONFIRMED = os.getenv("SEND_UNCONFIRMED", "false").lower() == "true"
DYNAMIC_SCORE_ENABLED = os.getenv("DYNAMIC_SCORE_ENABLED", "true").lower() == "true"
# global score settings
MIN_SCORE_DEFAULT = int(os.getenv("MIN_SCORE_DEFAULT", 5))
MIN_SCORE_TRENDING = int(os.getenv("MIN_SCORE_TRENDING", 5))
MIN_SCORE_RANGING = int(os.getenv("MIN_SCORE_RANGING", 4))

# Timeframe-specific minimum score requirements
TIMEFRAME_MIN_SCORES = {
    "1m": 3,
    "5m": 4,
    "15m": 5,
    "1h": 6,
    "4h": 7,
    "1d": 7,
}
MIN_ATR_RATIO = float(os.getenv("MIN_ATR_RATIO", 0.005))

ADX_PERIOD = int(os.getenv("ADX_PERIOD", 14))
ADX_THRESHOLD = int(os.getenv("ADX_THRESHOLD", 28))
ADX_RSI_MODE = os.getenv("ADX_RSI_MODE", "adx").lower()

# RSI Configuration for Ranging Markets (standard oversold/overbought)
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 28))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", 72))
RSI_MOMENTUM = float(os.getenv("RSI_MOMENTUM", 50))

# RSI Configuration for Trending Markets (when ADX_RSI_MODE="adx" and ADX >= threshold)
# Option 1: Use more extreme levels in trends (default)
RSI_TRENDING_OVERSOLD = float(os.getenv("RSI_TRENDING_OVERSOLD", 30))  # More extreme for trends
RSI_TRENDING_OVERBOUGHT = float(os.getenv("RSI_TRENDING_OVERBOUGHT", 70))  # More extreme for trends

# Option 2: Use momentum pullback levels (alternative approach)
# Set RSI_TRENDING_MODE to "pullback" to look for mild pullbacks in trends instead of extremes
RSI_TRENDING_MODE = os.getenv("RSI_TRENDING_MODE", "extreme").lower()  # "extreme" or "pullback"
RSI_TRENDING_PULLBACK_LONG = float(os.getenv("RSI_TRENDING_PULLBACK_LONG", 40))  # Buy pullbacks above this in uptrends
RSI_TRENDING_PULLBACK_SHORT = float(os.getenv("RSI_TRENDING_PULLBACK_SHORT", 60))  # Sell pullbacks below this in downtrends

MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))
MACD_MIN_DIFF = float(os.getenv("MACD_MIN_DIFF", 0.8))
MACD_MIN_DIFF_ENABLED = os.getenv("MACD_MIN_DIFF_ENABLED", "true").lower() == "true"

EMA_FAST = int(os.getenv("EMA_FAST", 9))
EMA_SLOW = int(os.getenv("EMA_SLOW", 21))
EMA_MIN_DIFF = int(os.getenv("EMA_MIN_DIFF", 12))
EMA_MIN_DIFF_ENABLED = os.getenv("EMA_MIN_DIFF_ENABLED", "true").lower() == "true"

# Stochastic Oscillator Configuration
STOCH_K_PERIOD = int(os.getenv("STOCH_K_PERIOD", 14))  # %K period
STOCH_D_PERIOD = int(os.getenv("STOCH_D_PERIOD", 3))   # %D period (smoothing)
STOCH_OVERSOLD = float(os.getenv("STOCH_OVERSOLD", 20))
STOCH_OVERBOUGHT = float(os.getenv("STOCH_OVERBOUGHT", 80))
STOCH_ENABLED = os.getenv("STOCH_ENABLED", "true").lower() == "true"

DB_ENABLED = os.getenv("DB_ENABLED", "false").lower() == "true"
DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    if DB_ENABLED else None
)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# how many candles to use when computing ATR
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
# how many ATRs away to place your stop
ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", 1.2))
# how many ATRs away to place your take profit
ATR_TP_MULTIPLIER = float(os.getenv("ATR_TP_MULTIPLIER", 2.4))

# Volume confirmation
VOLUME_CONFIRMATION_ENABLED = os.getenv("VOLUME_CONFIRMATION_ENABLED", "true").lower() == "true"
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "0.8"))

# Timeframe-specific cooldown periods (in minutes)
TIMEFRAME_COOLDOWNS = {
    "1m": 5,     # 5 minutes cooldown for 1m timeframe
    "5m": 15,    # 15 minutes cooldown for 5m timeframe
    "15m": 30,   # 30 minutes cooldown for 15m timeframe
    "1h": 120,   # 2 hours cooldown for 1h timeframe
    "4h": 480,   # 8 hours cooldown for 4h timeframe
    "1d": 1440,  # 24 hours cooldown for 1d timeframe
}

# Timeframe-specific volume ratios
TIMEFRAME_VOLUME_RATIOS = {
    "1m": 0.5,   # Lower volume threshold for 1m (more noise)
    "5m": 0.6,   # Slightly higher for 5m
    "15m": 0.8,  # Standard volume for 15m
    "1h": 1.0,   # Higher volume requirement for 1h
    "4h": 1.2,   # Even higher for 4h
    "1d": 1.5,   # Highest volume requirement for daily
}

# Timeframe-specific ADX thresholds for trend detection
TIMEFRAME_ADX_THRESHOLDS = {
    "1m": 35,    # Higher threshold for 1m (more noise, need stronger trends)
    "5m": 32,    # Slightly lower for 5m
    "15m": 28,   # Standard threshold for 15m (current default)
    "1h": 25,    # Lower for 1h (trends develop more clearly)
    "4h": 22,    # Even lower for 4h (strong trends easier to detect)
    "1d": 20,    # Lowest for daily (clear trend signals)
}

# Time-based filtering
TIME_FILTER_ENABLED = os.getenv("TIME_FILTER_ENABLED", "true").lower() == "true"
TIME_FILTER_TIMEZONE = os.getenv("TIME_FILTER_TIMEZONE", "Europe/Paris")  # CEST timezone
AVOID_HOURS_START = int(os.getenv("AVOID_HOURS_START", "0"))
AVOID_HOURS_END = int(os.getenv("AVOID_HOURS_END", "7"))

# Market summary configuration
MARKET_SUMMARY_ENABLED = os.getenv("MARKET_SUMMARY_ENABLED", "true").lower() == "true"

def tf_to_minutes(tf: str) -> int:
    m = re.match(r"(\d+)([mh])", tf)
    if not m:
        raise ValueError(f"Invalid timeframe: {tf}")
    v, u = int(m.group(1)), m.group(2)
    return v * (60 if u == "h" else 1)

HIGHER_TF_MAP = {
    "1m":  "5m",
    "5m":  "15m",
    "15m": "1h",
    "1h":  "4h",
    "4h":  "1d",
    "1d":  "1w"
}

# Helper functions for timeframe-specific configurations
def get_min_score_for_timeframe(timeframe: str) -> int:
    """Get minimum score requirement for specific timeframe."""
    return TIMEFRAME_MIN_SCORES.get(timeframe, MIN_SCORE_DEFAULT)

def get_cooldown_for_timeframe(timeframe: str) -> int:
    """Get cooldown period in minutes for specific timeframe."""
    return TIMEFRAME_COOLDOWNS.get(timeframe, tf_to_minutes(timeframe) * 4 + 1)

def get_volume_ratio_for_timeframe(timeframe: str) -> float:
    """Get volume ratio requirement for specific timeframe."""
    return TIMEFRAME_VOLUME_RATIOS.get(timeframe, MIN_VOLUME_RATIO)

def get_adx_threshold_for_timeframe(timeframe: str) -> int:
    """Get ADX threshold for trend detection for specific timeframe."""
    return TIMEFRAME_ADX_THRESHOLDS.get(timeframe, ADX_THRESHOLD)
