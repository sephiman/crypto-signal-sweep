import os
import re

PAIRS = os.getenv("PAIRS", "BTC/USDT").split(",")
TIMEFRAMES = os.getenv("TIMEFRAMES", "15m").split(",")
RUN_AT_START = os.getenv("RUN_AT_START", "false").lower() == "true"
USE_HIGHER_TF_CONFIRM = os.getenv("USE_HIGHER_TF_CONFIRM", "true").lower() == "true"
HIGHER_TF = os.getenv("HIGHER_TF", "1h")
USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "true").lower() == "true"
TREND_MA_PERIOD = int(os.getenv("TREND_MA_PERIOD", 21))
REQUIRED_MA_BARS = int(os.getenv("REQUIRED_MA_BARS", 2))
SEND_UNCONFIRMED = os.getenv("SEND_UNCONFIRMED", "false").lower() == "true"
DYNAMIC_SCORE_ENABLED = os.getenv("DYNAMIC_SCORE_ENABLED", "false").lower() == "true"
MIN_SCORE_DEFAULT = int(os.getenv("MIN_SCORE_DEFAULT", 6))
MIN_SCORE_RANGING = int(os.getenv("MIN_SCORE_RANGING", 4))
MIN_ATR_RATIO = float(os.getenv("MIN_ATR_RATIO", 0.0025))

ADX_PERIOD = int(os.getenv("ADX_PERIOD", 14))
ADX_THRESHOLD = int(os.getenv("ADX_THRESHOLD", 25))
ADX_RSI_MODE = os.getenv("ADX_RSI_MODE", "adx").lower()

RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 35))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", 65))
RSI_MOMENTUM = float(os.getenv("RSI_MOMENTUM", 50))

MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))
MACD_MIN_DIFF = float(os.getenv("MACD_MIN_DIFF", 0.5))
MACD_MIN_DIFF_ENABLED = os.getenv("MACD_MIN_DIFF_ENABLED", "true").lower() == "true"

EMA_FAST = int(os.getenv("EMA_FAST", 9))
EMA_SLOW = int(os.getenv("EMA_SLOW", 21))
EMA_MIN_DIFF = int(os.getenv("EMA_MIN_DIFF", 10))
EMA_MIN_DIFF_ENABLED = os.getenv("EMA_MIN_DIFF_ENABLED", "true").lower() == "true"

SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", 1.0))
TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", 2.0))

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
ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", 1.0))
# how many ATRs away to place your take profit
ATR_TP_MULTIPLIER = float(os.getenv("ATR_TP_MULTIPLIER", 2.0))


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
