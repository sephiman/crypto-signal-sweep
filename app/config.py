import os

PAIRS = os.getenv("PAIRS", "BTC/USDT").split(",")
TIMEFRAMES = os.getenv("TIMEFRAMES", "15m").split(",")
USE_HIGHER_TF_CONFIRM = os.getenv("USE_HIGHER_TF_CONFIRM", "false").lower() == "true"
HIGHER_TF = os.getenv("HIGHER_TF", "1h")

RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 30))
MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))
EMA_FAST = int(os.getenv("EMA_FAST", 9))
EMA_SLOW = int(os.getenv("EMA_SLOW", 21))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", 1.5))
TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", 3.0))

DB_ENABLED = os.getenv("DB_ENABLED", "false").lower() == "true"
DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    if DB_ENABLED else None
)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# how many candles to use when computing ATR
ATR_PERIOD         = int(os.getenv("ATR_PERIOD", 14))
# how many ATRs away to place your stop
ATR_SL_MULTIPLIER  = float(os.getenv("ATR_SL_MULTIPLIER", 1.0))
# how many ATRs away to place your take profit
ATR_TP_MULTIPLIER  = float(os.getenv("ATR_TP_MULTIPLIER", 2.0))