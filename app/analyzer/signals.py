import datetime
import logging

import ccxt
import pandas as pd
from ta.volatility import AverageTrueRange

from app.config import (
    RSI_PERIOD, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_FAST, EMA_SLOW,
    ATR_PERIOD, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER
)

logger = logging.getLogger(__name__)
exchange = ccxt.binance()


def analyze_market(pairs, timeframe):
    signals = []

    for pair in pairs:
        # 1) Fetch and frame your OHLCV
        candles = exchange.fetch_ohlcv(pair, timeframe)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # 2) Indicators
        rsi = df["close"].pct_change().rolling(RSI_PERIOD).mean().iloc[-1] * 100
        ema_fast = df["close"].ewm(span=EMA_FAST).mean().iloc[-1]
        ema_slow = df["close"].ewm(span=EMA_SLOW).mean().iloc[-1]
        macd_series = df["close"].ewm(span=MACD_FAST).mean() - df["close"].ewm(span=MACD_SLOW).mean()
        signal_line = macd_series.ewm(span=MACD_SIGNAL).mean().iloc[-1]
        macd = macd_series.iloc[-1]
        price = df["close"].iloc[-1]

        # 3) ATR for SL/TP
        atr = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=ATR_PERIOD
        ).average_true_range().iloc[-1]

        # 4) Signal decision
        side = "NONE"
        if rsi < RSI_OVERSOLD and ema_fast > ema_slow and macd > signal_line:
            side = "LONG"
        elif ema_fast < ema_slow and macd < signal_line:
            side = "SHORT"

        # 5) Log & build SL/TP only if we have a directional signal
        if side != "NONE":
            sl = price - atr * ATR_SL_MULTIPLIER if side == "LONG" else price + atr * ATR_SL_MULTIPLIER
            tp = price + atr * ATR_TP_MULTIPLIER if side == "LONG" else price - atr * ATR_TP_MULTIPLIER

            logger.info(
                f"{timeframe} | {pair} | side={side} | "
                f"RSI={rsi:.2f} | MACD={macd:.2f}/{signal_line:.2f} | "
                f"EMA={ema_fast:.2f}/{ema_slow:.2f} | price={price:.2f} | "
                f"ATR={atr:.2f} | SL={sl:.2f} | TP={tp:.2f}"
            )

            signals.append({
                "pair": pair,
                "timeframe": timeframe,
                "side": side,
                "price": price,
                "stop_loss": sl,
                "take_profit": tp,
                "timestamp": datetime.datetime.now(datetime.UTC)
            })
        else:
            logger.info(
                f"{timeframe} | {pair} | side={side} | "
                f"RSI={rsi:.2f} | MACD={macd:.2f}/{signal_line:.2f} | "
                f"EMA={ema_fast:.2f}/{ema_slow:.2f} | price={price:.2f}"
            )

    return signals
