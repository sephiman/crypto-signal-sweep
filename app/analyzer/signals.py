import datetime
import logging

import ccxt
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

from app.config import (
    USE_HIGHER_TF_CONFIRM, HIGHER_TF_MAP,
    USE_TREND_FILTER, TREND_MA_PERIOD,
    RSI_PERIOD, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_FAST, EMA_SLOW,
    ATR_PERIOD, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER
)

logger = logging.getLogger(__name__)
exchange = ccxt.binance()


def analyze_market(pairs, timeframe):
    """
    Analyze market for given pairs and timeframe, applying RSI, MACD, EMA, ATR,
    plus optional trend filter and higher-timeframe confirmation.
    """
    # Fetch main timeframe data
    df = _fetch_ohlcv_df(pairs, timeframe)
    signals = []

    for pair in pairs:
        data = df[pair]
        price = data['close'].iloc[-1]

        # Primary timeframe indicators
        rsi = RSIIndicator(data['close'], window=RSI_PERIOD).rsi().iloc[-1]
        macd_obj = MACD(
            close=data['close'],
            window_slow=MACD_SLOW,
            window_fast=MACD_FAST,
            window_sign=MACD_SIGNAL
        )
        macd = macd_obj.macd().iloc[-1]
        signal_line = macd_obj.macd_signal().iloc[-1]
        ema_fast = data['close'].ewm(span=EMA_FAST).mean().iloc[-1]
        ema_slow = data['close'].ewm(span=EMA_SLOW).mean().iloc[-1]

        # Trend filter
        if USE_TREND_FILTER:
            ma = data['close'].rolling(window=TREND_MA_PERIOD).mean().iloc[-1]
            # require price above MA for LONG, below for SHORT
            trend_ok_long = price > ma
            trend_ok_short = price < ma
        else:
            trend_ok_long = trend_ok_short = True

        # Higher timeframe confirmation
        if USE_HIGHER_TF_CONFIRM:
            higher_tf = HIGHER_TF_MAP.get(timeframe)
            if higher_tf:
                hdf = _fetch_ohlcv_df([pair], higher_tf)[pair]
                ht_rsi = RSIIndicator(hdf['close'], window=RSI_PERIOD).rsi().iloc[-1]
                ht_macd_obj = MACD(
                    close=hdf['close'],
                    window_slow=MACD_SLOW,
                    window_fast=MACD_FAST,
                    window_sign=MACD_SIGNAL
                )
                ht_macd = ht_macd_obj.macd().iloc[-1]
                ht_signal = ht_macd_obj.macd_signal().iloc[-1]
                confirm_long = ht_rsi < RSI_OVERSOLD and ht_macd > ht_signal
                confirm_short = ht_rsi > RSI_OVERSOLD and ht_macd < ht_signal
            else:
                confirm_long = confirm_short = True
        else:
            confirm_long = confirm_short = True

        # Determine side
        side = "NONE"
        if (rsi < RSI_OVERSOLD and macd > signal_line
                and ema_fast > ema_slow and trend_ok_long and confirm_long):
            side = "LONG"
        elif (rsi > RSI_OVERSOLD and macd < signal_line
              and ema_fast < ema_slow and trend_ok_short and confirm_short):
            side = "SHORT"

        # ATR-based SL/TP
        atr = AverageTrueRange(
            high=data['high'], low=data['low'], close=data['close'], window=ATR_PERIOD
        ).average_true_range().iloc[-1]
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


def _fetch_ohlcv_df(pairs, timeframe):
    """
    Fetches OHLCV for each pair and returns a dict of DataFrames.
    """
    result = {}
    for pair in pairs:
        candles = exchange.fetch_ohlcv(pair, timeframe)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        result[pair] = df
    return result
