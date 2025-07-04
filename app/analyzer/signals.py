import datetime
import logging

import ccxt
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

from app.config import (
    USE_HIGHER_TF_CONFIRM, HIGHER_TF_MAP,
    USE_TREND_FILTER, TREND_MA_PERIOD, REQUIRED_MA_BARS,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, MACD_MIN_DIFF,
    EMA_FAST, EMA_SLOW,
    ATR_PERIOD, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    SEND_UNCONFIRMED, ADX_PERIOD, ADX_THRESHOLD, RSI_MOMENTUM, EMA_MIN_DIFF, ADX_RSI_MODE, MACD_MIN_DIFF_ENABLED,
    EMA_MIN_DIFF_ENABLED
)

logger = logging.getLogger(__name__)
exchange = ccxt.binance()


def analyze_market(pairs, timeframe):
    """
     Analyze market for given pairs and timeframe, applying:

       1. ADX‐based regime detection:
          • If ADX ≥ ADX_THRESHOLD → momentum regime
            – use RSI > RSI_MOMENTUM / RSI < (100 - RSI_MOMENTUM) for entries
          • Else → mean‐reversion regime
            – use RSI < RSI_OVERSOLD / RSI > RSI_OVERBOUGHT for entries

       2. RSI (window=RSI_PERIOD) with regime‐aware gates

       3. MACD momentum confirmation:
          • True MACD‐signal cross on the most recent closed candle
          • Histogram difference ≥ MACD_MIN_DIFF

       4. EMA trend filter:
          • Require EMA_FAST and EMA_SLOW to be separated by ≥ EMA_MIN_DIFF

       5. Optional SMA trend filter (last REQUIRED_MA_BARS above/below TREND_MA_PERIOD‐SMA)

       6. Optional higher‐timeframe confirmation (using HIGHER_TF_MAP)

       7. ATR‐based stop‐loss and take‐profit:
          • SL = entry ± ATR * ATR_SL_MULTIPLIER
          • TP = entry ± ATR * ATR_TP_MULTIPLIER

     Returns a list of signal dicts (pair, timeframe, side, entry price, SL, TP, timestamp, etc.).
     """
    df = _fetch_ohlcv_df(pairs, timeframe)
    signals = []

    for pair in pairs:
        data = df[pair]
        price = _get_last_price(pair)

        # 1) Primary indicators
        rsi = RSIIndicator(data['close'], window=RSI_PERIOD).rsi().iloc[-1]
        macd_obj = MACD(
            close=data['close'],
            window_slow=MACD_SLOW,
            window_fast=MACD_FAST,
            window_sign=MACD_SIGNAL
        )
        macd = macd_obj.macd().iloc[-1]
        signal_line = macd_obj.macd_signal().iloc[-1]

        # The cross of signals may have happened before
        diff = macd - signal_line

        if MACD_MIN_DIFF_ENABLED:
            momentum_ok_long = (macd > signal_line) and (diff >= MACD_MIN_DIFF)
            momentum_ok_short = (macd < signal_line) and (diff <= -MACD_MIN_DIFF)
        else:
            momentum_ok_long = macd > signal_line
            momentum_ok_short = macd < signal_line

        ema_fast = data['close'].ewm(span=EMA_FAST).mean().iloc[-1]
        ema_slow = data['close'].ewm(span=EMA_SLOW).mean().iloc[-1]

        if EMA_MIN_DIFF_ENABLED:
            ema_ok_long = (ema_fast - ema_slow) >= EMA_MIN_DIFF
            ema_ok_short = (ema_slow - ema_fast) >= EMA_MIN_DIFF
        else:
            ema_ok_long = ema_fast > ema_slow
            ema_ok_short = ema_fast < ema_slow

        # 2) ADX‐based regime switch
        from ta.trend import ADXIndicator
        adx = ADXIndicator(
            high=data['high'], low=data['low'], close=data['close'], window=ADX_PERIOD
        ).adx().iloc[-1]
        if ADX_RSI_MODE == "rsi":
            rsi_ok_long = rsi < RSI_OVERSOLD
            rsi_ok_short = rsi > RSI_OVERBOUGHT
        else:
            is_trending = adx >= ADX_THRESHOLD

            # Regime‐aware RSI gates
            if is_trending:
                # momentum breakout regime
                rsi_ok_long = rsi > RSI_MOMENTUM
                rsi_ok_short = rsi < RSI_MOMENTUM
            else:
                # range/mean‐reversion regime
                rsi_ok_long = rsi < RSI_OVERSOLD
                rsi_ok_short = rsi > RSI_OVERBOUGHT

        # 3) Optional SMA trend filter
        if USE_TREND_FILTER:
            sma = data['close'].rolling(window=TREND_MA_PERIOD).mean()
            recent_closes = data['close'].iloc[-REQUIRED_MA_BARS:]
            recent_sma = sma.iloc[-REQUIRED_MA_BARS:]
            trend_ok_long = (recent_closes > recent_sma).all()
            trend_ok_short = (recent_closes < recent_sma).all()
        else:
            trend_ok_long = trend_ok_short = True

        # 4) Higher‐timeframe confirmation
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
                confirm_short = ht_rsi > RSI_OVERBOUGHT and ht_macd < ht_signal
            else:
                confirm_long = confirm_short = True
        else:
            confirm_long = confirm_short = True

        # 5) Decide side (now passing in rsi_ok flags)
        side = _determine_side(
            rsi_ok_long, rsi_ok_short,
            confirm_long, confirm_short,
            ema_ok_long, ema_ok_short,
            macd, signal_line,
            trend_ok_long, trend_ok_short,
            momentum_ok_long, momentum_ok_short
        )

        # 6) ATR‐based SL/TP
        atr = AverageTrueRange(
            high=data['high'], low=data['low'], close=data['close'], window=ATR_PERIOD
        ).average_true_range().iloc[-1]

        if side != "NONE":
            sl = price - atr * ATR_SL_MULTIPLIER if side == "LONG" else price + atr * ATR_SL_MULTIPLIER
            tp = price + atr * ATR_TP_MULTIPLIER if side == "LONG" else price - atr * ATR_TP_MULTIPLIER

            logger.info(
                f"{timeframe} | {pair} | side={side} | "
                f"RSI={rsi:.2f} | MACD={macd:.2f}/{signal_line:.2f} (Δ={diff:.2f}) | "
                f"EMA={ema_fast:.2f}/{ema_slow:.2f} | ADX={adx:.2f} | price={price:.2f} | "
                f"ATR={atr:.2f} | SL={sl:.2f} | TP={tp:.2f} | "
                f"momentum_ok={momentum_ok_long if side == 'LONG' else momentum_ok_short} | "
                f"trend_ok={trend_ok_long if side == 'LONG' else trend_ok_short} | "
                f"higher_tf_ok={confirm_long if side == 'LONG' else confirm_short}"
            )

            signals.append({
                "pair": pair,
                "timeframe": timeframe,
                "side": side,
                "price": price,
                "stop_loss": sl,
                "take_profit": tp,
                "timestamp": datetime.datetime.now(datetime.UTC),
                "momentum_ok": momentum_ok_long if side == "LONG" else momentum_ok_short,
                "trend_confirmed": trend_ok_long if side == "LONG" else trend_ok_short,
                "higher_tf_confirmed": confirm_long if side == "LONG" else confirm_short,
                "unconfirmed": (not ((trend_ok_long if side == "LONG" else trend_ok_short) and (
                    confirm_long if side == "LONG" else confirm_short)))
            })
        else:
            logger.info(
                f"{timeframe} | {pair} | side={side} | "
                f"RSI={rsi:.2f} | MACD={macd:.2f}/{signal_line:.2f} | ADX={adx:.2f} | "
                f"EMA={ema_fast:.2f}/{ema_slow:.2f} | price={price:.2f} | "
                f"momentum_ok(L/S)={momentum_ok_long}/{momentum_ok_short} | "
                f"trend_ok(L/S)={trend_ok_long}/{trend_ok_short} | "
                f"higher_tf_ok(L/S)={confirm_long}/{confirm_short}"
            )

    return signals


def _determine_side(
        rsi_ok_long, rsi_ok_short,
        confirm_long, confirm_short,
        ema_ok_long, ema_ok_short,
        macd, signal_line,
        trend_ok_long, trend_ok_short,
        momentum_ok_long, momentum_ok_short
):
    """
    Decide LONG/SHORT/NONE based on:
     - Regime-aware RSI gate (rsi_ok_*)
     - MACD crossover & minimum-diff (momentum_ok_*)
     - EMA trend (fast vs. slow)
     - Optional SMA & higher-TF confirmations
    """
    # LONG conditions
    if (
            rsi_ok_long and  # RSI in the right regime zone
            macd > signal_line and  # MACD above its signal line
            momentum_ok_long and  # macd - signal >= MACD_MIN_DIFF
            ema_ok_long and  # up-trend on EMAs
            (trend_ok_long and confirm_long  # all confirmations passed
             or SEND_UNCONFIRMED)  # or send it anyway if unconfirmed
    ):
        return "LONG"

    # SHORT conditions
    if (
            rsi_ok_short and  # RSI in the right regime zone
            macd < signal_line and  # MACD below its signal line
            momentum_ok_short and  # signal - macd >= MACD_MIN_DIFF
            ema_ok_short and  # down-trend on EMAs
            (trend_ok_short and confirm_short  # all confirmations passed
             or SEND_UNCONFIRMED)  # or send anyway if unconfirmed
    ):
        return "SHORT"

    return "NONE"


def _get_last_price(pair):
    ticker = exchange.fetch_ticker(pair)
    return float(ticker["last"])


def _fetch_ohlcv_df(pairs, timeframe):
    """
    Fetches OHLCV for each pair, drops the _incomplete_ bar,
    and returns a dict of DataFrames of only closed candles.
    """
    result = {}
    for pair in pairs:
        candles = exchange.fetch_ohlcv(pair, timeframe)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        # drop the in-flight (incomplete) candle
        if len(df) > 1:
            df = df.iloc[:-1]
        result[pair] = df
    return result
