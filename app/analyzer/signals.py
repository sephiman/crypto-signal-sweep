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
    SEND_UNCONFIRMED, ADX_PERIOD, ADX_THRESHOLD, RSI_MOMENTUM, ADX_RSI_MODE, MACD_MIN_DIFF_ENABLED,
    EMA_MIN_DIFF_ENABLED, DYNAMIC_SCORE_ENABLED, MIN_SCORE_RANGING, MIN_SCORE_DEFAULT, MIN_ATR_RATIO,
    MIN_SCORE_TRENDING, TIME_FILTER_ENABLED, AVOID_HOURS_START, AVOID_HOURS_END, MIN_VOLUME_RATIO,
    VOLUME_CONFIRMATION_ENABLED
)

logger = logging.getLogger(__name__)
exchange = ccxt.binance()


def analyze_market(pairs, timeframe):
    """
    Enhanced market analysis with improved filtering and scoring
    """
    # Add time filter check
    if TIME_FILTER_ENABLED and not _is_valid_trading_time():
        logger.info(f"Skipping analysis - outside valid trading hours")
        return []

    df = _fetch_ohlcv_df(pairs, timeframe)
    signals = []

    for pair in pairs:
        try:
            data = df[pair]
            if len(data) < 50:  # Need sufficient data
                continue

            price = _get_last_price(pair)

            # Volume confirmation check
            if VOLUME_CONFIRMATION_ENABLED:
                if not _check_volume_confirmation(data):
                    logger.debug(f"Skipping {pair} - insufficient volume")
                    continue

            # Calculate indicators
            rsi = RSIIndicator(data['close'], window=RSI_PERIOD).rsi().iloc[-1]
            macd_obj = MACD(
                close=data['close'],
                window_slow=MACD_SLOW,
                window_fast=MACD_FAST,
                window_sign=MACD_SIGNAL
            )
            macd = macd_obj.macd().iloc[-1]
            signal_line = macd_obj.macd_signal().iloc[-1]
            diff = macd - signal_line

            # Enhanced MACD momentum check
            if MACD_MIN_DIFF_ENABLED:
                momentum_ok_long = (macd > signal_line) and (diff >= MACD_MIN_DIFF)
                momentum_ok_short = (macd < signal_line) and (diff <= -MACD_MIN_DIFF)
            else:
                momentum_ok_long = macd > signal_line
                momentum_ok_short = macd < signal_line

            # Enhanced EMA check with minimum separation
            ema_fast = data['close'].ewm(span=EMA_FAST).mean().iloc[-1]
            ema_slow = data['close'].ewm(span=EMA_SLOW).mean().iloc[-1]

            # Calculate ATR for dynamic EMA separation
            atr = AverageTrueRange(
                high=data['high'], low=data['low'], close=data['close'], window=ATR_PERIOD
            ).average_true_range().iloc[-1]

            min_ema_separation = atr * 0.5  # Dynamic minimum separation

            if EMA_MIN_DIFF_ENABLED:
                ema_separation = abs(ema_fast - ema_slow)
                ema_ok_long = (ema_fast > ema_slow) and (ema_separation >= min_ema_separation)
                ema_ok_short = (ema_fast < ema_slow) and (ema_separation >= min_ema_separation)
            else:
                ema_ok_long = ema_fast > ema_slow
                ema_ok_short = ema_fast < ema_slow

            # Enhanced ADX-based regime detection
            from ta.trend import ADXIndicator
            adx = ADXIndicator(
                high=data['high'], low=data['low'], close=data['close'], window=ADX_PERIOD
            ).adx().iloc[-1]

            # Stricter RSI regime logic
            if ADX_RSI_MODE == "rsi":
                rsi_ok_long = rsi < RSI_OVERSOLD  # Now 25 instead of 30
                rsi_ok_short = rsi > RSI_OVERBOUGHT  # Now 75 instead of 70
            else:
                is_trending = adx >= ADX_THRESHOLD  # Now 28 instead of 25

                if is_trending:
                    # In trending markets, only take extreme RSI signals
                    rsi_ok_long = rsi < 35 and rsi > RSI_MOMENTUM  # More selective
                    rsi_ok_short = rsi > 65 and rsi < RSI_MOMENTUM  # More selective
                else:
                    # In ranging markets, use extreme RSI
                    rsi_ok_long = rsi < RSI_OVERSOLD
                    rsi_ok_short = rsi > RSI_OVERBOUGHT

            # Existing trend filter logic
            if USE_TREND_FILTER:
                sma = data['close'].rolling(window=TREND_MA_PERIOD).mean()
                recent_closes = data['close'].iloc[-REQUIRED_MA_BARS:]
                recent_sma = sma.iloc[-REQUIRED_MA_BARS:]
                trend_ok_long = (recent_closes > recent_sma).all()
                trend_ok_short = (recent_closes < recent_sma).all()
            else:
                trend_ok_long = trend_ok_short = True

            # Enhanced higher timeframe confirmation
            if USE_HIGHER_TF_CONFIRM:
                higher_tf = HIGHER_TF_MAP.get(timeframe)
                if higher_tf:
                    confirm_long, confirm_short = _get_htf_confirmation(pair, higher_tf)
                else:
                    confirm_long = confirm_short = True
            else:
                confirm_long = confirm_short = True

            # Enhanced scoring system with stricter requirements
            long_gates = [
                rsi_ok_long,
                macd > signal_line,
                momentum_ok_long,
                ema_ok_long,
                trend_ok_long,
                confirm_long or SEND_UNCONFIRMED
            ]

            short_gates = [
                rsi_ok_short,
                macd < signal_line,
                momentum_ok_short,
                ema_ok_short,
                trend_ok_short,
                confirm_short or SEND_UNCONFIRMED
            ]

            long_score = sum(long_gates)
            short_score = sum(short_gates)

            # Enhanced minimum score logic
            min_score = _enhanced_min_score(is_trending if 'is_trending' in locals() else adx >= ADX_THRESHOLD)

            side = "NONE"

            # For LONG: require RSI extreme + strong momentum + trend alignment
            if (long_score >= min_score and
                    rsi_ok_long and momentum_ok_long and ema_ok_long and
                    (not USE_HIGHER_TF_CONFIRM or confirm_long)):
                side = "LONG"

            # For SHORT: require RSI extreme + strong momentum + trend alignment
            elif (short_score >= min_score and
                  rsi_ok_short and momentum_ok_short and ema_ok_short and
                  (not USE_HIGHER_TF_CONFIRM or confirm_short)):
                side = "SHORT"

            # ATR volatility filter
            atr_pct = atr / price
            volume_ratio = _get_volume_ratio(data)

            # Single comprehensive log line with all metrics
            logger.info(
                f"{timeframe} | {pair} | "
                f"RSI:{rsi:.1f} ADX:{adx:.1f} MACD:{diff:.3f} "
                f"EMA:{ema_fast:.1f}/{ema_slow:.1f} ATR:{atr_pct:.4%} Vol:{volume_ratio:.1f}x | "
                f"Regime:{'trending' if adx >= ADX_THRESHOLD else 'ranging'} | "
                f"Gates L:{long_score}/S:{short_score} Min:{min_score}"
            )

            if atr_pct < MIN_ATR_RATIO:
                logger.info(f"❌ {timeframe} | {pair} | SKIPPED: ATR too low ({atr_pct:.4%} < {MIN_ATR_RATIO:.4%})")
                continue

            if side != "NONE":
                # Improved SL/TP calculation
                sl, tp = _calculate_enhanced_sl_tp(price, atr, side, pair)

                # Ensure minimum 2:1 risk-reward ratio
                risk = abs(price - sl)
                reward = abs(tp - price)

                if reward / risk < 2.0:
                    # Adjust TP to maintain 2:1 ratio
                    if side == "LONG":
                        tp = price + (risk * 2.0)
                    else:
                        tp = price - (risk * 2.0)

                logger.info(
                    f"{timeframe} | {pair} | {side} Signal Generated\n"
                    f"  Price: {price:.4f} | SL: {sl:.4f} | TP: {tp:.4f}\n"
                    f"  RSI: {rsi:.1f} | ADX: {adx:.1f} | MACD: {diff:.3f}\n"
                    f"  Score: {long_score if side == 'LONG' else short_score}/{min_score}\n"
                    f"  RR Ratio: {reward / risk:.2f}:1"
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
                    "confirmed": ((trend_ok_long if side == "LONG" else trend_ok_short) and
                                  (confirm_long if side == "LONG" else confirm_short)),
                    "score": long_score if side == "LONG" else short_score,
                    "required_score": min_score,
                    "rsi_ok": rsi_ok_long if side == "LONG" else rsi_ok_short,
                    "ema_ok": ema_ok_long if side == "LONG" else ema_ok_short,
                    "macd_ok": (macd > signal_line) if side == "LONG" else (macd < signal_line),
                    "macd_momentum_ok": momentum_ok_long if side == "LONG" else momentum_ok_short,
                    "rsi": rsi,
                    "adx": adx,
                    "macd": macd,
                    "macd_signal": signal_line,
                    "macd_diff": diff,
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "ema_diff": abs(ema_fast - ema_slow),
                    "atr": atr,
                    "atr_pct": atr / price,
                    "regime": "momentum" if adx >= ADX_THRESHOLD else "mean-reversion",
                    "htf_used": USE_HIGHER_TF_CONFIRM,
                    "volume_ratio": _get_volume_ratio(data),  # New field
                    "confidence": "HIGH" if long_score >= min_score + 1 or short_score >= min_score + 1 else "MEDIUM"
                    # New field
                })

        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}")
            continue

    return signals


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


def _dynamic_min_score(adx_value: float) -> int:
    """
    Decide the min score depending on the ADX value.
    If DYNAMIC_SCORE_ENABLED is enabled and the market is not in a strong trend, it allows lower score
    """
    if DYNAMIC_SCORE_ENABLED:
        return MIN_SCORE_RANGING if adx_value < ADX_THRESHOLD else MIN_SCORE_DEFAULT
    return MIN_SCORE_DEFAULT


def _is_valid_trading_time():
    """Check if current time is within valid trading hours"""
    if not TIME_FILTER_ENABLED:
        return True

    current_hour = datetime.datetime.utcnow().hour
    return not (AVOID_HOURS_START <= current_hour < AVOID_HOURS_END)


def _check_volume_confirmation(data):
    """Check if current volume is above average"""
    if len(data) < 20:
        return True  # Not enough data, allow signal

    current_volume = data['volume'].iloc[-1]
    avg_volume = data['volume'].iloc[-20:-1].mean()  # Exclude current candle

    if avg_volume == 0:
        return True

    volume_ratio = current_volume / avg_volume
    return volume_ratio >= MIN_VOLUME_RATIO


def _get_volume_ratio(data):
    """Calculate volume ratio for database storage"""
    if len(data) < 20:
        return 1.0

    current_volume = data['volume'].iloc[-1]
    avg_volume = data['volume'].iloc[-20:-1].mean()

    if avg_volume == 0:
        return 1.0

    return current_volume / avg_volume


def _get_htf_confirmation(pair, higher_tf):
    """Enhanced higher timeframe confirmation"""
    try:
        hdf = _fetch_ohlcv_df([pair], higher_tf)[pair]
        if len(hdf) < 30:
            return True, True

        ht_rsi = RSIIndicator(hdf['close'], window=RSI_PERIOD).rsi().iloc[-1]
        ht_macd_obj = MACD(
            close=hdf['close'],
            window_slow=MACD_SLOW,
            window_fast=MACD_FAST,
            window_sign=MACD_SIGNAL
        )
        ht_macd = ht_macd_obj.macd().iloc[-1]
        ht_signal = ht_macd_obj.macd_signal().iloc[-1]

        # Stricter HTF confirmation
        confirm_long = ht_rsi > 45 and ht_macd > ht_signal and (ht_macd - ht_signal) > 0.5
        confirm_short = ht_rsi < 55 and ht_macd < ht_signal and (ht_signal - ht_macd) > 0.5

        return confirm_long, confirm_short
    except Exception:
        return True, True  # Default to allowing signals if HTF fails


def _enhanced_min_score(is_trending):
    """Enhanced dynamic scoring with stricter requirements"""
    if DYNAMIC_SCORE_ENABLED:
        if is_trending:
            return MIN_SCORE_TRENDING
        else:
            return MIN_SCORE_RANGING
    return MIN_SCORE_DEFAULT


def _calculate_enhanced_sl_tp(price, atr, side, pair):
    """Enhanced SL/TP calculation with better risk-reward ratios"""
    sl_mult = ATR_SL_MULTIPLIER
    tp_mult = ATR_TP_MULTIPLIER

    if side == "LONG":
        sl = price - (atr * sl_mult)
        tp = price + (atr * tp_mult)
    else:
        sl = price + (atr * sl_mult)
        tp = price - (atr * tp_mult)

    return sl, tp
