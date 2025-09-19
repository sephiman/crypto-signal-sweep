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
    MIN_SCORE_TRENDING, TIME_FILTER_ENABLED, TIME_FILTER_TIMEZONE, AVOID_HOURS_START, AVOID_HOURS_END, MIN_VOLUME_RATIO,
    VOLUME_CONFIRMATION_ENABLED, RSI_TRENDING_MODE, RSI_TRENDING_PULLBACK_LONG, RSI_TRENDING_PULLBACK_SHORT,
    RSI_TRENDING_OVERSOLD, RSI_TRENDING_OVERBOUGHT
)

logger = logging.getLogger(__name__)
exchange = ccxt.binance()

from app.db.tracker import save_market_analysis


def analyze_market(pairs, timeframe):
    """
    Market analysis with improved filtering and scoring
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
                logger.info(f"⏭️ SKIP | {timeframe} | {pair} | Reason:INSUFFICIENT_DATA (<50 candles)")
                continue

            price = _get_last_price(pair)

            # Calculate ALL indicators first (always)
            volume_ratio = _get_volume_ratio(data)

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

            ema_fast = data['close'].ewm(span=EMA_FAST).mean().iloc[-1]
            ema_slow = data['close'].ewm(span=EMA_SLOW).mean().iloc[-1]

            atr = AverageTrueRange(
                high=data['high'], low=data['low'], close=data['close'], window=ATR_PERIOD
            ).average_true_range().iloc[-1]
            atr_pct = atr / price

            from ta.trend import ADXIndicator
            adx = ADXIndicator(
                high=data['high'], low=data['low'], close=data['close'], window=ADX_PERIOD
            ).adx().iloc[-1]

            # Calculate filter conditions
            volume_pass = not VOLUME_CONFIRMATION_ENABLED or _check_volume_confirmation(data)
            atr_pass = atr_pct >= MIN_ATR_RATIO

            min_ema_separation = atr * 0.5

            # MACD momentum check
            if MACD_MIN_DIFF_ENABLED:
                momentum_ok_long = (macd > signal_line) and (diff >= MACD_MIN_DIFF)
                momentum_ok_short = (macd < signal_line) and (diff <= -MACD_MIN_DIFF)
            else:
                momentum_ok_long = macd > signal_line
                momentum_ok_short = macd < signal_line

            # EMA check
            if EMA_MIN_DIFF_ENABLED:
                ema_separation = abs(ema_fast - ema_slow)
                ema_ok_long = (ema_fast > ema_slow) and (ema_separation >= min_ema_separation)
                ema_ok_short = (ema_fast < ema_slow) and (ema_separation >= min_ema_separation)
            else:
                ema_ok_long = ema_fast > ema_slow
                ema_ok_short = ema_fast < ema_slow

            # RSI regime logic
            if ADX_RSI_MODE == "rsi":
                # Simple mode: always use standard oversold/overbought levels
                rsi_ok_long = rsi < RSI_OVERSOLD
                rsi_ok_short = rsi > RSI_OVERBOUGHT
            else:
                # ADX-based adaptive mode
                is_trending = adx >= ADX_THRESHOLD

                if is_trending:
                    # Market is trending - use trending-specific RSI strategy
                    if RSI_TRENDING_MODE == "pullback":
                        # Pullback mode: Look for mild retracements in strong trends
                        # Long: RSI pulls back but stays above support level
                        rsi_ok_long = RSI_TRENDING_PULLBACK_LONG < rsi < RSI_MOMENTUM
                        # Short: RSI pulls back but stays below resistance level
                        rsi_ok_short = RSI_MOMENTUM < rsi < RSI_TRENDING_PULLBACK_SHORT
                    else:
                        # Extreme mode (default): Require even more extreme levels in trends
                        # This filters out weak signals in strong trends
                        rsi_ok_long = rsi < RSI_TRENDING_OVERSOLD
                        rsi_ok_short = rsi > RSI_TRENDING_OVERBOUGHT
                else:
                    # Market is ranging - use standard oversold/overbought levels
                    rsi_ok_long = rsi < RSI_OVERSOLD
                    rsi_ok_short = rsi > RSI_OVERBOUGHT

            # Trend filter logic
            if USE_TREND_FILTER:
                sma = data['close'].rolling(window=TREND_MA_PERIOD).mean()
                recent_closes = data['close'].iloc[-REQUIRED_MA_BARS:]
                recent_sma = sma.iloc[-REQUIRED_MA_BARS:]
                trend_ok_long = (recent_closes > recent_sma).all()
                trend_ok_short = (recent_closes < recent_sma).all()
            else:
                trend_ok_long = trend_ok_short = True

            # Higher timeframe confirmation
            if USE_HIGHER_TF_CONFIRM:
                higher_tf = HIGHER_TF_MAP.get(timeframe)
                if higher_tf:
                    confirm_long, confirm_short = _get_htf_confirmation(pair, higher_tf)
                else:
                    confirm_long = confirm_short = True
            else:
                confirm_long = confirm_short = True

            # Scoring system
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

            min_score = _dynamic_min_score_trending(is_trending if 'is_trending' in locals() else adx >= ADX_THRESHOLD)

            # Determine signal side based on scoring system only
            if long_score >= min_score and long_score >= short_score:
                side = "LONG"
            elif short_score >= min_score:
                side = "SHORT"
            else:
                side = "NONE"
                # Provide detailed failure reason
                long_fails = []
                short_fails = []

                if not rsi_ok_long: long_fails.append(f"RSI({rsi:.1f})")
                if not (macd > signal_line): long_fails.append("MACD_DIR")
                if not momentum_ok_long: long_fails.append(f"MACD_MOM({diff:.6f})")
                if not ema_ok_long: long_fails.append("EMA")
                if not trend_ok_long: long_fails.append("TREND")
                if USE_HIGHER_TF_CONFIRM and not confirm_long: long_fails.append("HTF")

                if not rsi_ok_short: short_fails.append(f"RSI({rsi:.1f})")
                if not (macd < signal_line): short_fails.append("MACD_DIR")
                if not momentum_ok_short: short_fails.append(f"MACD_MOM({diff:.6f})")
                if not ema_ok_short: short_fails.append("EMA")
                if not trend_ok_short: short_fails.append("TREND")
                if USE_HIGHER_TF_CONFIRM and not confirm_short: short_fails.append("HTF")

                if long_score >= short_score:
                    none_reason = f"LONG_FAIL(S{long_score}<{min_score}:{','.join(long_fails)})"
                else:
                    none_reason = f"SHORT_FAIL(S{short_score}<{min_score}:{','.join(short_fails)})"

            # Determine final status and reason
            if not volume_pass:
                status = "⏭️ SKIP"
                result = f"LOW_VOL({volume_ratio:.1f}x<{MIN_VOLUME_RATIO}x)"
            elif not atr_pass:
                status = "⏭️ SKIP"
                result = f"LOW_ATR({atr_pct:.3%}<{MIN_ATR_RATIO:.3%})"
            elif side != "NONE":
                status = "✅ SIGNAL"
                result = side
            else:
                status = "⏭️ SKIP"
                result = none_reason if none_reason else "NO_SIGNAL"

            # All metrics
            logger.info(
                f"{status} | {timeframe} | {pair} | "
                f"Price:{price:.2f} RSI:{rsi:.1f} ADX:{adx:.1f} MACD:{diff:.4f} "
                f"EMA:{ema_fast:.2f}/{ema_slow:.2f} ATR:{atr_pct:.3%} VOL:{volume_ratio:.1f}x | "
                f"Regime:{'TREND' if adx >= ADX_THRESHOLD else 'RANGE'} "
                f"Score:L{long_score}/S{short_score}(min:{min_score}) | "
                f"Gates[L/S]: RSI:{int(rsi_ok_long)}/{int(rsi_ok_short)} "
                f"MACD:{int(momentum_ok_long)}/{int(momentum_ok_short)} "
                f"EMA:{int(ema_ok_long)}/{int(ema_ok_short)} "
                f"Trend:{int(trend_ok_long)}/{int(trend_ok_short)} "
                f"HTF:{int(confirm_long)}/{int(confirm_short)} | "
                f"Result:{result}"
            )

            # Save market analysis data to database (both signals and non-signals)
            analysis_data = {
                "pair": pair,
                "timeframe": timeframe,
                "timestamp": datetime.datetime.now(datetime.UTC),
                "price": price,
                "rsi": rsi,
                "adx": adx,
                "macd": macd,
                "macd_signal": signal_line,
                "macd_diff": diff,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "ema_diff": abs(ema_fast - ema_slow),
                "atr": atr,
                "atr_pct": atr_pct,
                "volume_ratio": volume_ratio,
                "rsi_ok_long": rsi_ok_long,
                "rsi_ok_short": rsi_ok_short,
                "macd_ok_long": macd > signal_line,
                "macd_ok_short": macd < signal_line,
                "momentum_ok_long": momentum_ok_long,
                "momentum_ok_short": momentum_ok_short,
                "ema_ok_long": ema_ok_long,
                "ema_ok_short": ema_ok_short,
                "trend_ok_long": trend_ok_long,
                "trend_ok_short": trend_ok_short,
                "htf_confirm_long": confirm_long,
                "htf_confirm_short": confirm_short,
                "volume_pass": volume_pass,
                "atr_pass": atr_pass,
                "time_pass": True,  # We already filtered for time above
                "long_score": long_score,
                "short_score": short_score,
                "min_score_required": min_score,
                "regime": "TREND" if adx >= ADX_THRESHOLD else "RANGE",
                "is_trending": adx >= ADX_THRESHOLD,
                "signal_generated": side != "NONE" and volume_pass and atr_pass,
                "signal_side": side if side != "NONE" and volume_pass and atr_pass else None,
                "skip_reason": result if not (side != "NONE" and volume_pass and atr_pass) else None
            }
            save_market_analysis(analysis_data)

            # Skip if filters don't pass
            if not atr_pass or not volume_pass or side == "NONE":
                continue

            # Generate signal details with dual TP
            sl, tp1, tp2 = _calculate_sl_tp(price, atr, side, pair)

            # Ensure minimum 2:1 risk-reward ratio for TP2
            risk = abs(price - sl)
            reward = abs(tp2 - price)

            if reward / risk < 2.0:
                if side == "LONG":
                    tp2 = price + (risk * 2.0)
                    # Recalculate TP1 as 50% of the new distance to TP2
                    tp_distance = tp2 - price
                    tp1 = price + (tp_distance * 0.5)
                else:
                    tp2 = price - (risk * 2.0)
                    # Recalculate TP1 as 50% of the new distance to TP2
                    tp_distance = price - tp2
                    tp1 = price - (tp_distance * 0.5)

            signals.append({
                "pair": pair,
                "timeframe": timeframe,
                "side": side,
                "price": price,
                "stop_loss": sl,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
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
                "atr_pct": atr_pct,
                "regime": "momentum" if adx >= ADX_THRESHOLD else "mean-reversion",
                "htf_used": USE_HIGHER_TF_CONFIRM,
                "volume_ratio": volume_ratio,
                "confidence": "HIGH" if long_score >= min_score + 1 or short_score >= min_score + 1 else "MEDIUM"
            })

        except Exception as e:
            logger.error(f"ERROR | {timeframe} | {pair} | Exception: {e}")
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
    """Check if current time is within valid trading hours (configured timezone)"""
    if not TIME_FILTER_ENABLED:
        return True

    import zoneinfo
    
    # Get current time in configured timezone
    try:
        target_tz = zoneinfo.ZoneInfo(TIME_FILTER_TIMEZONE)
        current_time = datetime.datetime.now(target_tz)
        current_hour = current_time.hour
        
        return not (AVOID_HOURS_START <= current_hour < AVOID_HOURS_END)
    except Exception as e:
        logger.warning(f"Invalid timezone {TIME_FILTER_TIMEZONE}, falling back to UTC: {e}")
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
    return volume_ratio >= (MIN_VOLUME_RATIO - 0.01)


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
    """Higher timeframe confirmation"""
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

        # HTF confirmation
        confirm_long = ht_rsi > 45 and ht_macd > ht_signal and (ht_macd - ht_signal) > 0.5
        confirm_short = ht_rsi < 55 and ht_macd < ht_signal and (ht_signal - ht_macd) > 0.5

        return confirm_long, confirm_short
    except Exception:
        return True, True  # Default to allowing signals if HTF fails


def _dynamic_min_score_trending(is_trending):
    """Dynamic scoring with stricter requirements"""
    if DYNAMIC_SCORE_ENABLED:
        if is_trending:
            return MIN_SCORE_TRENDING
        else:
            return MIN_SCORE_RANGING
    return MIN_SCORE_DEFAULT


def _calculate_sl_tp(price, atr, side, pair):
    """SL/TP calculation with dual take profit levels"""
    sl_mult = ATR_SL_MULTIPLIER
    tp_mult = ATR_TP_MULTIPLIER

    if side == "LONG":
        sl = price - (atr * sl_mult)
        tp2 = price + (atr * tp_mult)  # Full TP (original level)
        
        # Calculate TP1 at 50% of the distance to TP2 for partial profit
        tp_distance = tp2 - price
        tp1 = price + (tp_distance * 0.5)
        
    else:
        sl = price + (atr * sl_mult)
        tp2 = price - (atr * tp_mult)  # Full TP (original level)
        
        # Calculate TP1 at 50% of the distance to TP2 for partial profit
        tp_distance = price - tp2
        tp1 = price - (tp_distance * 0.5)

    return sl, tp1, tp2
