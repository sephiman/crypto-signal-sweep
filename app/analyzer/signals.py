import datetime
import logging
import uuid

import ccxt
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

from app.config import (
    USE_HIGHER_TF_CONFIRM, HIGHER_TF_MAP,
    USE_TREND_FILTER, TREND_MA_PERIOD, REQUIRED_MA_BARS,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, MACD_MIN_DIFF,
    EMA_FAST, EMA_SLOW,
    ATR_PERIOD, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    SEND_UNCONFIRMED, ADX_PERIOD, RSI_MOMENTUM, ADX_RSI_MODE, MACD_MIN_DIFF_ENABLED,
    EMA_MIN_DIFF_ENABLED, DYNAMIC_SCORE_ENABLED, MIN_SCORE_RANGING, MIN_ATR_RATIO,
    MIN_SCORE_TRENDING, TIME_FILTER_ENABLED, TIME_FILTER_TIMEZONE, AVOID_HOURS_START, AVOID_HOURS_END,
    VOLUME_CONFIRMATION_ENABLED, RSI_TRENDING_MODE, RSI_TRENDING_PULLBACK_LONG, RSI_TRENDING_PULLBACK_SHORT,
    RSI_TRENDING_OVERSOLD, RSI_TRENDING_OVERBOUGHT, STOCH_K_PERIOD, STOCH_D_PERIOD, STOCH_OVERSOLD, STOCH_OVERBOUGHT,
    STOCH_ENABLED, BB_PERIOD, BB_STD_DEV, BB_WIDTH_MIN, BB_ENABLED,
    get_min_score_for_timeframe, get_volume_ratio_for_timeframe, get_adx_threshold_for_timeframe
)

logger = logging.getLogger(__name__)

from app.db.tracker import save_market_analysis
from app.data_provider import get_current_price, fetch_ohlcv_df


class TechnicalIndicators:
    """Container for all calculated technical indicators"""
    def __init__(self, rsi, macd, signal_line, diff, ema_fast, ema_slow, atr, atr_pct, adx, stoch_k, stoch_d, volume_ratio, bb_width, bb_width_prev):
        self.rsi = rsi
        self.macd = macd
        self.signal_line = signal_line
        self.diff = diff
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr = atr
        self.atr_pct = atr_pct
        self.adx = adx
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.volume_ratio = volume_ratio
        self.bb_width = bb_width
        self.bb_width_prev = bb_width_prev


class MarketConditions:
    """Container for all market condition checks"""
    def __init__(self, rsi_ok_long, rsi_ok_short, momentum_ok_long, momentum_ok_short,
                 ema_ok_long, ema_ok_short, trend_ok_long, trend_ok_short,
                 stoch_ok_long, stoch_ok_short, confirm_long, confirm_short,
                 volume_pass, atr_pass, is_trending, bb_pass):
        self.rsi_ok_long = rsi_ok_long
        self.rsi_ok_short = rsi_ok_short
        self.momentum_ok_long = momentum_ok_long
        self.momentum_ok_short = momentum_ok_short
        self.ema_ok_long = ema_ok_long
        self.ema_ok_short = ema_ok_short
        self.trend_ok_long = trend_ok_long
        self.trend_ok_short = trend_ok_short
        self.stoch_ok_long = stoch_ok_long
        self.stoch_ok_short = stoch_ok_short
        self.confirm_long = confirm_long
        self.confirm_short = confirm_short
        self.volume_pass = volume_pass
        self.atr_pass = atr_pass
        self.is_trending = is_trending
        self.bb_pass = bb_pass


def analyze_market(pairs, timeframe):
    """
    Market analysis with improved filtering and scoring
    """
    df = fetch_ohlcv_df(pairs, timeframe)
    signals = []

    for pair in pairs:
        try:
            data = df[pair]
            if len(data) < 50:  # Need sufficient data
                logger.info(f"⏭️ SKIP | {timeframe} | {pair} | Reason:INSUFFICIENT_DATA (<50 candles)")
                continue

            price = get_current_price(pair)

            # Calculate ALL indicators first (always)
            indicators = _calculate_technical_indicators(data, price)

            # Calculate filter conditions and market conditions
            conditions = _calculate_market_conditions(data, indicators, timeframe, pair)

            # Scoring system
            long_score, short_score, min_score = _calculate_scores(indicators, conditions, timeframe)

            # Determine signal side and status
            side, status, result = _determine_signal_side(long_score, short_score, min_score, conditions, indicators, timeframe)

            # Log analysis and save to database
            _log_and_save_analysis(pair, timeframe, price, status, result, long_score, short_score,
                                 min_score, indicators, conditions, side)

            # Skip if filters don't pass or signal generation not allowed
            if not conditions.atr_pass or not conditions.volume_pass or side == "NONE" or status != "✅ SIGNAL":
                continue

            # Generate and add signal
            signal = _generate_signal_details(pair, timeframe, side, price, indicators, conditions,
                                            long_score, short_score, min_score)
            signals.append(signal)

        except Exception as e:
            logger.error(f"ERROR | {timeframe} | {pair} | Exception: {e}")
            continue

    return signals





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


def _get_dynamic_min_score_for_timeframe(timeframe, is_trending):
    """
    Get minimum score based on timeframe and market regime (trending vs ranging).
    Uses timeframe-specific base scores but applies dynamic adjustment for ranging markets.
    """
    base_score = get_min_score_for_timeframe(timeframe)

    if DYNAMIC_SCORE_ENABLED:
        if is_trending:
            # In trending markets, use the timeframe base score or trending minimum
            return max(base_score, MIN_SCORE_TRENDING)
        else:
            # In ranging markets, allow lower scores (more permissive)
            return min(base_score, MIN_SCORE_RANGING)

    return base_score


def _check_volume_confirmation(data, timeframe):
    """Check if current volume is above average"""
    if len(data) < 20:
        return True  # Not enough data, allow signal

    current_volume = data['volume'].iloc[-1]
    avg_volume = data['volume'].iloc[-20:-1].mean()  # Exclude current candle

    if avg_volume == 0:
        return True

    volume_ratio = current_volume / avg_volume
    required_ratio = get_volume_ratio_for_timeframe(timeframe)
    return volume_ratio >= (required_ratio - 0.01)


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
        hdf = fetch_ohlcv_df([pair], higher_tf)[pair]
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


def _calculate_technical_indicators(data, price):
    """Calculate all technical indicators"""
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

    # Stochastic Oscillator calculation
    if STOCH_ENABLED:
        stoch_obj = StochasticOscillator(
            high=data['high'], low=data['low'], close=data['close'],
            window=STOCH_K_PERIOD, smooth_window=STOCH_D_PERIOD
        )
        stoch_k = stoch_obj.stoch().iloc[-1]
        stoch_d = stoch_obj.stoch_signal().iloc[-1]
    else:
        stoch_k = stoch_d = 50  # Neutral values when disabled

    # Bollinger Bands Width calculation
    if BB_ENABLED and len(data) >= BB_PERIOD + 1:
        bb_obj = BollingerBands(
            close=data['close'],
            window=BB_PERIOD,
            window_dev=BB_STD_DEV
        )
        bb_upper = bb_obj.bollinger_hband()
        bb_lower = bb_obj.bollinger_lband()

        # Calculate BB width as percentage: (upper - lower) / price
        bb_width_series = (bb_upper - bb_lower) / data['close']
        bb_width = bb_width_series.iloc[-1]
        bb_width_prev = bb_width_series.iloc[-2]
    else:
        bb_width = bb_width_prev = 0.0  # Neutral values when disabled

    return TechnicalIndicators(rsi, macd, signal_line, diff, ema_fast, ema_slow,
                             atr, atr_pct, adx, stoch_k, stoch_d, volume_ratio, bb_width, bb_width_prev)


def _calculate_market_conditions(data, indicators, timeframe, pair):
    """Calculate all market conditions and filters"""
    volume_pass = not VOLUME_CONFIRMATION_ENABLED or _check_volume_confirmation(data, timeframe)
    atr_pass = indicators.atr_pct >= MIN_ATR_RATIO

    min_ema_separation = indicators.atr * 0.5

    # MACD momentum check
    if MACD_MIN_DIFF_ENABLED:
        momentum_ok_long = (indicators.macd > indicators.signal_line) and (indicators.diff >= MACD_MIN_DIFF)
        momentum_ok_short = (indicators.macd < indicators.signal_line) and (indicators.diff <= -MACD_MIN_DIFF)
    else:
        momentum_ok_long = indicators.macd > indicators.signal_line
        momentum_ok_short = indicators.macd < indicators.signal_line

    # EMA check
    if EMA_MIN_DIFF_ENABLED:
        ema_separation = abs(indicators.ema_fast - indicators.ema_slow)
        ema_ok_long = (indicators.ema_fast > indicators.ema_slow) and (ema_separation >= min_ema_separation)
        ema_ok_short = (indicators.ema_fast < indicators.ema_slow) and (ema_separation >= min_ema_separation)
    else:
        ema_ok_long = indicators.ema_fast > indicators.ema_slow
        ema_ok_short = indicators.ema_fast < indicators.ema_slow

    # RSI regime logic
    adx_threshold = get_adx_threshold_for_timeframe(timeframe)
    is_trending = indicators.adx >= adx_threshold
    if ADX_RSI_MODE == "rsi":
        # Simple mode: always use standard oversold/overbought levels
        rsi_ok_long = indicators.rsi < RSI_OVERSOLD
        rsi_ok_short = indicators.rsi > RSI_OVERBOUGHT
    else:
        # ADX-based adaptive mode
        if is_trending:
            # Market is trending - use trending-specific RSI strategy
            if RSI_TRENDING_MODE == "pullback":
                # Pullback mode: Look for mild retracements in strong trends
                rsi_ok_long = RSI_TRENDING_PULLBACK_LONG < indicators.rsi < RSI_MOMENTUM
                rsi_ok_short = RSI_MOMENTUM < indicators.rsi < RSI_TRENDING_PULLBACK_SHORT
            else:
                # Extreme mode (default): Require even more extreme levels in trends
                rsi_ok_long = indicators.rsi < RSI_TRENDING_OVERSOLD
                rsi_ok_short = indicators.rsi > RSI_TRENDING_OVERBOUGHT
        else:
            # Market is ranging - use standard oversold/overbought levels
            rsi_ok_long = indicators.rsi < RSI_OVERSOLD
            rsi_ok_short = indicators.rsi > RSI_OVERBOUGHT

    # Trend filter logic
    if USE_TREND_FILTER:
        sma = data['close'].rolling(window=TREND_MA_PERIOD).mean()
        recent_closes = data['close'].iloc[-REQUIRED_MA_BARS:]
        recent_sma = sma.iloc[-REQUIRED_MA_BARS:]
        trend_ok_long = (recent_closes > recent_sma).all()
        trend_ok_short = (recent_closes < recent_sma).all()
    else:
        trend_ok_long = trend_ok_short = True

    # Stochastic conditions
    if STOCH_ENABLED:
        stoch_ok_long = indicators.stoch_k < STOCH_OVERSOLD and indicators.stoch_d < STOCH_OVERSOLD
        stoch_ok_short = indicators.stoch_k > STOCH_OVERBOUGHT and indicators.stoch_d > STOCH_OVERBOUGHT
    else:
        stoch_ok_long = stoch_ok_short = True  # Allow signals when disabled

    # Higher timeframe confirmation
    if USE_HIGHER_TF_CONFIRM:
        higher_tf = HIGHER_TF_MAP.get(timeframe)
        if higher_tf:
            confirm_long, confirm_short = _get_htf_confirmation(pair, higher_tf)
        else:
            confirm_long = confirm_short = True
    else:
        confirm_long = confirm_short = True

    # Bollinger Bands Width filter
    if BB_ENABLED:
        # Check if BB width is expanding (current > previous) AND above minimum threshold
        bb_expanding = indicators.bb_width > indicators.bb_width_prev
        bb_above_min = indicators.bb_width >= BB_WIDTH_MIN
        bb_pass = bb_expanding and bb_above_min
    else:
        bb_pass = True  # Allow signals when disabled

    return MarketConditions(rsi_ok_long, rsi_ok_short, momentum_ok_long, momentum_ok_short,
                          ema_ok_long, ema_ok_short, trend_ok_long, trend_ok_short,
                          stoch_ok_long, stoch_ok_short, confirm_long, confirm_short,
                          volume_pass, atr_pass, is_trending, bb_pass)


def _calculate_scores(indicators, conditions, timeframe):
    """Calculate scoring system for long and short signals"""
    long_gates = [
        conditions.rsi_ok_long,
        indicators.macd > indicators.signal_line,
        conditions.momentum_ok_long,
        conditions.ema_ok_long,
        conditions.trend_ok_long,
        conditions.stoch_ok_long,
        conditions.confirm_long or SEND_UNCONFIRMED
    ]

    short_gates = [
        conditions.rsi_ok_short,
        indicators.macd < indicators.signal_line,
        conditions.momentum_ok_short,
        conditions.ema_ok_short,
        conditions.trend_ok_short,
        conditions.stoch_ok_short,
        conditions.confirm_short or SEND_UNCONFIRMED
    ]

    long_score = sum(long_gates)
    short_score = sum(short_gates)
    min_score = _get_dynamic_min_score_for_timeframe(timeframe, conditions.is_trending)

    return long_score, short_score, min_score


def _determine_signal_side(long_score, short_score, min_score, conditions, indicators, timeframe):
    """Determine signal side and status based on scores and conditions"""
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

        if not conditions.rsi_ok_long: long_fails.append(f"RSI({indicators.rsi:.1f})")
        if not (indicators.macd > indicators.signal_line): long_fails.append("MACD_DIR")
        if not conditions.momentum_ok_long: long_fails.append(f"MACD_MOM({indicators.diff:.6f})")
        if not conditions.ema_ok_long: long_fails.append("EMA")
        if not conditions.trend_ok_long: long_fails.append("TREND")
        if STOCH_ENABLED and not conditions.stoch_ok_long: long_fails.append(f"STOCH({indicators.stoch_k:.1f}/{indicators.stoch_d:.1f})")
        if USE_HIGHER_TF_CONFIRM and not conditions.confirm_long: long_fails.append("HTF")

        if not conditions.rsi_ok_short: short_fails.append(f"RSI({indicators.rsi:.1f})")
        if not (indicators.macd < indicators.signal_line): short_fails.append("MACD_DIR")
        if not conditions.momentum_ok_short: short_fails.append(f"MACD_MOM({indicators.diff:.6f})")
        if not conditions.ema_ok_short: short_fails.append("EMA")
        if not conditions.trend_ok_short: short_fails.append("TREND")
        if STOCH_ENABLED and not conditions.stoch_ok_short: short_fails.append(f"STOCH({indicators.stoch_k:.1f}/{indicators.stoch_d:.1f})")
        if USE_HIGHER_TF_CONFIRM and not conditions.confirm_short: short_fails.append("HTF")

        if long_score >= short_score:
            none_reason = f"LONG_FAIL(S{long_score}<{min_score}:{','.join(long_fails)})"
        else:
            none_reason = f"SHORT_FAIL(S{short_score}<{min_score}:{','.join(short_fails)})"

    # Determine final status and reason
    if not conditions.volume_pass:
        status = "⏭️ SKIP"
        required_vol = get_volume_ratio_for_timeframe(timeframe)
        result = f"LOW_VOL({indicators.volume_ratio:.1f}x<{required_vol}x)"
    elif not conditions.atr_pass:
        status = "⏭️ SKIP"
        result = f"LOW_ATR({indicators.atr_pct:.3%}<{MIN_ATR_RATIO:.3%})"
    elif not conditions.bb_pass:
        status = "⏭️ SKIP"
        if indicators.bb_width < BB_WIDTH_MIN:
            result = f"BB_LOW_WIDTH({indicators.bb_width:.3%}<{BB_WIDTH_MIN:.3%})"
        else:
            result = f"BB_NOT_EXPANDING({indicators.bb_width:.3%}<={indicators.bb_width_prev:.3%})"
    elif side != "NONE":
        # Check time filter only when we have a valid signal
        if TIME_FILTER_ENABLED and not _is_valid_trading_time():
            status = "⏭️ SKIP"
            result = f"{side}_TIME_FILTER"
        else:
            status = "✅ SIGNAL"
            result = side
    else:
        status = "⏭️ SKIP"
        result = none_reason if 'none_reason' in locals() else "NO_SIGNAL"

    return side, status, result


def _log_and_save_analysis(pair, timeframe, price, status, result, long_score, short_score,
                         min_score, indicators, conditions, side):
    """Log analysis results and save to database"""
    # All metrics
    stoch_info = f"STOCH:{indicators.stoch_k:.1f}/{indicators.stoch_d:.1f} " if STOCH_ENABLED else ""
    stoch_gates = f"Stoch:{int(conditions.stoch_ok_long)}/{int(conditions.stoch_ok_short)} " if STOCH_ENABLED else ""
    adx_threshold = get_adx_threshold_for_timeframe(timeframe)

    logger.info(
        f"{status} | {timeframe} | {pair} | "
        f"Price:{price:.2f} RSI:{indicators.rsi:.1f} ADX:{indicators.adx:.1f} MACD:{indicators.diff:.4f} "
        f"EMA:{indicators.ema_fast:.2f}/{indicators.ema_slow:.2f} {stoch_info}ATR:{indicators.atr_pct:.3%} VOL:{indicators.volume_ratio:.1f}x | "
        f"Regime:{'TREND' if indicators.adx >= adx_threshold else 'RANGE'} "
        f"Score:L{long_score}/S{short_score}(min:{min_score}) | "
        f"Gates[L/S]: RSI:{int(conditions.rsi_ok_long)}/{int(conditions.rsi_ok_short)} "
        f"MACD:{int(conditions.momentum_ok_long)}/{int(conditions.momentum_ok_short)} "
        f"EMA:{int(conditions.ema_ok_long)}/{int(conditions.ema_ok_short)} "
        f"Trend:{int(conditions.trend_ok_long)}/{int(conditions.trend_ok_short)} "
        f"{stoch_gates}"
        f"HTF:{int(conditions.confirm_long)}/{int(conditions.confirm_short)} | "
        f"Result:{result}"
    )

    # Save market analysis data to database (both signals and non-signals)
    analysis_data = {
        "pair": pair,
        "timeframe": timeframe,
        "timestamp": datetime.datetime.now(datetime.UTC),
        "price": price,
        "rsi": indicators.rsi,
        "adx": indicators.adx,
        "macd": indicators.macd,
        "macd_signal": indicators.signal_line,
        "macd_diff": indicators.diff,
        "ema_fast": indicators.ema_fast,
        "ema_slow": indicators.ema_slow,
        "ema_diff": abs(indicators.ema_fast - indicators.ema_slow),
        "stoch_k": indicators.stoch_k,
        "stoch_d": indicators.stoch_d,
        "atr": indicators.atr,
        "atr_pct": indicators.atr_pct,
        "volume_ratio": indicators.volume_ratio,
        "bb_width": indicators.bb_width,
        "bb_width_prev": indicators.bb_width_prev,
        "rsi_ok_long": conditions.rsi_ok_long,
        "rsi_ok_short": conditions.rsi_ok_short,
        "macd_ok_long": indicators.macd > indicators.signal_line,
        "macd_ok_short": indicators.macd < indicators.signal_line,
        "momentum_ok_long": conditions.momentum_ok_long,
        "momentum_ok_short": conditions.momentum_ok_short,
        "ema_ok_long": conditions.ema_ok_long,
        "ema_ok_short": conditions.ema_ok_short,
        "trend_ok_long": conditions.trend_ok_long,
        "trend_ok_short": conditions.trend_ok_short,
        "stoch_ok_long": conditions.stoch_ok_long,
        "stoch_ok_short": conditions.stoch_ok_short,
        "htf_confirm_long": conditions.confirm_long,
        "htf_confirm_short": conditions.confirm_short,
        "volume_pass": conditions.volume_pass,
        "atr_pass": conditions.atr_pass,
        "time_pass": True,  # We already filtered for time above
        "bb_pass": conditions.bb_pass,
        "long_score": long_score,
        "short_score": short_score,
        "min_score_required": min_score,
        "regime": "TREND" if indicators.adx >= adx_threshold else "RANGE",
        "is_trending": indicators.adx >= adx_threshold,
        "signal_generated": side != "NONE" and conditions.volume_pass and conditions.atr_pass,
        "signal_side": side if side != "NONE" and conditions.volume_pass and conditions.atr_pass else None,
        "skip_reason": result if not (side != "NONE" and conditions.volume_pass and conditions.atr_pass) else None
    }
    save_market_analysis(analysis_data)


def _generate_signal_details(pair, timeframe, side, price, indicators, conditions,
                           long_score, short_score, min_score):
    """Generate signal details with dual TP"""
    sl, tp1, tp2 = _calculate_sl_tp(price, indicators.atr, side, pair)

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

    return {
        "signal_uuid": str(uuid.uuid4()),
        "pair": pair,
        "timeframe": timeframe,
        "side": side,
        "price": price,
        "stop_loss": sl,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "timestamp": datetime.datetime.now(datetime.UTC),
        "momentum_ok": conditions.momentum_ok_long if side == "LONG" else conditions.momentum_ok_short,
        "trend_confirmed": conditions.trend_ok_long if side == "LONG" else conditions.trend_ok_short,
        "higher_tf_confirmed": conditions.confirm_long if side == "LONG" else conditions.confirm_short,
        "confirmed": ((conditions.trend_ok_long if side == "LONG" else conditions.trend_ok_short) and
                      (conditions.confirm_long if side == "LONG" else conditions.confirm_short)),
        "score": long_score if side == "LONG" else short_score,
        "required_score": min_score,
        "rsi_ok": conditions.rsi_ok_long if side == "LONG" else conditions.rsi_ok_short,
        "ema_ok": conditions.ema_ok_long if side == "LONG" else conditions.ema_ok_short,
        "macd_ok": (indicators.macd > indicators.signal_line) if side == "LONG" else (indicators.macd < indicators.signal_line),
        "macd_momentum_ok": conditions.momentum_ok_long if side == "LONG" else conditions.momentum_ok_short,
        "stoch_ok": conditions.stoch_ok_long if side == "LONG" else conditions.stoch_ok_short,
        "rsi": indicators.rsi,
        "adx": indicators.adx,
        "macd": indicators.macd,
        "macd_signal": indicators.signal_line,
        "macd_diff": indicators.diff,
        "ema_fast": indicators.ema_fast,
        "ema_slow": indicators.ema_slow,
        "ema_diff": abs(indicators.ema_fast - indicators.ema_slow),
        "stoch_k": indicators.stoch_k,
        "stoch_d": indicators.stoch_d,
        "atr": indicators.atr,
        "atr_pct": indicators.atr_pct,
        "bb_width": indicators.bb_width,
        "bb_width_prev": indicators.bb_width_prev,
        "regime": "momentum" if indicators.adx >= get_adx_threshold_for_timeframe(timeframe) else "mean-reversion",
        "htf_used": USE_HIGHER_TF_CONFIRM,
        "volume_ratio": indicators.volume_ratio,
        "confidence": "HIGH" if long_score >= min_score + 1 or short_score >= min_score + 1 else "MEDIUM"
    }
