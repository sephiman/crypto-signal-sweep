"""
Backtest cache operations module.

This module contains all backtest-specific functions for retrieving
pre-calculated indicators and market data from cache. Separating these
functions keeps signals.py focused on core analysis logic.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from app.analyzer.signals import TechnicalIndicators, MarketConditions

logger = logging.getLogger(__name__)


def get_indicators_from_cache(candle_ts, prev_candle_ts, indicators_cache, price) -> TechnicalIndicators:
    """
    Extract all indicators from pre-calculated cache (backtest mode only).
    Pure dictionary lookups, no calculations.

    Args:
        candle_ts: Current candle timestamp
        prev_candle_ts: Previous candle timestamp (for bb_width_prev)
        indicators_cache: Pre-calculated indicators dict
        price: Current price

    Returns:
        TechnicalIndicators object
    """
    # Direct O(1) lookups from cache
    rsi = indicators_cache.get('rsi', {}).get(candle_ts)
    macd = indicators_cache.get('macd', {}).get(candle_ts)
    signal_line = indicators_cache.get('macd_signal', {}).get(candle_ts)
    diff = indicators_cache.get('macd_diff', {}).get(candle_ts)
    ema_fast = indicators_cache.get('ema_fast', {}).get(candle_ts)
    ema_slow = indicators_cache.get('ema_slow', {}).get(candle_ts)
    atr = indicators_cache.get('atr', {}).get(candle_ts)
    atr_pct = indicators_cache.get('atr_pct', {}).get(candle_ts)
    adx = indicators_cache.get('adx', {}).get(candle_ts)
    stoch_k = indicators_cache.get('stoch_k', {}).get(candle_ts, 50)
    stoch_d = indicators_cache.get('stoch_d', {}).get(candle_ts, 50)
    volume_ratio = indicators_cache.get('volume_ratio', {}).get(candle_ts, 1.0)
    bb_width = indicators_cache.get('bb_width', {}).get(candle_ts, 0.0)
    bb_width_prev = indicators_cache.get('bb_width', {}).get(prev_candle_ts, 0.0) if prev_candle_ts else 0.0

    # Compute derived values if needed
    if diff is None and macd is not None and signal_line is not None:
        diff = macd - signal_line
    if atr_pct is None and atr is not None:
        atr_pct = atr / price

    return TechnicalIndicators(
        rsi=rsi, macd=macd, signal_line=signal_line, diff=diff,
        ema_fast=ema_fast, ema_slow=ema_slow, atr=atr, atr_pct=atr_pct,
        adx=adx, stoch_k=stoch_k, stoch_d=stoch_d,
        volume_ratio=volume_ratio, bb_width=bb_width, bb_width_prev=bb_width_prev
    )


def get_conditions_from_cache(data, indicators: TechnicalIndicators, timeframe: str,
                               sma_cache: Dict, htf_cache: Dict) -> MarketConditions:
    """
    Calculate market conditions using pre-calculated caches (backtest mode only).
    No conditionals for BACKTEST_MODE, assumes all caches available.

    Args:
        data: OHLCV DataFrame (for timestamps and recent bars)
        indicators: TechnicalIndicators object (from cache)
        timeframe: Trading timeframe
        sma_cache: Pre-calculated SMA dict
        htf_cache: Pre-calculated HTF indicators dict (pair_cache)

    Returns:
        MarketConditions object
    """
    from app.config import (
        VOLUME_CONFIRMATION_ENABLED, MIN_ATR_RATIO, MACD_MIN_DIFF_ENABLED, MACD_MIN_DIFF,
        EMA_MIN_DIFF_ENABLED, ADX_RSI_MODE, RSI_OVERSOLD, RSI_OVERBOUGHT,
        RSI_TRENDING_MODE, RSI_TRENDING_PULLBACK_LONG, RSI_TRENDING_PULLBACK_SHORT,
        RSI_MOMENTUM, RSI_TRENDING_OVERSOLD, RSI_TRENDING_OVERBOUGHT,
        USE_TREND_FILTER, REQUIRED_MA_BARS, STOCH_ENABLED, STOCH_OVERSOLD, STOCH_OVERBOUGHT,
        USE_HIGHER_TF_CONFIRM, HIGHER_TF_MAP, BB_ENABLED, BB_WIDTH_MIN,
        get_adx_threshold_for_timeframe, get_volume_ratio_for_timeframe
    )

    # Volume and ATR filters
    volume_pass = not VOLUME_CONFIRMATION_ENABLED or indicators.volume_ratio >= (get_volume_ratio_for_timeframe(timeframe) - 0.01)
    atr_pass = indicators.atr_pct >= MIN_ATR_RATIO
    min_ema_separation = indicators.atr * 0.5

    # MACD momentum
    if MACD_MIN_DIFF_ENABLED:
        momentum_ok_long = (indicators.macd > indicators.signal_line) and (indicators.diff >= MACD_MIN_DIFF)
        momentum_ok_short = (indicators.macd < indicators.signal_line) and (indicators.diff <= -MACD_MIN_DIFF)
    else:
        momentum_ok_long = indicators.macd > indicators.signal_line
        momentum_ok_short = indicators.macd < indicators.signal_line

    # EMA
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
        rsi_ok_long = indicators.rsi < RSI_OVERSOLD
        rsi_ok_short = indicators.rsi > RSI_OVERBOUGHT
    else:
        if is_trending:
            if RSI_TRENDING_MODE == "pullback":
                rsi_ok_long = RSI_TRENDING_PULLBACK_LONG < indicators.rsi < RSI_MOMENTUM
                rsi_ok_short = RSI_MOMENTUM < indicators.rsi < RSI_TRENDING_PULLBACK_SHORT
            else:
                rsi_ok_long = indicators.rsi < RSI_TRENDING_OVERSOLD
                rsi_ok_short = indicators.rsi > RSI_TRENDING_OVERBOUGHT
        else:
            rsi_ok_long = indicators.rsi < RSI_OVERSOLD
            rsi_ok_short = indicators.rsi > RSI_OVERBOUGHT

    # Trend filter (using pre-calculated SMA)
    if USE_TREND_FILTER and sma_cache and len(data) >= REQUIRED_MA_BARS:
        recent_timestamps = data['timestamp'].iloc[-REQUIRED_MA_BARS:].tolist()
        recent_closes = data['close'].iloc[-REQUIRED_MA_BARS:].tolist()
        recent_sma_values = [sma_cache.get(ts) for ts in recent_timestamps]

        if all(sma_val is not None for sma_val in recent_sma_values):
            trend_ok_long = all(close > sma for close, sma in zip(recent_closes, recent_sma_values))
            trend_ok_short = all(close < sma for close, sma in zip(recent_closes, recent_sma_values))
        else:
            trend_ok_long = trend_ok_short = True
    else:
        trend_ok_long = trend_ok_short = True

    # Stochastic
    if STOCH_ENABLED:
        stoch_ok_long = indicators.stoch_k < STOCH_OVERSOLD and indicators.stoch_d < STOCH_OVERSOLD
        stoch_ok_short = indicators.stoch_k > STOCH_OVERBOUGHT and indicators.stoch_d > STOCH_OVERBOUGHT
    else:
        stoch_ok_long = stoch_ok_short = True

    # Higher timeframe confirmation (using pre-calculated HTF indicators)
    if USE_HIGHER_TF_CONFIRM:
        higher_tf = HIGHER_TF_MAP.get(timeframe)
        if higher_tf and htf_cache:
            confirm_long, confirm_short = get_htf_confirmation_from_cache(data, higher_tf, htf_cache)
        else:
            confirm_long = confirm_short = True
    else:
        confirm_long = confirm_short = True

    # Bollinger Bands
    if BB_ENABLED:
        bb_expanding = indicators.bb_width > indicators.bb_width_prev
        bb_above_min = indicators.bb_width >= BB_WIDTH_MIN
        bb_pass = bb_expanding and bb_above_min
    else:
        bb_pass = True

    return MarketConditions(
        rsi_ok_long, rsi_ok_short, momentum_ok_long, momentum_ok_short,
        ema_ok_long, ema_ok_short, trend_ok_long, trend_ok_short,
        stoch_ok_long, stoch_ok_short, confirm_long, confirm_short,
        volume_pass, atr_pass, is_trending, bb_pass
    )


def get_htf_confirmation_from_cache(data: pd.DataFrame, higher_tf: str,
                                    htf_cache: Dict) -> Tuple[bool, bool]:
    """
    Get HTF confirmation from pre-calculated cache (backtest mode only).

    Args:
        data: Current timeframe OHLCV data
        higher_tf: Higher timeframe to check
        htf_cache: Pre-calculated HTF indicators dict (pair_cache)

    Returns:
        Tuple of (confirm_long, confirm_short)
    """
    current_ts = data['timestamp'].iloc[-1]

    # Get HTF indicators
    htf_data = htf_cache.get(higher_tf)
    htf_indicators = htf_cache.get(f'{higher_tf}_indicators')

    if htf_data is not None and htf_indicators and len(htf_data) > 0:
        # Find most recent HTF candle at or before current timestamp
        valid_candles = htf_data[htf_data['timestamp'] <= current_ts]
        if len(valid_candles) == 0:
            return True, True

        htf_ts = valid_candles['timestamp'].iloc[-1]

        # Lookup indicators
        ht_rsi = htf_indicators.get('rsi', {}).get(htf_ts)
        ht_macd = htf_indicators.get('macd', {}).get(htf_ts)
        ht_signal = htf_indicators.get('macd_signal', {}).get(htf_ts)

        if ht_rsi is None or ht_macd is None or ht_signal is None:
            return True, True

        # HTF confirmation logic
        confirm_long = ht_rsi > 45 and ht_macd > ht_signal and (ht_macd - ht_signal) > 0.5
        confirm_short = ht_rsi < 55 and ht_macd < ht_signal and (ht_signal - ht_macd) > 0.5

        return confirm_long, confirm_short

    return True, True
