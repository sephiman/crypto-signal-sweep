"""
Data provider abstraction layer for live and backtest modes.
Provides unified interface for fetching market data.
"""
import logging
import ccxt
import pandas as pd
from datetime import datetime
from typing import List, Optional
from app.config import BACKTEST_MODE

logger = logging.getLogger(__name__)

# Global state for backtest mode
_backtest_data_cache = {}
_backtest_current_timestamp = None

# Global exchange instance for live mode (reused across calls)
_exchange = ccxt.binance() if not BACKTEST_MODE else None


def set_backtest_data(data_cache: dict):
    """Set the backtest data cache (called by backtest engine)"""
    global _backtest_data_cache
    _backtest_data_cache = data_cache


def set_backtest_timestamp(timestamp: datetime):
    """Set the current backtest timestamp (called by backtest engine)"""
    global _backtest_current_timestamp
    _backtest_current_timestamp = timestamp


def clear_backtest_data():
    """
    Clear the backtest data cache to free memory.
    Called after processing each pair in one-at-a-time mode.
    """
    global _backtest_data_cache, _backtest_current_timestamp
    _backtest_data_cache.clear()
    _backtest_current_timestamp = None
    logger.info("Backtest data cache cleared")


def get_current_price(pair: str, timestamp: Optional[datetime] = None) -> float:
    """
    Get current price for a pair.

    Args:
        pair: Trading pair (e.g., 'BTC/USDT')
        timestamp: Optional timestamp for backtest mode

    Returns:
        Current price as float
    """
    if BACKTEST_MODE:
        # Use provided timestamp or global backtest timestamp
        ts = timestamp or _backtest_current_timestamp
        if ts is None:
            raise ValueError("Backtest mode requires timestamp to be set")

        # Get 1m data for the pair
        if pair not in _backtest_data_cache:
            raise ValueError(f"No backtest data available for {pair}")

        # Access the 1m DataFrame from the cache structure
        pair_data = _backtest_data_cache[pair]
        df = pair_data.get('1m')

        if df is None:
            raise ValueError(f"No 1m data available for {pair}")

        # Find the closest candle at or before the timestamp
        df_filtered = df[df['timestamp'] <= ts]
        if df_filtered.empty:
            raise ValueError(f"No data available for {pair} at {ts}")

        # Return the close price of the most recent candle
        return float(df_filtered.iloc[-1]['close'])
    else:
        # Live mode: fetch from exchange
        ticker = _exchange.fetch_ticker(pair)
        return float(ticker["last"])


def fetch_ohlcv(pair: str, timeframe: str, limit: Optional[int] = None) -> List[list]:
    """
    Fetch OHLCV data for a pair and timeframe.

    Args:
        pair: Trading pair (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '1m', '15m', '1h', '4h')
        limit: Optional limit on number of candles

    Returns:
        List of OHLCV candles: [[timestamp_ms, open, high, low, close, volume], ...]
    """
    if BACKTEST_MODE:
        # Use global backtest timestamp
        ts = _backtest_current_timestamp
        if ts is None:
            raise ValueError("Backtest mode requires timestamp to be set")

        # Get pre-computed timeframe data for the pair
        if pair not in _backtest_data_cache:
            raise ValueError(f"No backtest data available for {pair}")

        pair_data = _backtest_data_cache[pair]

        # Get the requested timeframe (skip the '1m_indexed' dict)
        if timeframe in pair_data and timeframe != '1m_indexed':
            df = pair_data[timeframe]
        else:
            raise ValueError(f"Timeframe {timeframe} not pre-computed for {pair}")

        # Ensure timestamp is a column (not index)
        if 'timestamp' not in df.columns:
            # If timestamp is the index, reset it
            df = df.reset_index()

        # Filter data up to current timestamp (no look-ahead bias)
        df_filtered = df[df['timestamp'] <= ts].copy()

        if df_filtered.empty:
            logger.debug(f"{pair}: No {timeframe} data up to {ts}")
            return []

        logger.debug(f"{pair}: Have {len(df_filtered)} {timeframe} candles up to {ts}")

        # Convert to list format
        result = df_filtered[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
        # Convert timestamp to milliseconds
        result = [[int(row[0].timestamp() * 1000)] + row[1:] for row in result]

        if limit:
            result = result[-limit:]

        return result
    else:
        # Live mode: fetch from exchange
        candles = _exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        return candles


def fetch_ohlcv_df(pairs: List[str], timeframe: str) -> dict:
    """
    Fetch OHLCV data for multiple pairs and return as DataFrames.
    This is a helper function that mimics the behavior in signals.py.

    Args:
        pairs: List of trading pairs
        timeframe: Timeframe string

    Returns:
        Dictionary mapping pair to DataFrame with columns: timestamp, open, high, low, close, volume
    """
    result = {}
    for pair in pairs:
        try:
            candles = fetch_ohlcv(pair, timeframe)

            if not candles:
                # Silently skip - normal during backtest warmup period
                logger.debug(f"No data available for {pair} on {timeframe} (warmup period)")
                continue

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # In live mode, drop the incomplete candle
            # In backtest mode, we already filtered to only complete candles
            if not BACKTEST_MODE and len(df) > 1:
                df = df.iloc[:-1]

            result[pair] = df

        except ValueError as e:
            # Backtest mode: pair not in cache or no data at timestamp
            # Silently skip during warmup period
            logger.debug(f"Skipping {pair} on {timeframe}: {e}")
            continue
        except Exception as e:
            # Unexpected errors only
            logger.warning(f"Error fetching data for {pair} on {timeframe}: {e}")
            continue

    return result
