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


def set_backtest_data(data_cache: dict):
    """Set the backtest data cache (called by backtest engine)"""
    global _backtest_data_cache
    _backtest_data_cache = data_cache


def set_backtest_timestamp(timestamp: datetime):
    """Set the current backtest timestamp (called by backtest engine)"""
    global _backtest_current_timestamp
    _backtest_current_timestamp = timestamp


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

        df = _backtest_data_cache[pair]

        # Find the closest candle at or before the timestamp
        df_filtered = df[df['timestamp'] <= ts]
        if df_filtered.empty:
            raise ValueError(f"No data available for {pair} at {ts}")

        # Return the close price of the most recent candle
        return float(df_filtered.iloc[-1]['close'])
    else:
        # Live mode: fetch from exchange
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(pair)
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

        # Get 1m data for the pair
        if pair not in _backtest_data_cache:
            raise ValueError(f"No backtest data available for {pair}")

        df_1m = _backtest_data_cache[pair]

        # Filter data up to current timestamp (no look-ahead bias)
        df_1m = df_1m[df_1m['timestamp'] <= ts].copy()

        if df_1m.empty:
            return []

        # If timeframe is 1m, return as-is
        if timeframe == '1m':
            result = df_1m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
            # Convert timestamp to milliseconds
            result = [[int(row[0].timestamp() * 1000)] + row[1:] for row in result]
            if limit:
                result = result[-limit:]
            return result

        # Aggregate to requested timeframe using pandas resample
        df_1m = df_1m.set_index('timestamp')

        # Map timeframe to pandas offset
        tf_map = {
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }

        if timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        offset = tf_map[timeframe]

        # Resample with proper alignment
        # origin='start' ensures bars align with Binance timing
        # label='left' puts the timestamp at the start of the interval
        # closed='left' includes the left edge but not the right
        df_resampled = df_1m.resample(
            offset,
            origin='start',
            label='left',
            closed='left'
        ).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Convert back to list format
        result = []
        for ts_idx, row in df_resampled.iterrows():
            result.append([
                int(ts_idx.timestamp() * 1000),  # timestamp in ms
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ])

        if limit:
            result = result[-limit:]

        return result
    else:
        # Live mode: fetch from exchange
        exchange = ccxt.binance()
        candles = exchange.fetch_ohlcv(pair, timeframe, limit=limit)
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
        candles = fetch_ohlcv(pair, timeframe)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # In live mode, drop the incomplete candle
        # In backtest mode, we already filtered to only complete candles
        if not BACKTEST_MODE and len(df) > 1:
            df = df.iloc[:-1]

        result[pair] = df

    return result
