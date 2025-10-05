"""
Historical data collector for backtesting.
Fetches 1m OHLCV data from Binance and stores in database.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import List

import ccxt
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import HistoricalOHLCV

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects historical 1m OHLCV data for backtesting"""

    def __init__(self, pairs: List[str], days: int = 365):
        """
        Initialize data collector.

        Args:
            pairs: List of trading pairs to collect
            days: Number of days of history to collect (default 365)
        """
        self.pairs = pairs
        self.days = days
        self.exchange = ccxt.binance()
        self.max_candles_per_request = 1000
        self.rate_limit_per_minute = 1200
        self.request_count = 0
        self.minute_start = time.time()

    def collect_all(self):
        """Collect data for all pairs"""
        logger.info(f"Starting data collection for {len(self.pairs)} pairs, {self.days} days of 1m candles")

        for i, pair in enumerate(self.pairs, 1):
            logger.info(f"[{i}/{len(self.pairs)}] Processing {pair}")
            try:
                self.collect_pair(pair)
            except Exception as e:
                logger.error(f"Failed to collect data for {pair}: {e}")
                # Continue with next pair instead of failing entire process
                continue

        logger.info("Data collection completed")

    def collect_pair(self, pair: str):
        """
        Collect historical 1m data for a single pair.

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
        """
        db = SessionLocal()
        try:
            # Calculate date range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=self.days)

            # Check what data we already have
            existing_start, existing_end = self._get_existing_range(db, pair)

            if existing_start and existing_end:
                # We have some data, check for gaps
                logger.info(f"{pair}: Existing data from {existing_start} to {existing_end}")

                # Collect missing data before existing range
                if start_time < existing_start:
                    logger.info(f"{pair}: Collecting data from {start_time} to {existing_start}")
                    self._fetch_and_store(db, pair, start_time, existing_start)

                # Collect missing data after existing range
                if existing_end < end_time:
                    logger.info(f"{pair}: Collecting data from {existing_end} to {end_time}")
                    self._fetch_and_store(db, pair, existing_end, end_time)
            else:
                # No existing data, collect everything
                logger.info(f"{pair}: No existing data, collecting from {start_time} to {end_time}")
                self._fetch_and_store(db, pair, start_time, end_time)

        finally:
            db.close()

    def _get_existing_range(self, db: Session, pair: str):
        """Get the date range of existing data for a pair"""
        min_ts = db.query(HistoricalOHLCV.timestamp).filter(
            HistoricalOHLCV.pair == pair
        ).order_by(HistoricalOHLCV.timestamp.asc()).first()

        max_ts = db.query(HistoricalOHLCV.timestamp).filter(
            HistoricalOHLCV.pair == pair
        ).order_by(HistoricalOHLCV.timestamp.desc()).first()

        if min_ts and max_ts:
            return min_ts[0], max_ts[0]
        return None, None

    def _fetch_and_store(self, db: Session, pair: str, start: datetime, end: datetime):
        """
        Fetch data from exchange and store in database.

        Args:
            db: Database session
            pair: Trading pair
            start: Start datetime
            end: End datetime
        """
        current = start
        total_candles = 0
        retry_count = 0
        max_retries = 3

        while current < end:
            # Rate limiting check
            self._check_rate_limit()

            # Calculate how many candles we can fetch
            since_ms = int(current.timestamp() * 1000)

            try:
                # Fetch candles from exchange
                candles = self.exchange.fetch_ohlcv(
                    pair,
                    '1m',
                    since=since_ms,
                    limit=self.max_candles_per_request
                )

                self.request_count += 1

                if not candles:
                    # No more data available (token might be newer than requested date)
                    logger.info(f"{pair}: No more data available from {current}")
                    break

                # Store candles in database
                stored = self._store_candles(db, pair, candles)
                total_candles += stored

                # Log progress every 1000 candles
                if total_candles % 1000 == 0:
                    logger.info(f"{pair}: Stored {total_candles} candles so far")

                # Move to next batch
                last_candle_time = datetime.utcfromtimestamp(candles[-1][0] / 1000)
                current = last_candle_time + timedelta(minutes=1)

                # Reset retry counter on success
                retry_count = 0

                # If we got less than requested, we've reached the end
                if len(candles) < self.max_candles_per_request:
                    break

            except ccxt.NetworkError as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"{pair}: Network error after {max_retries} retries: {e}")
                    raise
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"{pair}: Network error, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)

            except ccxt.ExchangeError as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"{pair}: Exchange error after {max_retries} retries: {e}")
                    raise
                wait_time = 2 ** retry_count
                logger.warning(f"{pair}: Exchange error, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"{pair}: Unexpected error: {e}")
                raise

        logger.info(f"{pair}: Successfully stored {total_candles} candles")

    def _store_candles(self, db: Session, pair: str, candles: List[list]) -> int:
        """
        Store candles in database, skipping duplicates.

        Args:
            db: Database session
            pair: Trading pair
            candles: List of OHLCV candles

        Returns:
            Number of candles stored
        """
        if not candles:
            return 0

        # Extract all timestamps from incoming candles
        timestamps = [datetime.utcfromtimestamp(c[0] / 1000) for c in candles]

        # Batch check which timestamps already exist (single query instead of N queries)
        existing_timestamps = set(
            row[0] for row in db.query(HistoricalOHLCV.timestamp).filter(
                and_(
                    HistoricalOHLCV.pair == pair,
                    HistoricalOHLCV.timestamp.in_(timestamps)
                )
            ).all()
        )

        # Prepare bulk insert for new records only
        new_records = []
        for candle in candles:
            timestamp = datetime.utcfromtimestamp(candle[0] / 1000)

            # Skip if already exists
            if timestamp in existing_timestamps:
                continue

            new_records.append(
                HistoricalOHLCV(
                    pair=pair,
                    timeframe='1m',
                    timestamp=timestamp,
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5])
                )
            )

        # Bulk insert all new records
        if new_records:
            db.bulk_save_objects(new_records)
            db.commit()

        return len(new_records)

    def _check_rate_limit(self):
        """Check and enforce rate limits"""
        # Reset counter every minute
        current_time = time.time()
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time

        # If we're approaching the limit, wait
        if self.request_count >= self.rate_limit_per_minute - 10:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.info(f"Rate limit approaching, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.minute_start = time.time()


def collect_historical_data(pairs: List[str], days: int = 365):
    """
    Main function to collect historical data.

    Args:
        pairs: List of trading pairs
        days: Number of days of history to collect
    """
    collector = DataCollector(pairs, days)
    collector.collect_all()
