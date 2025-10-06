"""
Backtesting engine that simulates trading with historical data.
Uses 1m precision for TP/SL simulation and aggregated timeframes for signal generation.
"""
import logging
import json
import pandas as pd
import bisect
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Dict
from app.db.models import HistoricalOHLCV, BacktestSignal, BacktestRun
from app.db.database import SessionLocal
from app.analyzer.signals import analyze_market
from app.data_provider import set_backtest_data, set_backtest_timestamp
from app.config import (
    BACKTEST_START_DATE, BACKTEST_END_DATE, PAIRS, TIMEFRAMES,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER
)

logger = logging.getLogger(__name__)

# Timeframe mapping constant (avoid recreating per pair)
TF_MAP = {
    '5m': '5min',
    '15m': '15min',
    '1h': '1h',
    '4h': '4h',
    '1d': '1D'
}


class BacktestEngine:
    """Engine for running backtests with dual TP simulation"""

    def __init__(self, pairs: List[str], timeframes: List[str], start_date: str, end_date: str):
        """
        Initialize backtest engine.

        Args:
            pairs: List of trading pairs
            timeframes: List of timeframes to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.pairs = pairs
        self.timeframes = timeframes
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.run_id = None
        self.active_signals = {}  # Track active signals per pair/timeframe
        self.completed_signals = []
        self.first_signal_logged = False
        self.pending_updates = 0  # Track uncommitted updates for batch commits
        self.signal_buffer = []  # Buffer for bulk write at end
        self.candle_close_times = []  # Pre-computed candle close times for fast lookup

        # Parallel mode support
        self.worker_id = None  # Set by worker process
        self.main_run_id = None  # Main run_id for parallel mode
        self.is_worker = False  # Flag to indicate worker mode

    def run(self):
        """Execute the backtest"""
        db = SessionLocal()
        try:
            # Add worker ID prefix if in worker mode
            worker_prefix = f"[W{self.worker_id}] " if self.is_worker else ""

            # Create backtest run record
            self.run_id = self._create_backtest_run(db)
            logger.warning(f"{worker_prefix}Starting backtest run {self.run_id}")

            # Load all 1m data into memory for each pair
            logger.warning(f"{worker_prefix}Loading historical 1m data into memory...")
            data_cache = self._load_historical_data(db)

            # Filter pairs to only those with data
            available_pairs = list(data_cache.keys())
            if not available_pairs:
                logger.error("No data available for any pairs in the specified date range")
                self._fail_backtest_run(db, "No data available for any pairs")
                return

            if len(available_pairs) < len(self.pairs):
                missing_pairs = set(self.pairs) - set(available_pairs)
                logger.warning(f"{worker_prefix}Skipping pairs without data: {missing_pairs}")
                logger.warning(f"{worker_prefix}Running backtest on {len(available_pairs)} pairs: {available_pairs}")

            # Set the data cache for data_provider
            set_backtest_data(data_cache)

            # Pre-compute all candle close times for fast lookup
            self.candle_close_times = self._precompute_candle_close_times()

            # Walk forward through time
            logger.warning(f"{worker_prefix}Walking forward from {self.start_date} to {self.end_date}")
            current_time = self.start_date
            start_time = datetime.now()

            total_bars = int((self.end_date - self.start_date).total_seconds() / 60)
            processed_bars = 0
            skipped_bars = 0  # Track skipped timestamps for efficiency reporting

            # Pre-calculate log interval (every 1% or min 1000 bars)
            log_interval = max(1000, total_bars // 100)
            next_log = log_interval

            while current_time <= self.end_date:
                # Update data provider with current timestamp
                set_backtest_timestamp(current_time)

                # Process each timeframe
                for timeframe in self.timeframes:
                    # Check if we should generate signals for this timeframe at this time
                    if self._should_check_timeframe(current_time, timeframe):
                        self._generate_signals(current_time, timeframe, db)

                # Check active signals for TP/SL hits using 1m precision
                self._check_signal_hits(current_time, data_cache, db)

                # SMART TIME JUMPING: Skip empty minutes when no active signals
                if self.active_signals:
                    # Have active signals → must check every minute for TP/SL hits
                    prev_time = current_time
                    current_time += timedelta(minutes=1)
                    processed_bars += 1
                else:
                    # No active signals → jump to next candle close time using pre-computed list
                    prev_time = current_time

                    # Fast O(log n) binary search lookup in pre-computed sorted list
                    idx = bisect.bisect_right(self.candle_close_times, current_time)

                    # Get next candle close time
                    if idx < len(self.candle_close_times):
                        next_time = self.candle_close_times[idx]
                    else:
                        # Reached end of pre-computed times (shouldn't happen)
                        next_time = self._get_next_candle_close_time(current_time)

                    current_time = next_time

                    # Track statistics
                    minutes_skipped = int((current_time - prev_time).total_seconds() / 60) - 1
                    if minutes_skipped > 0:
                        skipped_bars += minutes_skipped
                    processed_bars += 1

                # Log progress at intervals
                if processed_bars >= next_log:
                    progress = (processed_bars / total_bars) * 100
                    active = len(self.active_signals)
                    completed = len(self.completed_signals)

                    # Calculate timing information
                    elapsed = datetime.now() - start_time
                    elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds

                    # Estimate remaining time
                    if progress > 0:
                        total_estimated = elapsed / (progress / 100)
                        remaining = total_estimated - elapsed
                        remaining_str = str(remaining).split('.')[0]
                        eta_str = f" | ETA: {remaining_str}"
                    else:
                        eta_str = ""

                    # Calculate skip efficiency
                    total_checked = processed_bars + skipped_bars
                    skip_efficiency = (skipped_bars / total_checked * 100) if total_checked > 0 else 0

                    # Add worker ID prefix if in worker mode
                    worker_prefix = f"[W{self.worker_id}] " if self.is_worker else ""

                    logger.warning(f"{worker_prefix}PROGRESS: {current_time.strftime('%Y-%m-%d')} | {progress:.1f}% | Elapsed: {elapsed_str}{eta_str} | Skipped: {skip_efficiency:.1f}% | Total: {active + completed} (Active: {active}, Completed: {completed})")
                    next_log += log_interval

            # Final commit for any pending updates
            if self.pending_updates > 0:
                db.commit()
                self.pending_updates = 0

            # Complete the backtest run
            self._complete_backtest_run(db)

            # Log final timing and efficiency summary
            total_elapsed = datetime.now() - start_time
            total_elapsed_str = str(total_elapsed).split('.')[0]
            total_checked = processed_bars + skipped_bars
            skip_efficiency = (skipped_bars / total_checked * 100) if total_checked > 0 else 0
            logger.warning(f"{worker_prefix}Backtest run {self.run_id} completed in {total_elapsed_str}")
            logger.warning(f"{worker_prefix}Performance: Processed {processed_bars:,} timestamps, Skipped {skipped_bars:,} ({skip_efficiency:.1f}% efficiency)")

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            if self.run_id:
                self._fail_backtest_run(db, str(e))
            raise
        finally:
            db.close()

    def _load_historical_data(self, db: Session) -> Dict[str, Dict]:
        """
        Load all historical data and pre-compute timeframes with HTF indicators.
        Returns: {pair: {'1m_indexed': {ts: candle}, '1m': df, '15m': df, '15m_indicators': {...}, ...}}
        """
        from ta.momentum import RSIIndicator
        from ta.trend import MACD
        from app.config import RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, HIGHER_TF_MAP, USE_HIGHER_TF_CONFIRM

        data_cache = {}

        for pair in self.pairs:
            logger.info(f"Loading {pair} 1m data...")

            # Query all 1m data for this pair in the date range
            records = db.query(HistoricalOHLCV).filter(
                and_(
                    HistoricalOHLCV.pair == pair,
                    HistoricalOHLCV.timeframe == '1m',
                    HistoricalOHLCV.timestamp >= self.start_date,
                    HistoricalOHLCV.timestamp <= self.end_date
                )
            ).order_by(HistoricalOHLCV.timestamp.asc()).all()

            if not records:
                logger.warning(f"No data found for {pair}")
                continue

            # Convert to DataFrame using direct attribute access (10x faster than dict comprehension)
            df_1m = pd.DataFrame({
                'timestamp': [r.timestamp for r in records],
                'open': [r.open for r in records],
                'high': [r.high for r in records],
                'low': [r.low for r in records],
                'close': [r.close for r in records],
                'volume': [r.volume for r in records]
            })

            logger.info(f"{pair}: Loaded {len(df_1m)} 1m candles, pre-computing timeframes...")

            # Create timestamp-indexed dict using to_dict() (100x faster than iterrows)
            df_1m_dict = df_1m.set_index('timestamp')[['high', 'low', 'close']].to_dict('index')

            # Pre-compute all required timeframes
            df_1m_indexed = df_1m.set_index('timestamp')

            timeframes_data = {'1m': df_1m, '1m_indexed': df_1m_dict}

            # Collect all higher timeframes that need indicator pre-calculation
            htf_to_precompute = set()
            if USE_HIGHER_TF_CONFIRM:
                for tf in self.timeframes:
                    higher_tf = HIGHER_TF_MAP.get(tf)
                    if higher_tf:
                        htf_to_precompute.add(higher_tf)

            for tf in self.timeframes:
                if tf == '1m':
                    continue  # Already have 1m

                if tf in TF_MAP:
                    offset = TF_MAP[tf]
                    df_resampled = df_1m_indexed.resample(
                        offset,
                        origin='epoch',
                        label='left',
                        closed='left'
                    ).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna(subset=['close'])

                    # Convert back to DataFrame with timestamp column
                    df_resampled = df_resampled.reset_index()
                    timeframes_data[tf] = df_resampled
                    logger.info(f"{pair}: Pre-computed {len(df_resampled)} {tf} candles")

                    # Pre-calculate ALL indicators for this timeframe (for performance)
                    if len(df_resampled) >= 50:
                        try:
                            logger.warning(f"{pair} {tf}: Starting indicator pre-calculation...")

                            # Import all indicator classes
                            from ta.momentum import RSIIndicator, StochasticOscillator
                            from ta.trend import MACD, ADXIndicator
                            from ta.volatility import AverageTrueRange, BollingerBands
                            from app.config import (
                                RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
                                EMA_FAST, EMA_SLOW, ATR_PERIOD, ADX_PERIOD,
                                STOCH_ENABLED, STOCH_K_PERIOD, STOCH_D_PERIOD,
                                BB_ENABLED, BB_PERIOD, BB_STD_DEV,
                                USE_TREND_FILTER, TREND_MA_PERIOD
                            )

                            # Calculate RSI
                            rsi_obj = RSIIndicator(df_resampled['close'], window=RSI_PERIOD)
                            rsi_series = rsi_obj.rsi()

                            # Calculate MACD
                            macd_obj = MACD(
                                close=df_resampled['close'],
                                window_slow=MACD_SLOW,
                                window_fast=MACD_FAST,
                                window_sign=MACD_SIGNAL
                            )
                            macd_series = macd_obj.macd()
                            macd_signal_series = macd_obj.macd_signal()
                            macd_diff_series = macd_series - macd_signal_series

                            # Calculate EMAs
                            ema_fast_series = df_resampled['close'].ewm(span=EMA_FAST).mean()
                            ema_slow_series = df_resampled['close'].ewm(span=EMA_SLOW).mean()
                            ema_diff_series = abs(ema_fast_series - ema_slow_series)

                            # Calculate ATR
                            atr_obj = AverageTrueRange(
                                high=df_resampled['high'],
                                low=df_resampled['low'],
                                close=df_resampled['close'],
                                window=ATR_PERIOD
                            )
                            atr_series = atr_obj.average_true_range()
                            atr_pct_series = atr_series / df_resampled['close']

                            # Calculate ADX
                            adx_obj = ADXIndicator(
                                high=df_resampled['high'],
                                low=df_resampled['low'],
                                close=df_resampled['close'],
                                window=ADX_PERIOD
                            )
                            adx_series = adx_obj.adx()

                            # Calculate Stochastic
                            if STOCH_ENABLED:
                                stoch_obj = StochasticOscillator(
                                    high=df_resampled['high'],
                                    low=df_resampled['low'],
                                    close=df_resampled['close'],
                                    window=STOCH_K_PERIOD,
                                    smooth_window=STOCH_D_PERIOD
                                )
                                stoch_k_series = stoch_obj.stoch()
                                stoch_d_series = stoch_obj.stoch_signal()
                            else:
                                stoch_k_series = pd.Series([50] * len(df_resampled), index=df_resampled.index)
                                stoch_d_series = pd.Series([50] * len(df_resampled), index=df_resampled.index)

                            # Calculate Bollinger Bands
                            if BB_ENABLED and len(df_resampled) >= BB_PERIOD + 1:
                                bb_obj = BollingerBands(
                                    close=df_resampled['close'],
                                    window=BB_PERIOD,
                                    window_dev=BB_STD_DEV
                                )
                                bb_upper = bb_obj.bollinger_hband()
                                bb_lower = bb_obj.bollinger_lband()
                                bb_width_series = (bb_upper - bb_lower) / df_resampled['close']
                            else:
                                bb_width_series = pd.Series([0.0] * len(df_resampled), index=df_resampled.index)

                            # Calculate volume ratio (rolling 20-period)
                            volume_ma = df_resampled['volume'].rolling(window=20, min_periods=1).mean()
                            volume_ratio_series = df_resampled['volume'] / volume_ma.shift(1)
                            volume_ratio_series = volume_ratio_series.fillna(1.0)

                            # Calculate SMA for trend filter (if enabled)
                            if USE_TREND_FILTER:
                                sma_series = df_resampled['close'].rolling(window=TREND_MA_PERIOD).mean()
                            else:
                                sma_series = pd.Series([0.0] * len(df_resampled), index=df_resampled.index)

                            # Create timestamp-indexed dictionaries for O(1) lookup
                            indicators_dict = {
                                'rsi': dict(zip(df_resampled['timestamp'], rsi_series)),
                                'macd': dict(zip(df_resampled['timestamp'], macd_series)),
                                'macd_signal': dict(zip(df_resampled['timestamp'], macd_signal_series)),
                                'macd_diff': dict(zip(df_resampled['timestamp'], macd_diff_series)),
                                'ema_fast': dict(zip(df_resampled['timestamp'], ema_fast_series)),
                                'ema_slow': dict(zip(df_resampled['timestamp'], ema_slow_series)),
                                'ema_diff': dict(zip(df_resampled['timestamp'], ema_diff_series)),
                                'atr': dict(zip(df_resampled['timestamp'], atr_series)),
                                'atr_pct': dict(zip(df_resampled['timestamp'], atr_pct_series)),
                                'adx': dict(zip(df_resampled['timestamp'], adx_series)),
                                'stoch_k': dict(zip(df_resampled['timestamp'], stoch_k_series)),
                                'stoch_d': dict(zip(df_resampled['timestamp'], stoch_d_series)),
                                'bb_width': dict(zip(df_resampled['timestamp'], bb_width_series)),
                                'volume_ratio': dict(zip(df_resampled['timestamp'], volume_ratio_series)),
                                'sma': dict(zip(df_resampled['timestamp'], sma_series))
                            }

                            timeframes_data[f'{tf}_indicators'] = indicators_dict

                            # Calculate memory usage for this cache
                            indicator_count = len(indicators_dict)
                            values_per_indicator = len(df_resampled)
                            total_values = indicator_count * values_per_indicator
                            estimated_mb = (total_values * 8) / (1024 * 1024)  # 8 bytes per float

                            logger.warning(
                                f"{pair} {tf}: ✅ Pre-computed ALL indicators | "
                                f"Indicators: {indicator_count} (inc. SMA) | Candles: {values_per_indicator} | "
                                f"Total values: {total_values:,} | Est. memory: ~{estimated_mb:.2f} MB"
                            )
                        except Exception as e:
                            logger.warning(f"{pair} {tf}: ❌ Failed to pre-compute indicators: {e}")

            data_cache[pair] = timeframes_data

        return data_cache

    def _should_check_timeframe(self, current_time: datetime, timeframe: str) -> bool:
        """
        Determine if we should check for signals on this timeframe at this time.
        This aligns with Binance candle close times.
        """
        if timeframe == '1m':
            return True  # Check every minute
        elif timeframe == '5m':
            return current_time.minute % 5 == 0
        elif timeframe == '15m':
            return current_time.minute % 15 == 0
        elif timeframe == '1h':
            return current_time.minute == 0
        elif timeframe == '4h':
            return current_time.minute == 0 and current_time.hour % 4 == 0
        elif timeframe == '1d':
            return current_time.minute == 0 and current_time.hour == 0

        return False

    def _precompute_candle_close_times(self) -> List[datetime]:
        """
        Pre-compute all candle close times for configured timeframes.
        Returns a sorted list of unique timestamps where any timeframe candle closes.
        This is computed once at startup for O(1) lookups during backtest.
        """
        worker_prefix = f"[W{self.worker_id}] " if self.is_worker else ""
        logger.warning(f"{worker_prefix}Pre-computing candle close times for fast lookup...")

        close_times_set = set()

        for tf in self.timeframes:
            current = self.start_date

            if tf == '1m':
                # Every minute
                while current <= self.end_date:
                    close_times_set.add(current)
                    current += timedelta(minutes=1)

            elif tf == '5m':
                # Align to 5-minute boundaries
                current = current.replace(second=0, microsecond=0)
                minutes_offset = current.minute % 5
                if minutes_offset > 0:
                    current += timedelta(minutes=5 - minutes_offset)

                while current <= self.end_date:
                    close_times_set.add(current)
                    current += timedelta(minutes=5)

            elif tf == '15m':
                # Align to 15-minute boundaries (0, 15, 30, 45)
                current = current.replace(second=0, microsecond=0)
                minutes_offset = current.minute % 15
                if minutes_offset > 0:
                    current += timedelta(minutes=15 - minutes_offset)

                while current <= self.end_date:
                    close_times_set.add(current)
                    current += timedelta(minutes=15)

            elif tf == '1h':
                # Align to hour boundaries
                current = current.replace(minute=0, second=0, microsecond=0)
                if current < self.start_date:
                    current += timedelta(hours=1)

                while current <= self.end_date:
                    close_times_set.add(current)
                    current += timedelta(hours=1)

            elif tf == '4h':
                # Align to 4-hour boundaries (0, 4, 8, 12, 16, 20)
                current = current.replace(minute=0, second=0, microsecond=0)
                hours_offset = current.hour % 4
                if hours_offset > 0:
                    current += timedelta(hours=4 - hours_offset)

                while current <= self.end_date:
                    close_times_set.add(current)
                    current += timedelta(hours=4)

            elif tf == '1d':
                # Align to day boundaries (midnight)
                current = current.replace(hour=0, minute=0, second=0, microsecond=0)
                if current < self.start_date:
                    current += timedelta(days=1)

                while current <= self.end_date:
                    close_times_set.add(current)
                    current += timedelta(days=1)

        # Convert to sorted list for fast iteration
        close_times_sorted = sorted(close_times_set)

        logger.warning(f"{worker_prefix}✅ Pre-computed {len(close_times_sorted):,} unique candle close times")

        return close_times_sorted

    def _get_next_candle_close_time(self, current_time: datetime) -> datetime:
        """
        Calculate the next timestamp where any configured timeframe candle closes.
        Returns the earliest next candle close across all timeframes.
        """
        next_times = []

        for tf in self.timeframes:
            if tf == '1m':
                # Next minute
                next_times.append(current_time + timedelta(minutes=1))
            elif tf == '5m':
                # Next 5-minute boundary
                minutes_until_next = 5 - (current_time.minute % 5)
                if minutes_until_next == 0:
                    minutes_until_next = 5
                next_times.append(current_time + timedelta(minutes=minutes_until_next))
            elif tf == '15m':
                # Next 15-minute boundary
                minutes_until_next = 15 - (current_time.minute % 15)
                if minutes_until_next == 0:
                    minutes_until_next = 15
                next_times.append(current_time + timedelta(minutes=minutes_until_next))
            elif tf == '1h':
                # Next hour boundary
                minutes_until_next = 60 - current_time.minute
                if minutes_until_next == 0:
                    minutes_until_next = 60
                next_times.append(current_time + timedelta(minutes=minutes_until_next))
            elif tf == '4h':
                # Next 4-hour boundary
                hours_until_next = 4 - (current_time.hour % 4)
                if hours_until_next == 0:
                    hours_until_next = 4
                next_time = current_time.replace(minute=0, second=0, microsecond=0)
                next_time += timedelta(hours=hours_until_next)
                next_times.append(next_time)
            elif tf == '1d':
                # Next day boundary (midnight)
                if current_time.hour == 0 and current_time.minute == 0:
                    next_times.append(current_time + timedelta(days=1))
                else:
                    next_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                    next_time += timedelta(days=1)
                    next_times.append(next_time)

        # Return the earliest next candle close
        return min(next_times)

    def _generate_signals(self, current_time: datetime, timeframe: str, db: Session):
        """Generate signals for a timeframe at current time"""
        try:
            # Use the analyze_market function from signals.py
            # It will automatically use the data_provider which is in backtest mode
            signals = analyze_market(self.pairs, timeframe)

            if not signals:
                return

            # Store signals and track active ones (batch inserts)
            new_signal_records = []

            for signal in signals:
                # Log first signal generation
                if not self.first_signal_logged:
                    logger.warning(f"🎯 First signal generated: {signal['pair']} {timeframe} {signal['side']} at {current_time}")
                    self.first_signal_logged = True

                # Skip if we already have an active signal for this pair/timeframe
                key = f"{signal['pair']}_{timeframe}"
                if key in self.active_signals:
                    continue

                # Create signal record (convert numpy types to Python types)
                signal_record = BacktestSignal(
                    run_id=self.run_id,
                    signal_uuid=signal['signal_uuid'],
                    pair=signal['pair'],
                    timeframe=timeframe,
                    side=signal['side'],
                    price=float(signal['price']),
                    stop_loss=float(signal['stop_loss']),
                    take_profit_1=float(signal['take_profit_1']),
                    take_profit_2=float(signal['take_profit_2']),
                    timestamp=current_time,
                    hit='PENDING',
                    momentum_ok=bool(signal['momentum_ok']),
                    trend_confirmed=bool(signal['trend_confirmed']),
                    higher_tf_confirmed=bool(signal['higher_tf_confirmed']),
                    confirmed=bool(signal['confirmed']),
                    score=int(signal['score']),
                    required_score=int(signal['required_score']),
                    rsi_ok=bool(signal['rsi_ok']),
                    ema_ok=bool(signal['ema_ok']),
                    macd_ok=bool(signal['macd_ok']),
                    macd_momentum_ok=bool(signal['macd_momentum_ok']),
                    stoch_ok=bool(signal['stoch_ok']),
                    rsi=float(signal['rsi']),
                    adx=float(signal['adx']),
                    macd=float(signal['macd']),
                    macd_signal=float(signal['macd_signal']),
                    macd_diff=float(signal['macd_diff']),
                    ema_fast=float(signal['ema_fast']),
                    ema_slow=float(signal['ema_slow']),
                    ema_diff=float(signal['ema_diff']),
                    stoch_k=float(signal['stoch_k']),
                    stoch_d=float(signal['stoch_d']),
                    atr=float(signal['atr']),
                    atr_pct=float(signal['atr_pct']),
                    bb_width=float(signal['bb_width']) if signal.get('bb_width') is not None else None,
                    bb_width_prev=float(signal['bb_width_prev']) if signal.get('bb_width_prev') is not None else None,
                    regime=signal['regime'],
                    htf_used=bool(signal['htf_used']),
                    volume_ratio=float(signal['volume_ratio']),
                    confidence=signal['confidence']
                )

                new_signal_records.append(signal_record)

                # Track as active signal
                self.active_signals[key] = {
                    'record': signal_record,
                    'entry_price': float(signal['price']),
                    'sl': float(signal['stop_loss']),
                    'tp1': float(signal['take_profit_1']),
                    'tp2': float(signal['take_profit_2']),
                    'side': signal['side'],
                    'sl_moved_to_be': False,
                    'tp1_hit': False
                }
                logger.info(f"Added active signal: {key} - {signal['side']} at {signal['price']}, SL:{signal['stop_loss']}, TP1:{signal['take_profit_1']}, TP2:{signal['take_profit_2']}")

            # Buffer signals for bulk write at end (performance optimization)
            if new_signal_records:
                self.signal_buffer.extend(new_signal_records)
                logger.info(f"Buffered {len(new_signal_records)} signals. Active signals: {len(self.active_signals)}, Total buffered: {len(self.signal_buffer)}")

        except Exception as e:
            logger.error(f"Error generating signals for {timeframe} at {current_time}: {e}")
            db.rollback()

    def _check_signal_hits(self, current_time: datetime, data_cache: Dict[str, Dict], db: Session):
        """
        Check all active signals for TP/SL hits using 1m precision (pre-indexed for speed).
        Implements dual TP logic: TP1 → SL-to-BE → TP2/BE check
        """
        if not self.active_signals:
            return  # No active signals to check

        logger.info(f"🔍 Checking {len(self.active_signals)} active signals at {current_time}")

        keys_to_remove = []

        for key, signal_data in self.active_signals.items():
            # Get the 1m candle using pre-indexed dict (O(1) lookup)
            pair = signal_data['record'].pair
            if pair not in data_cache:
                logger.warning(f"❌ Pair {pair} not in data_cache")
                continue

            candle = data_cache[pair]['1m_indexed'].get(current_time)
            if not candle:
                logger.warning(f"❌ No 1m candle for {pair} at {current_time}")
                continue

            # Extract candle data
            high = candle['high']
            low = candle['low']

            # Extract signal data
            record = signal_data['record']
            side = signal_data['side']
            entry = signal_data['entry_price']
            sl = signal_data['sl']
            tp1 = signal_data['tp1']
            tp2 = signal_data['tp2']
            tp1_hit = signal_data['tp1_hit']

            if side == 'LONG':
                # Check TP1 first (if not already hit)
                if not tp1_hit and high >= tp1:
                    # TP1 hit! Move SL to breakeven
                    signal_data['tp1_hit'] = True
                    signal_data['sl_moved_to_be'] = True
                    signal_data['sl'] = entry
                    record.sl_moved_to_be = True
                    logger.info(f"🎯 {pair} LONG: TP1 hit at {current_time}, SL moved to BE")
                    continue

                # If TP1 was hit, check for TP2 or BE
                if tp1_hit:
                    # Check TP2
                    if high >= tp2:
                        # TP2 SUCCESS!
                        record.hit = 'TP2'
                        record.hit_timestamp = current_time
                        record.hit_price = tp2
                        record.pnl_percent = ((tp2 - entry) / entry) * 100
                        keys_to_remove.append(key)
                        self.completed_signals.append(record)
                        logger.info(f"✅ {pair} LONG: TP2 hit at {current_time}, PnL: {record.pnl_percent:.2f}%")
                    # Check BE SL (after TP1) - TP1 was already hit, so count as TP1 profit
                    elif low <= entry:
                        # TP1 was hit, now hitting breakeven = TP1 profit locked in
                        record.hit = 'TP1'
                        record.hit_timestamp = current_time
                        record.hit_price = entry  # Closed at entry (BE)
                        record.pnl_percent = ((tp1 - entry) / entry) * 100
                        keys_to_remove.append(key)
                        self.completed_signals.append(record)
                        logger.info(f"🎯 {pair} LONG: TP1 secured (closed at BE) at {current_time}, PnL: {record.pnl_percent:.2f}%")
                else:
                    # TP1 not hit yet, check original SL
                    if low <= sl:
                        # Stop loss hit
                        record.hit = 'SL'
                        record.hit_timestamp = current_time
                        record.hit_price = sl
                        record.pnl_percent = ((sl - entry) / entry) * 100
                        keys_to_remove.append(key)
                        self.completed_signals.append(record)
                        logger.info(f"❌ {pair} LONG: SL hit at {current_time}, PnL: {record.pnl_percent:.2f}%")

            else:  # SHORT
                # Check TP1 first (if not already hit)
                if not tp1_hit and low <= tp1:
                    # TP1 hit! Move SL to breakeven
                    signal_data['tp1_hit'] = True
                    signal_data['sl_moved_to_be'] = True
                    signal_data['sl'] = entry
                    record.sl_moved_to_be = True
                    logger.info(f"🎯 {pair} SHORT: TP1 hit at {current_time}, SL moved to BE")
                    continue

                # If TP1 was hit, check for TP2 or BE
                if tp1_hit:
                    # Check TP2
                    if low <= tp2:
                        # TP2 SUCCESS!
                        record.hit = 'TP2'
                        record.hit_timestamp = current_time
                        record.hit_price = tp2
                        record.pnl_percent = ((entry - tp2) / entry) * 100
                        keys_to_remove.append(key)
                        self.completed_signals.append(record)
                        logger.info(f"✅ {pair} SHORT: TP2 hit at {current_time}, PnL: {record.pnl_percent:.2f}%")
                    # Check BE SL (after TP1) - TP1 was already hit, so count as TP1 profit
                    elif high >= entry:
                        # TP1 was hit, now hitting breakeven = TP1 profit locked in
                        record.hit = 'TP1'
                        record.hit_timestamp = current_time
                        record.hit_price = entry  # Closed at entry (BE)
                        record.pnl_percent = ((entry - tp1) / entry) * 100
                        keys_to_remove.append(key)
                        self.completed_signals.append(record)
                        logger.info(f"🎯 {pair} SHORT: TP1 secured (closed at BE) at {current_time}, PnL: {record.pnl_percent:.2f}%")
                else:
                    # TP1 not hit yet, check original SL
                    if high >= sl:
                        # Stop loss hit
                        record.hit = 'SL'
                        record.hit_timestamp = current_time
                        record.hit_price = sl
                        record.pnl_percent = ((entry - sl) / entry) * 100
                        keys_to_remove.append(key)
                        self.completed_signals.append(record)
                        logger.info(f"❌ {pair} SHORT: SL hit at {current_time}, PnL: {record.pnl_percent:.2f}%")

        # Remove completed signals from active tracking
        for key in keys_to_remove:
            del self.active_signals[key]

        # Batch commits (commit every 50 signal updates to reduce I/O)
        if keys_to_remove:
            self.pending_updates += len(keys_to_remove)
            if self.pending_updates >= 50:
                db.commit()
                logger.warning(f"💾 Database commit: {self.pending_updates} updates")
                self.pending_updates = 0

    def _create_backtest_run(self, db: Session) -> int:
        """Create a new backtest run record"""
        # Worker mode: use provided main_run_id with worker in decimal places
        # Add 1 to worker_id so workers use IDs 1-N (not 0-N) to avoid collision with main run
        # E.g., main: 1977459200, worker 0: 1977459201, worker 1: 1977459202
        if self.is_worker and self.main_run_id is not None:
            run_id = self.main_run_id + self.worker_id + 1
        else:
            # Sequential mode: generate timestamp-based run_id
            run_id = int(datetime.utcnow().timestamp())

        # Capture config snapshot
        config_snapshot = {
            'pairs': self.pairs,
            'timeframes': self.timeframes,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'atr_sl_multiplier': ATR_SL_MULTIPLIER,
            'atr_tp_multiplier': ATR_TP_MULTIPLIER,
            'worker_id': self.worker_id if self.is_worker else None,
            # Add more config as needed
        }

        run = BacktestRun(
            run_id=run_id,
            start_date=self.start_date,
            end_date=self.end_date,
            pairs=json.dumps(self.pairs),
            timeframes=json.dumps(self.timeframes),
            config_snapshot=json.dumps(config_snapshot),
            status='running',
            created_at=datetime.utcnow()
        )

        db.add(run)
        db.commit()

        return run_id

    def _complete_backtest_run(self, db: Session):
        """Mark backtest run as completed and calculate summary stats"""
        from sqlalchemy import func, case

        worker_prefix = f"[W{self.worker_id}] " if self.is_worker else ""

        # Bulk write all buffered signals at once (PERFORMANCE OPTIMIZATION)
        if self.signal_buffer:
            logger.warning(f"{worker_prefix}💾 Writing {len(self.signal_buffer)} signals to database in bulk...")
            db.add_all(self.signal_buffer)
            db.commit()
            logger.warning(f"{worker_prefix}✅ Successfully wrote {len(self.signal_buffer)} signals to database")
            self.signal_buffer = []  # Clear buffer

        # Worker mode: Skip stats calculation (main process will aggregate)
        if self.is_worker:
            run = db.query(BacktestRun).filter(BacktestRun.run_id == self.run_id).first()
            if run:
                run.status = 'completed'
                run.completed_at = datetime.utcnow()
                db.commit()
            logger.warning(f"{worker_prefix}✅ Worker {self.worker_id} completed")
            return

        # Sequential mode: Calculate full stats
        run = db.query(BacktestRun).filter(BacktestRun.run_id == self.run_id).first()

        if not run:
            return

        # Use SQL aggregation for performance (instead of loading all records)
        stats = db.query(
            func.count(BacktestSignal.id).label('total'),
            func.sum(case((BacktestSignal.hit == 'TP2', 1), else_=0)).label('winners'),
            func.sum(case((BacktestSignal.hit == 'SL', 1), else_=0)).label('losers'),
            func.sum(case((BacktestSignal.hit == 'TP1', 1), else_=0)).label('tp1_wins'),  # TP1 outcomes
            func.sum(BacktestSignal.pnl_percent).label('total_pnl')
        ).filter(BacktestSignal.run_id == self.run_id).first()

        total_trades = stats.total or 0
        winners = stats.winners or 0
        losers = stats.losers or 0
        tp1_wins = stats.tp1_wins or 0
        total_pnl = stats.total_pnl or 0.0

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0
        avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0.0

        # Update run record
        run.status = 'completed'
        run.total_trades = total_trades
        run.total_winners = winners
        run.total_losers = losers
        run.total_breakeven = tp1_wins
        run.win_rate = win_rate
        run.total_pnl = total_pnl
        run.avg_pnl_per_trade = avg_pnl
        run.completed_at = datetime.utcnow()

        db.commit()

    def _fail_backtest_run(self, db: Session, error_message: str):
        """Mark backtest run as failed"""
        run = db.query(BacktestRun).filter(BacktestRun.run_id == self.run_id).first()

        if run:
            run.status = 'failed'
            run.error_message = error_message
            run.completed_at = datetime.utcnow()
            db.commit()


def run_backtest_worker(worker_id, pair_subset, timeframes, start_date, end_date, main_run_id):
    """
    Worker function for parallel backtest - runs in separate process.

    Args:
        worker_id: Unique worker identifier (0, 1, 2, 3...)
        pair_subset: Subset of pairs for this worker
        timeframes: List of timeframes
        start_date: Start date string
        end_date: End date string
        main_run_id: Main run ID to attach worker results to

    Returns:
        Dict with worker results
    """
    logger.warning(f"[W{worker_id}] Worker {worker_id} starting with pairs: {pair_subset}")

    # Create engine for this worker
    engine = BacktestEngine(pair_subset, timeframes, start_date, end_date)

    # Configure as worker
    engine.is_worker = True
    engine.worker_id = worker_id
    engine.main_run_id = main_run_id

    # Run backtest
    engine.run()

    return {
        'worker_id': worker_id,
        'pairs': pair_subset,
        'signal_count': len(engine.signal_buffer),
        'run_id': engine.run_id
    }


def merge_worker_results(main_run_id, worker_results):
    """
    Merge all worker runs into main run and calculate final stats.

    Args:
        main_run_id: Main run ID
        worker_results: List of dicts from workers
    """
    from sqlalchemy import func, case

    db = SessionLocal()

    try:
        logger.warning(f"🔀 Merging {len(worker_results)} worker results into main run {main_run_id}")

        # Update all worker signals to point to main run_id
        total_signals = 0
        for result in worker_results:
            worker_run_id = result['run_id']

            # Re-assign signals to main run
            count = db.query(BacktestSignal).filter(
                BacktestSignal.run_id == worker_run_id
            ).update({'run_id': main_run_id})

            total_signals += count
            logger.warning(f"  [W{result['worker_id']}] Merged {count} signals from {len(result['pairs'])} pairs")

            # Delete worker run record
            db.query(BacktestRun).filter(
                BacktestRun.run_id == worker_run_id
            ).delete()

        db.commit()

        # Calculate final aggregated stats for main run
        logger.warning(f"Calculating final stats for {total_signals} total signals...")

        run = db.query(BacktestRun).filter(BacktestRun.run_id == main_run_id).first()
        if not run:
            logger.error(f"Main run {main_run_id} not found!")
            return

        # Use SQL aggregation
        stats = db.query(
            func.count(BacktestSignal.id).label('total'),
            func.sum(case((BacktestSignal.hit == 'TP2', 1), else_=0)).label('winners'),
            func.sum(case((BacktestSignal.hit == 'SL', 1), else_=0)).label('losers'),
            func.sum(case((BacktestSignal.hit == 'TP1', 1), else_=0)).label('tp1_wins'),
            func.sum(BacktestSignal.pnl_percent).label('total_pnl')
        ).filter(BacktestSignal.run_id == main_run_id).first()

        total_trades = stats.total or 0
        winners = stats.winners or 0
        losers = stats.losers or 0
        tp1_wins = stats.tp1_wins or 0
        total_pnl = stats.total_pnl or 0.0

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0
        avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0.0

        # Update main run with final stats
        run.status = 'completed'
        run.total_trades = total_trades
        run.total_winners = winners
        run.total_losers = losers
        run.total_breakeven = tp1_wins
        run.win_rate = win_rate
        run.total_pnl = total_pnl
        run.avg_pnl_per_trade = avg_pnl
        run.completed_at = datetime.utcnow()

        db.commit()

        logger.warning(f"✅ Merge complete: {total_trades} trades, {win_rate:.1f}% win rate")

    finally:
        db.close()


def run_backtest_parallel():
    """Main entry point for parallel backtest"""
    from multiprocessing import Pool
    from app.config import BACKTEST_WORKERS

    n_workers = BACKTEST_WORKERS

    logger.warning("=" * 80)
    logger.warning(f"PARALLEL BACKTEST MODE ({n_workers} workers)")
    logger.warning("=" * 80)

    # Split pairs evenly across workers
    pair_groups = [PAIRS[i::n_workers] for i in range(n_workers)]

    logger.warning(f"Splitting {len(PAIRS)} pairs across {n_workers} workers:")
    for i, group in enumerate(pair_groups):
        logger.warning(f"  [W{i}] {len(group)} pairs: {group}")

    # Create main run record
    db = SessionLocal()

    # Generate base run_id: timestamp with 2 decimal places for worker IDs
    # Example: timestamp 1759774458, multiplied by 100 = 175977445800
    # This allows workers 0-99 to be added: 175977445800, 175977445801, ..., 175977445899
    # But 175977445800 exceeds INTEGER max (2,147,483,647)
    # So we divide by 2 first: 1759774458 / 2 * 100 = 87988722900 (still too big!)
    # Better: Use timestamp % 20,000,000 to stay under limit
    # Example: 1759774458 % 20000000 = 19774458, * 100 = 1977445800 ✓ (under 2.1B)
    timestamp = int(datetime.utcnow().timestamp())
    main_run_id = (timestamp % 20000000) * 100

    config_snapshot = {
        'pairs': PAIRS,
        'timeframes': TIMEFRAMES,
        'start_date': BACKTEST_START_DATE,
        'end_date': BACKTEST_END_DATE,
        'parallel_workers': n_workers,
        'atr_sl_multiplier': ATR_SL_MULTIPLIER,
        'atr_tp_multiplier': ATR_TP_MULTIPLIER,
    }

    run = BacktestRun(
        run_id=main_run_id,
        start_date=datetime.strptime(BACKTEST_START_DATE, '%Y-%m-%d'),
        end_date=datetime.strptime(BACKTEST_END_DATE, '%Y-%m-%d'),
        pairs=json.dumps(PAIRS),
        timeframes=json.dumps(TIMEFRAMES),
        config_snapshot=json.dumps(config_snapshot),
        status='running',
        created_at=datetime.utcnow()
    )

    db.add(run)
    db.commit()
    db.close()

    logger.warning(f"🚀 Created main run {main_run_id}, starting {n_workers} workers...")

    # Run workers in parallel
    with Pool(n_workers) as pool:
        args = [
            (i, pair_groups[i], TIMEFRAMES, BACKTEST_START_DATE, BACKTEST_END_DATE, main_run_id)
            for i in range(n_workers)
        ]
        results = pool.starmap(run_backtest_worker, args)

    logger.warning(f"✅ All {n_workers} workers completed, merging results...")

    # Merge all worker results
    merge_worker_results(main_run_id, results)

    logger.warning("=" * 80)
    logger.warning("PARALLEL BACKTEST COMPLETED")
    logger.warning("=" * 80)


def run_backtest():
    """Main entry point for running a backtest"""
    from app.config import BACKTEST_PARALLEL_ENABLED

    if BACKTEST_PARALLEL_ENABLED:
        run_backtest_parallel()
    else:
        # Sequential mode
        logger.warning("=" * 80)
        logger.warning("BACKTEST MODE")
        logger.warning("=" * 80)

        engine = BacktestEngine(
            pairs=PAIRS,
            timeframes=TIMEFRAMES,
            start_date=BACKTEST_START_DATE,
            end_date=BACKTEST_END_DATE
        )

        engine.run()

        logger.warning("=" * 80)
        logger.warning("BACKTEST COMPLETED")
        logger.warning("=" * 80)
