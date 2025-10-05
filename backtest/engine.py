"""
Backtesting engine that simulates trading with historical data.
Uses 1m precision for TP/SL simulation and aggregated timeframes for signal generation.
"""
import logging
import json
import pandas as pd
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

    def run(self):
        """Execute the backtest"""
        db = SessionLocal()
        try:
            # Create backtest run record
            self.run_id = self._create_backtest_run(db)
            logger.warning(f"Starting backtest run {self.run_id}")

            # Load all 1m data into memory for each pair
            logger.warning("Loading historical 1m data into memory...")
            data_cache = self._load_historical_data(db)

            # Filter pairs to only those with data
            available_pairs = list(data_cache.keys())
            if not available_pairs:
                logger.error("No data available for any pairs in the specified date range")
                self._fail_backtest_run(db, "No data available for any pairs")
                return

            if len(available_pairs) < len(self.pairs):
                missing_pairs = set(self.pairs) - set(available_pairs)
                logger.warning(f"Skipping pairs without data: {missing_pairs}")
                logger.warning(f"Running backtest on {len(available_pairs)} pairs: {available_pairs}")

            # Set the data cache for data_provider
            set_backtest_data(data_cache)

            # Walk forward through time
            logger.warning(f"Walking forward from {self.start_date} to {self.end_date}")
            current_time = self.start_date

            total_bars = int((self.end_date - self.start_date).total_seconds() / 60)
            processed_bars = 0

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

                # Move to next minute
                current_time += timedelta(minutes=1)
                processed_bars += 1

                # Log progress at intervals
                if processed_bars >= next_log:
                    progress = (processed_bars / total_bars) * 100
                    active = len(self.active_signals)
                    completed = len(self.completed_signals)
                    logger.warning(f"PROGRESS: {current_time.strftime('%Y-%m-%d')} | {progress:.1f}% | Total: {active + completed} (Active: {active}, Completed: {completed})")
                    next_log += log_interval

            # Final commit for any pending updates
            if self.pending_updates > 0:
                db.commit()
                self.pending_updates = 0

            # Complete the backtest run
            self._complete_backtest_run(db)
            logger.warning(f"Backtest run {self.run_id} completed")

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            if self.run_id:
                self._fail_backtest_run(db, str(e))
            raise
        finally:
            db.close()

    def _load_historical_data(self, db: Session) -> Dict[str, Dict]:
        """
        Load all historical data and pre-compute timeframes.
        Returns: {pair: {'1m_indexed': {ts: candle}, '1m': df, '15m': df, '1h': df, '4h': df}}
        """
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

            # Batch insert all new signals at once
            if new_signal_records:
                # Use add_all instead of bulk_save_objects to keep objects in session
                db.add_all(new_signal_records)
                db.commit()
                logger.info(f"Generated {len(new_signal_records)} signals. Active signals: {len(self.active_signals)}")

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
                    # Check BE SL (after TP1)
                    elif low <= entry:
                        # Breakeven
                        record.hit = 'BREAKEVEN'
                        record.hit_timestamp = current_time
                        record.hit_price = entry
                        record.pnl_percent = 0.0
                        keys_to_remove.append(key)
                        self.completed_signals.append(record)
                        logger.info(f"⚖️ {pair} LONG: BE hit at {current_time}")
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
                    # Check BE SL (after TP1)
                    elif high >= entry:
                        # Breakeven
                        record.hit = 'BREAKEVEN'
                        record.hit_timestamp = current_time
                        record.hit_price = entry
                        record.pnl_percent = 0.0
                        keys_to_remove.append(key)
                        self.completed_signals.append(record)
                        logger.info(f"⚖️ {pair} SHORT: BE hit at {current_time}")
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
            logger.warning(f"🏁 Completed {len(keys_to_remove)} signals at {current_time}. Remaining active: {len(self.active_signals)}")
            self.pending_updates += len(keys_to_remove)
            if self.pending_updates >= 50:
                db.commit()
                logger.warning(f"💾 Database commit: {self.pending_updates} updates")
                self.pending_updates = 0

    def _create_backtest_run(self, db: Session) -> int:
        """Create a new backtest run record"""
        # Generate run_id (timestamp-based)
        run_id = int(datetime.utcnow().timestamp())

        # Capture config snapshot
        config_snapshot = {
            'pairs': self.pairs,
            'timeframes': self.timeframes,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'atr_sl_multiplier': ATR_SL_MULTIPLIER,
            'atr_tp_multiplier': ATR_TP_MULTIPLIER,
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

        run = db.query(BacktestRun).filter(BacktestRun.run_id == self.run_id).first()

        if not run:
            return

        # Use SQL aggregation for performance (instead of loading all records)
        stats = db.query(
            func.count(BacktestSignal.id).label('total'),
            func.sum(case((BacktestSignal.hit == 'TP2', 1), else_=0)).label('winners'),
            func.sum(case((BacktestSignal.hit == 'SL', 1), else_=0)).label('losers'),
            func.sum(case((BacktestSignal.hit == 'BREAKEVEN', 1), else_=0)).label('breakeven'),
            func.sum(BacktestSignal.pnl_percent).label('total_pnl')
        ).filter(BacktestSignal.run_id == self.run_id).first()

        total_trades = stats.total or 0
        winners = stats.winners or 0
        losers = stats.losers or 0
        breakeven = stats.breakeven or 0
        total_pnl = stats.total_pnl or 0.0

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0
        avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0.0

        # Update run record
        run.status = 'completed'
        run.total_trades = total_trades
        run.total_winners = winners
        run.total_losers = losers
        run.total_breakeven = breakeven
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


def run_backtest():
    """Main entry point for running a backtest"""
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
