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

    def run(self):
        """Execute the backtest"""
        db = SessionLocal()
        try:
            # Create backtest run record
            self.run_id = self._create_backtest_run(db)
            logger.info(f"Starting backtest run {self.run_id}")

            # Load all 1m data into memory for each pair
            logger.info("Loading historical 1m data into memory...")
            data_cache = self._load_historical_data(db)

            # Set the data cache for data_provider
            set_backtest_data(data_cache)

            # Walk forward through time
            logger.info(f"Walking forward from {self.start_date} to {self.end_date}")
            current_time = self.start_date

            total_bars = int((self.end_date - self.start_date).total_seconds() / 60)
            processed_bars = 0

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

                # Log progress every 1% or every 1000 bars
                if processed_bars % max(1, total_bars // 100) == 0 or processed_bars % 1000 == 0:
                    progress = (processed_bars / total_bars) * 100
                    signals_count = len(self.completed_signals) + len(self.active_signals)
                    logger.info(f"{current_time.date()} | {progress:.1f}% | {signals_count} signals")

            # Complete the backtest run
            self._complete_backtest_run(db)
            logger.info(f"Backtest run {self.run_id} completed")

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            if self.run_id:
                self._fail_backtest_run(db, str(e))
            raise
        finally:
            db.close()

    def _load_historical_data(self, db: Session) -> Dict[str, pd.DataFrame]:
        """Load all 1m historical data into memory"""
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

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': r.timestamp,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            } for r in records])

            data_cache[pair] = df
            logger.info(f"{pair}: Loaded {len(df)} candles")

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

            # Store signals and track active ones
            for signal in signals:
                # Skip if we already have an active signal for this pair/timeframe
                key = f"{signal['pair']}_{timeframe}"
                if key in self.active_signals:
                    # In backtest mode, we skip cooldowns by default
                    # But we still don't want multiple simultaneous signals on same pair/tf
                    continue

                # Store signal in database
                signal_record = BacktestSignal(
                    run_id=self.run_id,
                    signal_uuid=signal['signal_uuid'],
                    pair=signal['pair'],
                    timeframe=timeframe,
                    side=signal['side'],
                    price=signal['price'],
                    stop_loss=signal['stop_loss'],
                    take_profit_1=signal['take_profit_1'],
                    take_profit_2=signal['take_profit_2'],
                    timestamp=current_time,
                    hit='PENDING',
                    momentum_ok=signal['momentum_ok'],
                    trend_confirmed=signal['trend_confirmed'],
                    higher_tf_confirmed=signal['higher_tf_confirmed'],
                    confirmed=signal['confirmed'],
                    score=signal['score'],
                    required_score=signal['required_score'],
                    rsi_ok=signal['rsi_ok'],
                    ema_ok=signal['ema_ok'],
                    macd_ok=signal['macd_ok'],
                    macd_momentum_ok=signal['macd_momentum_ok'],
                    stoch_ok=signal['stoch_ok'],
                    rsi=signal['rsi'],
                    adx=signal['adx'],
                    macd=signal['macd'],
                    macd_signal=signal['macd_signal'],
                    macd_diff=signal['macd_diff'],
                    ema_fast=signal['ema_fast'],
                    ema_slow=signal['ema_slow'],
                    ema_diff=signal['ema_diff'],
                    stoch_k=signal['stoch_k'],
                    stoch_d=signal['stoch_d'],
                    atr=signal['atr'],
                    atr_pct=signal['atr_pct'],
                    bb_width=signal.get('bb_width'),
                    bb_width_prev=signal.get('bb_width_prev'),
                    regime=signal['regime'],
                    htf_used=signal['htf_used'],
                    volume_ratio=signal['volume_ratio'],
                    confidence=signal['confidence']
                )

                db.add(signal_record)
                db.commit()

                # Track as active signal
                self.active_signals[key] = {
                    'record': signal_record,
                    'entry_price': signal['price'],
                    'sl': signal['stop_loss'],
                    'tp1': signal['take_profit_1'],
                    'tp2': signal['take_profit_2'],
                    'side': signal['side'],
                    'sl_moved_to_be': False,
                    'tp1_hit': False
                }

        except Exception as e:
            logger.error(f"Error generating signals for {timeframe} at {current_time}: {e}")

    def _check_signal_hits(self, current_time: datetime, data_cache: Dict[str, pd.DataFrame], db: Session):
        """
        Check all active signals for TP/SL hits using 1m precision.
        Implements dual TP logic: TP1 → SL-to-BE → TP2/BE check
        """
        keys_to_remove = []

        for key, signal_data in self.active_signals.items():
            pair = signal_data['record'].pair
            side = signal_data['side']

            # Get the 1m candle for this pair at current time
            if pair not in data_cache:
                continue

            df = data_cache[pair]
            candle = df[df['timestamp'] == current_time]

            if candle.empty:
                continue

            high = float(candle.iloc[0]['high'])
            low = float(candle.iloc[0]['low'])
            close = float(candle.iloc[0]['close'])

            entry = signal_data['entry_price']
            sl = signal_data['sl']
            tp1 = signal_data['tp1']
            tp2 = signal_data['tp2']
            tp1_hit = signal_data['tp1_hit']
            sl_moved_to_be = signal_data['sl_moved_to_be']

            if side == 'LONG':
                # Check TP1 first (if not already hit)
                if not tp1_hit and high >= tp1:
                    # TP1 hit! Move SL to breakeven
                    signal_data['tp1_hit'] = True
                    signal_data['sl_moved_to_be'] = True
                    signal_data['sl'] = entry
                    signal_data['record'].sl_moved_to_be = True
                    logger.debug(f"{pair} LONG: TP1 hit at {current_time}, SL moved to BE")
                    continue

                # If TP1 was hit, check for TP2 or BE
                if tp1_hit:
                    # Check TP2
                    if high >= tp2:
                        # TP2 SUCCESS!
                        pnl = ((tp2 - entry) / entry) * 100
                        signal_data['record'].hit = 'TP2'
                        signal_data['record'].hit_timestamp = current_time
                        signal_data['record'].hit_price = tp2
                        signal_data['record'].pnl_percent = pnl
                        keys_to_remove.append(key)
                        self.completed_signals.append(signal_data['record'])
                        logger.debug(f"{pair} LONG: TP2 hit at {current_time}, PnL: {pnl:.2f}%")
                    # Check BE SL (after TP1)
                    elif low <= entry:
                        # Breakeven
                        signal_data['record'].hit = 'BREAKEVEN'
                        signal_data['record'].hit_timestamp = current_time
                        signal_data['record'].hit_price = entry
                        signal_data['record'].pnl_percent = 0.0
                        keys_to_remove.append(key)
                        self.completed_signals.append(signal_data['record'])
                        logger.debug(f"{pair} LONG: BE hit at {current_time}")
                else:
                    # TP1 not hit yet, check original SL
                    if low <= sl:
                        # Stop loss hit
                        pnl = ((sl - entry) / entry) * 100
                        signal_data['record'].hit = 'SL'
                        signal_data['record'].hit_timestamp = current_time
                        signal_data['record'].hit_price = sl
                        signal_data['record'].pnl_percent = pnl
                        keys_to_remove.append(key)
                        self.completed_signals.append(signal_data['record'])
                        logger.debug(f"{pair} LONG: SL hit at {current_time}, PnL: {pnl:.2f}%")

            else:  # SHORT
                # Check TP1 first (if not already hit)
                if not tp1_hit and low <= tp1:
                    # TP1 hit! Move SL to breakeven
                    signal_data['tp1_hit'] = True
                    signal_data['sl_moved_to_be'] = True
                    signal_data['sl'] = entry
                    signal_data['record'].sl_moved_to_be = True
                    logger.debug(f"{pair} SHORT: TP1 hit at {current_time}, SL moved to BE")
                    continue

                # If TP1 was hit, check for TP2 or BE
                if tp1_hit:
                    # Check TP2
                    if low <= tp2:
                        # TP2 SUCCESS!
                        pnl = ((entry - tp2) / entry) * 100
                        signal_data['record'].hit = 'TP2'
                        signal_data['record'].hit_timestamp = current_time
                        signal_data['record'].hit_price = tp2
                        signal_data['record'].pnl_percent = pnl
                        keys_to_remove.append(key)
                        self.completed_signals.append(signal_data['record'])
                        logger.debug(f"{pair} SHORT: TP2 hit at {current_time}, PnL: {pnl:.2f}%")
                    # Check BE SL (after TP1)
                    elif high >= entry:
                        # Breakeven
                        signal_data['record'].hit = 'BREAKEVEN'
                        signal_data['record'].hit_timestamp = current_time
                        signal_data['record'].hit_price = entry
                        signal_data['record'].pnl_percent = 0.0
                        keys_to_remove.append(key)
                        self.completed_signals.append(signal_data['record'])
                        logger.debug(f"{pair} SHORT: BE hit at {current_time}")
                else:
                    # TP1 not hit yet, check original SL
                    if high >= sl:
                        # Stop loss hit
                        pnl = ((entry - sl) / entry) * 100
                        signal_data['record'].hit = 'SL'
                        signal_data['record'].hit_timestamp = current_time
                        signal_data['record'].hit_price = sl
                        signal_data['record'].pnl_percent = pnl
                        keys_to_remove.append(key)
                        self.completed_signals.append(signal_data['record'])
                        logger.debug(f"{pair} SHORT: SL hit at {current_time}, PnL: {pnl:.2f}%")

        # Remove completed signals from active tracking
        for key in keys_to_remove:
            del self.active_signals[key]

        # Commit updates to database in batches
        if keys_to_remove:
            db.commit()

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
        run = db.query(BacktestRun).filter(BacktestRun.run_id == self.run_id).first()

        if not run:
            return

        # Get all signals for this run
        all_signals = db.query(BacktestSignal).filter(
            BacktestSignal.run_id == self.run_id
        ).all()

        total_trades = len(all_signals)
        winners = len([s for s in all_signals if s.hit == 'TP2'])
        losers = len([s for s in all_signals if s.hit == 'SL'])
        breakeven = len([s for s in all_signals if s.hit == 'BREAKEVEN'])

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl = sum([s.pnl_percent or 0.0 for s in all_signals])
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
    logger.info("=" * 80)
    logger.info("BACKTEST MODE")
    logger.info("=" * 80)

    engine = BacktestEngine(
        pairs=PAIRS,
        timeframes=TIMEFRAMES,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE
    )

    engine.run()

    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETED")
    logger.info("=" * 80)
