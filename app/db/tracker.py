import datetime
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import ccxt
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import DB_ENABLED, DB_URL
from app.db.models import Base, Signal, MarketAnalysis

# Configure logging and database
logger = logging.getLogger(__name__)

engine = None
if DB_ENABLED:
    engine = create_engine(DB_URL, echo=False, future=True)
    Base.metadata.create_all(engine)

exchange = ccxt.binance()


# ============================================================================
# DATA CLASSES AND TYPES
# ============================================================================

@dataclass
class PriceData:
    """Container for price analysis data."""
    high: float
    low: float
    current_price: float
    ohlcv: List


@dataclass
class SignalUpdate:
    """Container for signal update information."""
    signal_uuid: str
    pair: str
    timeframe: str
    side: str
    price: float
    hit: str
    hit_timestamp: datetime.datetime
    action: str


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _to_native(obj) -> Any:
    """Convert numpy types to native Python types for SQLAlchemy."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def _create_update_record(signal: Signal, current_price: float, hit_type: str,
                          timestamp: datetime.datetime, action: str) -> Dict[str, Union[str, float, datetime.datetime]]:
    """Create an update record for signal status changes."""
    return {
        "signal_uuid": signal.signal_uuid,
        "pair": signal.pair,
        "timeframe": signal.timeframe,
        "side": signal.side,
        "price": current_price,
        "hit": hit_type,
        "hit_timestamp": timestamp,
        "action": action
    }


# ============================================================================
# PRICE ANALYSIS CLASS
# ============================================================================

class PriceAnalyzer:
    """Handles price data analysis and extreme value calculations."""

    @staticmethod
    def get_price_extremes(ohlcv: List[List[Union[int, float]]]) -> Tuple[float, float]:
        """Extract price extremes from OHLCV data."""
        if not ohlcv:
            raise ValueError("OHLCV data cannot be empty")

        highs = [candle[2] for candle in ohlcv]
        lows = [candle[3] for candle in ohlcv]
        return max(highs), min(lows)

    @staticmethod
    def get_price_extremes_since_timestamp(ohlcv: List[List[Union[int, float]]], since_timestamp: int) -> Tuple[Optional[float], Optional[float]]:
        """Extract price extremes from OHLCV data since a specific timestamp."""
        if not ohlcv:
            return None, None

        # Filter candles to only include those after the specified timestamp
        filtered_ohlcv = [candle for candle in ohlcv if candle[0] >= since_timestamp]

        if not filtered_ohlcv:
            return None, None

        highs = [candle[2] for candle in filtered_ohlcv]
        lows = [candle[3] for candle in filtered_ohlcv]
        return max(highs), min(lows)

    @staticmethod
    def get_current_price(ohlcv: List[List[Union[int, float]]]) -> float:
        """Get the current price (latest close) from OHLCV data."""
        if not ohlcv:
            raise ValueError("OHLCV data cannot be empty")
        return ohlcv[-1][4]  # Close price

    @classmethod
    def create_price_data(cls, ohlcv: List[List[Union[int, float]]]) -> PriceData:
        """Create a PriceData object from OHLCV data."""
        high, low = cls.get_price_extremes(ohlcv)
        current_price = cls.get_current_price(ohlcv)
        return PriceData(
            high=high,
            low=low,
            current_price=current_price,
            ohlcv=ohlcv
        )


# ============================================================================
# SIGNAL CONDITION CHECKER CLASS
# ============================================================================

class SignalChecker:
    """Handles signal condition checking (TP1, TP2, SL, Breakeven)."""

    @staticmethod
    def check_tp1_hit(signal: Signal, high: float, low: float) -> bool:
        """Check if TP1 is hit for a signal."""
        if signal.side == "LONG":
            return high >= signal.take_profit_1
        else:  # SHORT
            return low <= signal.take_profit_1

    @staticmethod
    def check_tp2_hit(signal: Signal, high: float, low: float) -> bool:
        """Check if TP2 is hit for a signal."""
        if signal.side == "LONG":
            return high >= signal.take_profit_2
        else:  # SHORT
            return low <= signal.take_profit_2

    @staticmethod
    def check_stop_loss_hit(signal: Signal, high: float, low: float) -> bool:
        """Check if original stop loss is hit."""
        if signal.side == "LONG":
            return low <= signal.stop_loss
        else:  # SHORT
            return high >= signal.stop_loss

    @staticmethod
    def check_breakeven_hit(signal: Signal, high: float, low: float) -> bool:
        """Check if breakeven stop loss (entry price) is hit."""
        if signal.side == "LONG":
            return low <= signal.price
        else:  # SHORT
            return high >= signal.price


# ============================================================================
# SIGNAL HANDLER FUNCTIONS
# ============================================================================

def _handle_pending_signal(signal: Signal, high: float, low: float, current_price: float, now: datetime.datetime) -> Optional[Dict]:
    """Handle a signal in PENDING status."""
    update = None

    # Check TP1 first (higher priority)
    if SignalChecker.check_tp1_hit(signal, high, low):
        signal.hit = "TP1_HIT"
        signal.sl_moved_to_be = True
        signal.hit_timestamp = now
        action = f"TP1 hit at {signal.take_profit_1:.6f}, SL moved to breakeven at {signal.price:.6f}"
        update = _create_update_record(signal, current_price, "TP1_HIT", now, action)
        logger.info(
            f"TP1 HIT: {signal.pair} {signal.side} - TP1: {signal.take_profit_1:.6f}, SL moved to BE: {signal.price:.6f}")

    # Check original SL (only if TP1 not hit)
    elif SignalChecker.check_stop_loss_hit(signal, high, low):
        signal.hit = "FAILURE"
        signal.hit_timestamp = now
        action = f"Stop Loss hit at {signal.stop_loss:.6f}"
        update = _create_update_record(signal, current_price, "FAILURE", now, action)
        logger.info(f"SL HIT: {signal.pair} {signal.side} - SL: {signal.stop_loss:.6f}")

    return update


def _handle_tp1_hit_signal(signal: Signal, high: float, low: float, current_price: float, now: datetime.datetime) -> Optional[Dict]:
    """Handle a signal that already hit TP1."""
    update = None

    # Check TP2 first (higher priority)
    if SignalChecker.check_tp2_hit(signal, high, low):
        signal.hit = "SUCCESS"
        signal.hit_timestamp = now
        action = f"TP2 hit at {signal.take_profit_2:.6f} (Full profit)"
        update = _create_update_record(signal, current_price, "SUCCESS", now, action)
        logger.info(f"TP2 HIT: {signal.pair} {signal.side} - TP2: {signal.take_profit_2:.6f}")

    # Check breakeven SL (only if TP2 not hit)
    elif SignalChecker.check_breakeven_hit(signal, high, low):
        signal.hit = "BREAKEVEN"
        signal.hit_timestamp = now
        action = f"Breakeven SL hit at {signal.price:.6f} (Partial profit secured)"
        update = _create_update_record(signal, current_price, "BREAKEVEN", now, action)
        logger.info(f"BREAKEVEN: {signal.pair} {signal.side} - BE SL: {signal.price:.6f}")

    return update


def _handle_tp1_hit_signal_with_proper_ranges(signal: Signal, ohlcv: List[List[Union[int, float]]], current_price: float, now: datetime.datetime) -> Optional[Dict]:
    """
    Handle a signal that already hit TP1 with proper price range checking.
    TP2 check uses full history, breakeven check uses only post-TP1 history.
    """
    update = None

    # For TP2 check, use the entire price history since signal creation
    full_high, full_low = PriceAnalyzer.get_price_extremes(ohlcv)

    # Check TP2 first (higher priority) - can happen any time since signal creation
    if SignalChecker.check_tp2_hit(signal, full_high, full_low):
        signal.hit = "SUCCESS"
        signal.hit_timestamp = now
        action = f"TP2 hit at {signal.take_profit_2:.6f} (Full profit)"
        update = _create_update_record(signal, current_price, "SUCCESS", now, action)
        logger.info(f"TP2 HIT: {signal.pair} {signal.side} - TP2: {signal.take_profit_2:.6f}")

    else:
        # For breakeven check, only consider price action AFTER TP1 was hit
        if signal.hit_timestamp:
            # Convert hit_timestamp to milliseconds for comparison with OHLCV data
            tp1_timestamp_ms = int(signal.hit_timestamp.timestamp() * 1000)
            post_tp1_high, post_tp1_low = PriceAnalyzer.get_price_extremes_since_timestamp(ohlcv, tp1_timestamp_ms)

            # Only check breakeven if we have valid post-TP1 data
            if post_tp1_high is not None and post_tp1_low is not None:
                if SignalChecker.check_breakeven_hit(signal, post_tp1_high, post_tp1_low):
                    signal.hit = "BREAKEVEN"
                    signal.hit_timestamp = now
                    action = f"Breakeven SL hit at {signal.price:.6f} (Partial profit secured)"
                    update = _create_update_record(signal, current_price, "BREAKEVEN", now, action)
                    logger.info(f"BREAKEVEN: {signal.pair} {signal.side} - BE SL: {signal.price:.6f} (post-TP1 check)")
            else:
                logger.debug(f"No post-TP1 price data available for breakeven check: {signal.pair}")

    return update


# ============================================================================
# MAIN TRACKING FUNCTION
# ============================================================================

def check_hit_signals() -> List[Dict]:
    """
    Signal tracking with dual TP levels and SL-to-breakeven management.
    Handles PENDING -> TP1_HIT -> SUCCESS/FAILURE flow.
    """
    if not DB_ENABLED:
        return []

    updated = []
    now = datetime.datetime.now(datetime.UTC)

    with Session(engine) as session:
        # Load signals that need monitoring (PENDING or TP1_HIT)
        active_signals = session.query(Signal).filter(
            Signal.hit.in_(["PENDING", "TP1_HIT"])
        ).all()

        if not active_signals:
            return []

        for signal in active_signals:
            try:
                # Fetch price data since signal was created
                since = int(signal.timestamp.timestamp() * 1000)
                ohlcv = exchange.fetch_ohlcv(signal.pair, "1m", since=since)

                if not ohlcv:
                    continue

                current_price = ohlcv[-1][4]  # Current close price

                # Handle signal based on its current status
                if signal.hit == "PENDING":
                    # For pending signals, check entire price history since signal creation
                    high, low = PriceAnalyzer.get_price_extremes(ohlcv)
                    update = _handle_pending_signal(signal, high, low, current_price, now)
                elif signal.hit == "TP1_HIT":
                    # For TP1_HIT signals, we need different price ranges for different checks
                    update = _handle_tp1_hit_signal_with_proper_ranges(signal, ohlcv, current_price, now)

                if update:
                    updated.append(update)

            except ccxt.BaseError as e:
                logger.error(f"Exchange API error for {signal.pair}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error checking signal {signal.pair}: {e}", exc_info=True)

        session.commit()

    return updated


# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

def summarize_and_notify() -> Optional[str]:
    """Summary with dual TP system results and profit calculations (based on $100 positions)."""
    if not DB_ENABLED:
        return None

    now = datetime.datetime.now(datetime.UTC)
    periods = [
        ("24h", datetime.timedelta(days=1)),
        ("7d", datetime.timedelta(days=7)),
        ("30d", datetime.timedelta(days=30)),
    ]
    lines = ["📊 Daily Trading Performance Summary", ""]

    with Session(engine) as session:
        for label, delta in periods:
            cutoff = now - delta

            # Get all completed signals with their data
            completed_signals = session.query(Signal) \
                .filter(Signal.hit_timestamp != None,
                        Signal.hit_timestamp > cutoff,
                        Signal.hit.in_(["SUCCESS", "FAILURE", "BREAKEVEN"])) \
                .all()

            if not completed_signals:
                lines.append(f"{label}: 0W/0BE/0L (0.0%) | P&L: +0.0% (+0.0% per trade)")
                continue

            total = len(completed_signals)
            total_pnl = 0.0
            full_success = 0
            partial_success = 0
            failures = 0

            for signal in completed_signals:
                if signal.hit == "SUCCESS":
                    # Full TP2 hit - calculate profit percentage
                    if signal.side == "LONG":
                        profit_pct = (signal.take_profit_2 - signal.price) / signal.price * 100
                    else:  # SHORT
                        profit_pct = (signal.price - signal.take_profit_2) / signal.price * 100

                    total_pnl += profit_pct
                    full_success += 1

                elif signal.hit == "BREAKEVEN":
                    # TP1 hit then returned to breakeven - calculate partial profit percentage
                    if signal.side == "LONG":
                        tp1_profit_pct = (signal.take_profit_1 - signal.price) / signal.price * 100
                    else:  # SHORT
                        tp1_profit_pct = (signal.price - signal.take_profit_1) / signal.price * 100

                    # Partial profit (50% position closed at TP1) minus spread cost (0.1% on remaining 50%)
                    partial_pnl = (tp1_profit_pct * 0.5) - (0.1 * 0.5)  # 50% at TP1 profit, 50% at -0.1% spread
                    total_pnl += partial_pnl
                    partial_success += 1

                elif signal.hit == "FAILURE":
                    # Stop loss hit - calculate loss percentage
                    if signal.side == "LONG":
                        loss_pct = (signal.stop_loss - signal.price) / signal.price * 100
                    else:  # SHORT
                        loss_pct = (signal.price - signal.stop_loss) / signal.price * 100

                    total_pnl += loss_pct
                    failures += 1

            # Calculate metrics
            win_rate = ((full_success + partial_success) / total * 100) if total else 0.0
            avg_pnl_per_trade = total_pnl / total if total else 0.0

            # Format the summary line in the new compact format
            lines.append(
                f"{label}: {full_success}W/{partial_success}BE/{failures}L "
                f"({win_rate:.1f}%) | "
                f"P&L: {total_pnl:+.1f}% ({avg_pnl_per_trade:+.1f}% per trade)"
            )

    # Add legend at the end
    lines.extend(["", "Legend: W=Win(TP2) BE=Breakeven(TP1) L=Loss"])

    summary = "\n".join(lines)
    logger.info(f"Trading performance summary:\n{summary}")
    return summary


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def save_signal(signal: Dict) -> None:
    """Insert a new Signal row, skipping if DB disabled."""
    if not DB_ENABLED:
        return

    # Clean up any numpy types in the payload
    cleaned = {k: _to_native(v) for k, v in signal.items()}

    # default fields
    cleaned.setdefault("hit", "PENDING")
    cleaned.setdefault("hit_timestamp", None)

    try:
        with Session(engine) as session:
            s = Signal(**cleaned)
            session.add(s)
            session.commit()
            logger.info(f"Saved signal: {s.id} | {s.signal_uuid} | {s.pair} {s.timeframe} {s.side}")
    except SQLAlchemyError as e:
        logger.error(f"DB error saving signal: {e}")


def save_market_analysis(analysis_data: Dict) -> None:
    """Insert a new MarketAnalysis row, skipping if DB disabled."""
    if not DB_ENABLED:
        return

    # Clean up any numpy types in the payload
    cleaned = {k: _to_native(v) for k, v in analysis_data.items()}

    try:
        with Session(engine) as session:
            analysis = MarketAnalysis(**cleaned)
            session.add(analysis)
            session.commit()
            logger.debug(f"Saved market analysis: {analysis.pair} {analysis.timeframe} - {analysis.regime}")
    except SQLAlchemyError as e:
        logger.error(f"DB error saving market analysis: {e}")


def has_recent_pending(pair: str, timeframe: str, cooldown_minutes: int, session: Session) -> bool:
    """
    Returns True if there is any PENDING signal for (pair, timeframe)
    in the last `cooldown_minutes` minutes.
    """
    if not DB_ENABLED:
        return False

    cutoff = datetime.datetime.now(datetime.UTC) - timedelta(minutes=cooldown_minutes)
    q = (
        select(Signal.id)
        .where(
            Signal.pair == pair,
            Signal.timeframe == timeframe,
            Signal.hit == "PENDING",
            Signal.timestamp >= cutoff,
        )
        .limit(1)
    )
    return session.execute(q).first() is not None