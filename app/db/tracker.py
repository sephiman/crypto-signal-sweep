import logging
from datetime import timedelta
from typing import Dict

import ccxt
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Base

logger = logging.getLogger(__name__)
import datetime
from sqlalchemy.orm import Session
from app.config import DB_ENABLED, DB_URL
from app.db.models import Signal, MarketAnalysis

engine = None
if DB_ENABLED:
    engine = create_engine(DB_URL, echo=False, future=True)
    Base.metadata.create_all(engine)

exchange = ccxt.binance()


def _to_native(obj):
    """Convert numpy types to native Python types for SQLAlchemy."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def save_signal(signal: Dict):
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


def _get_price_extremes(ohlcv):
    """Extract price extremes from OHLCV data."""
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    return max(highs), min(lows)


def _check_tp1_hit(signal, high, low):
    """Check if TP1 is hit for a signal."""
    if signal.side == "LONG":
        return high >= signal.take_profit_1
    else:  # SHORT
        return low <= signal.take_profit_1


def _check_tp2_hit(signal, high, low):
    """Check if TP2 is hit for a signal."""
    if signal.side == "LONG":
        return high >= signal.take_profit_2
    else:  # SHORT
        return low <= signal.take_profit_2


def _check_stop_loss_hit(signal, high, low):
    """Check if original stop loss is hit."""
    if signal.side == "LONG":
        return low <= signal.stop_loss
    else:  # SHORT
        return high >= signal.stop_loss


def _check_breakeven_hit(signal, high, low):
    """Check if breakeven stop loss (entry price) is hit."""
    if signal.side == "LONG":
        return low <= signal.price
    else:  # SHORT
        return high >= signal.price


def _create_update_record(signal, current_price, hit_type, now, action):
    """Create an update record for signal status changes."""
    return {
        "signal_uuid": signal.signal_uuid,
        "pair": signal.pair,
        "timeframe": signal.timeframe,
        "side": signal.side,
        "price": current_price,
        "hit": hit_type,
        "hit_timestamp": now,
        "action": action
    }


def _handle_pending_signal(signal, high, low, current_price, now):
    """Handle a signal in PENDING status."""
    update = None

    # Check TP1 first (higher priority)
    if _check_tp1_hit(signal, high, low):
        signal.hit = "TP1_HIT"
        signal.sl_moved_to_be = True
        signal.hit_timestamp = now
        action = f"TP1 hit at {signal.take_profit_1:.6f}, SL moved to breakeven at {signal.price:.6f}"
        update = _create_update_record(signal, current_price, "TP1_HIT", now, action)
        logger.info(
            f"TP1 HIT: {signal.pair} {signal.side} - TP1: {signal.take_profit_1:.6f}, SL moved to BE: {signal.price:.6f}")

    # Check original SL (only if TP1 not hit)
    elif _check_stop_loss_hit(signal, high, low):
        signal.hit = "FAILURE"
        signal.hit_timestamp = now
        action = f"Stop Loss hit at {signal.stop_loss:.6f}"
        update = _create_update_record(signal, current_price, "FAILURE", now, action)
        logger.info(f"SL HIT: {signal.pair} {signal.side} - SL: {signal.stop_loss:.6f}")

    return update


def _handle_tp1_hit_signal(signal, high, low, current_price, now):
    """Handle a signal that already hit TP1."""
    update = None

    # Check TP2 first (higher priority)
    if _check_tp2_hit(signal, high, low):
        signal.hit = "SUCCESS"
        signal.hit_timestamp = now
        action = f"TP2 hit at {signal.take_profit_2:.6f} (Full profit)"
        update = _create_update_record(signal, current_price, "SUCCESS", now, action)
        logger.info(f"TP2 HIT: {signal.pair} {signal.side} - TP2: {signal.take_profit_2:.6f}")

    # Check breakeven SL (only if TP2 not hit)
    elif _check_breakeven_hit(signal, high, low):
        signal.hit = "BREAKEVEN"
        signal.hit_timestamp = now
        action = f"Breakeven SL hit at {signal.price:.6f} (Partial profit secured)"
        update = _create_update_record(signal, current_price, "BREAKEVEN", now, action)
        logger.info(f"BREAKEVEN: {signal.pair} {signal.side} - BE SL: {signal.price:.6f}")

    return update


def check_hit_signals():
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

                # Extract price extremes
                high, low = _get_price_extremes(ohlcv)
                current_price = ohlcv[-1][4]  # Current close price

                # Handle signal based on its current status
                if signal.hit == "PENDING":
                    update = _handle_pending_signal(signal, high, low, current_price, now)
                elif signal.hit == "TP1_HIT":
                    update = _handle_tp1_hit_signal(signal, high, low, current_price, now)

                if update:
                    updated.append(update)

            except Exception as e:
                logger.error(f"Failed 1m candle check for {signal.pair}: {e}")

        session.commit()

    return updated


def summarize_and_notify():
    """Summary with dual TP system results and profit calculations (based on $100 positions)."""
    if not DB_ENABLED:
        return None

    now = datetime.datetime.now(datetime.UTC)
    periods = [
        ("24h", datetime.timedelta(days=1)),
        ("7d", datetime.timedelta(days=7)),
        ("30d", datetime.timedelta(days=30)),
    ]
    lines = ["📊 *Trading Performance Summary*"]

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
                lines.append(f"{label}: No completed trades")
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

            # Format the summary line
            pnl_color = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"

            lines.append(
                f"{label}: {full_success}🎯/{partial_success}⚖️/{failures}❌ "
                f"({win_rate:.1f}% wins)"
                f"\n  {pnl_color} P&L: {total_pnl:+.1f}% | "
                f"Avg: {avg_pnl_per_trade:+.1f}%/trade"
            )

    summary = "\n".join(lines)
    logger.info(f"Trading performance summary:\n{summary}")
    return summary


def has_recent_pending(pair: str, timeframe: str, cooldown_minutes: int, session) -> bool:
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


def save_market_analysis(analysis_data: Dict):
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
