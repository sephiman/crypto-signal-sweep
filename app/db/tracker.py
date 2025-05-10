import logging
from datetime import datetime
from typing import Dict

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import DB_URL, DB_ENABLED
from app.db.models import Base, Signal

logger = logging.getLogger(__name__)

engine = None
if DB_ENABLED:
    engine = create_engine(DB_URL, echo=False, future=True)
    Base.metadata.create_all(engine)


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
            logger.info(f"Saved signal: {s.id} | {s.pair} {s.timeframe} {s.side}")
    except SQLAlchemyError as e:
        logger.error(f"DB error saving signal: {e}")


def check_hit_signals():
    """Scan pending signals; mark SUCCESS/FAILURE when SL/TP hit."""
    if not DB_ENABLED:
        return
    import ccxt  # local import to avoid startup cost
    exchange = ccxt.binance()

    with Session(engine) as session:
        pending = session.query(Signal).filter_by(hit="PENDING").all()
        for s in pending:
            ticker = exchange.fetch_ticker(s.pair)
            current = float(ticker["last"])
            if s.side == "LONG":
                if current >= s.take_profit:
                    s.hit = "SUCCESS"
                elif current <= s.stop_loss:
                    s.hit = "FAILURE"
            elif s.side == "SHORT":
                if current <= s.take_profit:
                    s.hit = "SUCCESS"
                elif current >= s.stop_loss:
                    s.hit = "FAILURE"
            if s.hit != "PENDING":
                s.hit_timestamp = datetime.utcnow()
                logger.info(f"Signal {s.id} {s.side} → {s.hit} at {current:.4f}")
        session.commit()


def summarize_and_notify():
    """Return a 24h summary string for Telegram (or None if DB disabled)."""
    if not DB_ENABLED:
        return None

    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(days=1)
    with Session(engine) as session:
        succ = session.query(Signal).filter(Signal.hit == "SUCCESS", Signal.hit_timestamp > cutoff).count()
        fail = session.query(Signal).filter(Signal.hit == "FAILURE", Signal.hit_timestamp > cutoff).count()

    summary = f"Last 24h: {succ} ✅, {fail} ❌"
    logger.info(f"Daily summary: {summary}")
    return summary
