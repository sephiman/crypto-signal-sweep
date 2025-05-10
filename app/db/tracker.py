import logging
from datetime import datetime
from typing import Dict

import ccxt
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
            logger.info(f"Saved signal: {s.id} | {s.pair} {s.timeframe} {s.side}")
    except SQLAlchemyError as e:
        logger.error(f"DB error saving signal: {e}")


def check_hit_signals():
    """
    Scan all PENDING signals, fetch each pair's last price exactly once,
    mark those that have hit SL/TP, and return the ones we just flipped.
    """
    if not DB_ENABLED:
        return []

    updated = []
    now = datetime.datetime.utcnow()

    with Session(engine) as session:
        # 1) load all pending signals
        pending = session.query(Signal).filter_by(hit="PENDING").all()
        if not pending:
            return []

        # 2) build a unique list of pairs, fetch each price once
        pairs = {s.pair for s in pending}
        prices = {}
        for pair in pairs:
            try:
                prices[pair] = exchange.fetch_ticker(pair)["last"]
            except Exception as e:
                logger.error(f"Failed to fetch price for {pair}: {e}")
                prices[pair] = None

        # 3) iterate signals and compare against our one‐time fetch
        for s in pending:
            current = prices.get(s.pair)
            if current is None:
                continue

            new_hit = None
            if s.side == "LONG":
                if current >= s.take_profit:
                    new_hit = "SUCCESS"
                elif current <= s.stop_loss:
                    new_hit = "FAILURE"
            else:  # SHORT
                if current <= s.take_profit:
                    new_hit = "SUCCESS"
                elif current >= s.stop_loss:
                    new_hit = "FAILURE"

            if new_hit:
                s.hit = new_hit
                s.hit_timestamp = now
                updated.append({
                    "pair": s.pair,
                    "timeframe": s.timeframe,
                    "side": s.side,
                    "price": current,
                    "hit": new_hit,
                    "hit_timestamp": now,
                })

        session.commit()

    return updated


def summarize_and_notify():
    """Return a 24h summary string for Telegram (or None if DB disabled)."""
    if not DB_ENABLED:
        return None

    from datetime import timedelta

    cutoff = datetime.datetime.now(datetime.UTC) - timedelta(days=1)
    with Session(engine) as session:
        succ = session.query(Signal).filter(Signal.hit == "SUCCESS", Signal.hit_timestamp > cutoff).count()
        fail = session.query(Signal).filter(Signal.hit == "FAILURE", Signal.hit_timestamp > cutoff).count()

    summary = f"Last 24h: {succ} ✅, {fail} ❌"
    logger.info(f"Daily summary: {summary}")
    return summary
