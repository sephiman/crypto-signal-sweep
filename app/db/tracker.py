import datetime
import logging

import ccxt

from app.config import DB_ENABLED
from app.db.init_db import SessionLocal, init_db
from app.db.models import Signal

if DB_ENABLED:
    init_db()

logger = logging.getLogger(__name__)
exchange = ccxt.binance()


def save_signal(signal_data):
    if not DB_ENABLED:
        return
    session = SessionLocal()
    try:
        s = Signal(**signal_data)
        session.add(s)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"DB error saving signal: {e}")
    finally:
        session.close()


def check_hit_signals():
    if not DB_ENABLED:
        return
    session = SessionLocal()
    try:
        pending = session.query(Signal).filter(Signal.hit == 'PENDING').all()
        for s in pending:
            current = exchange.fetch_ticker(s.pair)['last']
            if s.side == 'LONG':
                if current >= s.take_profit:
                    s.hit = 'SUCCESS'
                elif current <= s.stop_loss:
                    s.hit = 'FAILURE'
            else:
                if current <= s.take_profit:
                    s.hit = 'SUCCESS'
                elif current >= s.stop_loss:
                    s.hit = 'FAILURE'
            if s.hit != 'PENDING':
                s.hit_timestamp = datetime.datetime.utcnow()
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"DB error checking hits: {e}")
    finally:
        session.close()


def summarize_and_notify():
    # returns counts for last 24h
    if not DB_ENABLED:
        return 'DB disabled'
    now = datetime.datetime.utcnow()
    since = now - datetime.timedelta(days=1)
    session = SessionLocal()
    try:
        succ = session.query(Signal).filter(
            Signal.hit == 'SUCCESS', Signal.hit_timestamp >= since
        ).count()
        fail = session.query(Signal).filter(
            Signal.hit == 'FAILURE', Signal.hit_timestamp >= since
        ).count()
        return f"Last 24h: {succ} SUCCESS / {fail} FAILURE"
    except Exception as e:
        logger.error(f"DB error summarizing: {e}")
        return "Error"
    finally:
        session.close()
