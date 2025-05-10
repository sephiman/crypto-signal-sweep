import datetime

import ccxt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import DB_URL, DB_ENABLED
from app.db.models import Signal

engine = create_engine(DB_URL) if DB_ENABLED else None

exchange = ccxt.binance()


def save_signal(signal):
    if not DB_ENABLED:
        return
    with Session(engine) as session:
        s = Signal(**signal, hit="PENDING")
        session.add(s)
        session.commit()


def check_hit_signals():
    if not DB_ENABLED:
        return
    with Session(engine) as session:
        signals = session.query(Signal).filter(Signal.hit == "PENDING").all()
        for s in signals:
            current_price = exchange.fetch_ticker(s.pair)['last']
            if s.side == "LONG" and current_price >= s.take_profit:
                s.hit = "SUCCESS"
            elif s.side == "LONG" and current_price <= s.stop_loss:
                s.hit = "FAILURE"
            elif s.side == "SHORT" and current_price <= s.take_profit:
                s.hit = "SUCCESS"
            elif s.side == "SHORT" and current_price >= s.stop_loss:
                s.hit = "FAILURE"
            else:
                continue
            s.hit_timestamp = datetime.datetime.utcnow()
        session.commit()


def summarize_and_notify():
    if not DB_ENABLED:
        return
    today = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    with Session(engine) as session:
        success = session.query(Signal).filter(Signal.hit == "SUCCESS", Signal.hit_timestamp > today).count()
        fail = session.query(Signal).filter(Signal.hit == "FAILURE", Signal.hit_timestamp > today).count()
        return f"Summary of last 24h: {success} SUCCESS / {fail} FAILURE"
