import logging

from sqlalchemy.orm import Session

from app.analyzer.signals import analyze_market
from app.config import PAIRS, tf_to_minutes
from app.db.init_db import engine
from app.db.tracker import check_hit_signals, summarize_and_notify
from app.db.tracker import has_recent_pending, save_signal
from app.telegram_bot import send_alerts

logger = logging.getLogger(__name__)


def run_analysis_job(timeframe):
    signals = analyze_market(PAIRS, timeframe)

    cooldown_mins = tf_to_minutes(timeframe) * 2

    with Session(engine) as session:
        to_alert = []
        for sig in signals:
            if has_recent_pending(sig["pair"], sig["timeframe"], cooldown_mins, session):
                logger.info(f"⏸ DB cooldown: skipping {sig['pair']}@{sig['timeframe']}")
                continue

            save_signal(sig)
            to_alert.append(sig)

        if to_alert:
            send_alerts(to_alert)


def run_midnight_summary_job():
    summary = summarize_and_notify()
    if summary:
        send_alerts([{"side": "SUMMARY", "summary": summary}])


def run_hit_polling_job():
    check_hit_signals()
