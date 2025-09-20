import logging

from sqlalchemy.orm import Session

from app.analyzer.signals import analyze_market
from app.config import PAIRS, tf_to_minutes
from app.db.init_db import engine
from app.db.tracker import check_hit_signals, summarize_and_notify
from app.db.tracker import has_recent_pending, save_signal
from app.telegram_bot import send_alerts, send_tp1_alerts, send_signal_outcome_alerts
from app.exception_notifier import send_exception_notification

logger = logging.getLogger(__name__)


def run_analysis_job(timeframe):
    try:
        signals = analyze_market(PAIRS, timeframe)

        cooldown_mins = tf_to_minutes(timeframe) * 4 + 1

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
    except Exception as e:
        logger.error(f"Error in analysis job for {timeframe}", exc_info=True)
        send_exception_notification(e, f"run_analysis_job({timeframe})", f"Failed to process {timeframe} analysis")


def run_midnight_summary_job():
    try:
        summary = summarize_and_notify()
        if summary:
            send_alerts([{"side": "SUMMARY", "summary": summary}])
    except Exception as e:
        logger.error("Error in midnight summary job", exc_info=True)
        send_exception_notification(e, "run_midnight_summary_job", "Failed to generate daily summary")


def run_hit_polling_job():
    try:
        hit_updates = check_hit_signals()

        # Send separate alerts for TP1 hits and final outcomes
        if hit_updates:
            send_tp1_alerts(hit_updates)
            send_signal_outcome_alerts(hit_updates)
    except Exception as e:
        logger.error("Error in hit polling job", exc_info=True)
        send_exception_notification(e, "run_hit_polling_job", "Failed to check signal hits")
