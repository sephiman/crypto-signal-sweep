from app.analyzer.signals import analyze_market
from app.config import PAIRS
from app.db.tracker import save_signal, check_hit_signals, summarize_and_notify
from app.telegram_bot import send_alerts


def run_analysis_job(timeframe):
    signals = analyze_market(PAIRS, timeframe)
    for signal in signals:
        save_signal(signal)
    if signals:
        send_alerts(signals)


def run_midnight_summary_job():
    summary = summarize_and_notify()
    if summary:
        send_alerts([{"side": "SUMMARY", "summary": summary}])


def run_hit_polling_job():
    check_hit_signals()
