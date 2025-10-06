import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import (
    TIMEFRAMES, RUN_AT_START, MARKET_SUMMARY_ENABLED,
    BACKTEST_MODE, COLLECT_HISTORICAL_DATA, PAIRS, BACKTEST_LOG_LEVEL
)
from app.config import tf_to_minutes
from app.jobs import run_analysis_job, run_midnight_summary_job, run_hit_polling_job, run_market_summary_job
from app.exception_notifier import setup_exception_notification

# Set logging level based on mode (reduced logging for backtest)
log_level = getattr(logging, BACKTEST_LOG_LEVEL) if BACKTEST_MODE else logging.INFO

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize exception notification system
setup_exception_notification()

scheduler = BlockingScheduler()

# Schedule analysis jobs at the end of each closed candle + 5s buffer
for tf in TIMEFRAMES:
    minutes = tf_to_minutes(tf)
    if tf.endswith("m"):
        # every N minutes at second=5
        trigger = CronTrigger(minute=f"*/{minutes}", second="5")
    else:
        hours = minutes // 60
        trigger = CronTrigger(hour=f"*/{hours}", minute="0", second="5")

    scheduler.add_job(
        lambda tf=tf: run_analysis_job(tf),
        trigger,
        id=f"run_analysis_{tf}"
    )
    if RUN_AT_START:
        run_analysis_job(tf)

if RUN_AT_START:
    run_midnight_summary_job()
    run_market_summary_job()

# Poll for SL/TP hits once per minute (no need to align to candles)
scheduler.add_job(
    run_hit_polling_job,
    "interval",
    minutes=1,
    id="run_hit_polling"
)

# Daily summary at UTC midnight + 10s buffer
scheduler.add_job(
    run_midnight_summary_job,
    trigger=CronTrigger(hour=0, minute=0, second="10"),
    id="run_midnight_summary"
)

# Hourly market summary at 4 minutes past the hour (X:04)
if MARKET_SUMMARY_ENABLED:
    scheduler.add_job(
        run_market_summary_job,
        trigger=CronTrigger(minute=4, second="0"),
        id="run_market_summary"
    )

if __name__ == "__main__":
    if BACKTEST_MODE:
        # BACKTEST MODE
        logging.info("Starting in BACKTEST mode")

        # Collect historical data if requested
        if COLLECT_HISTORICAL_DATA:
            logging.info("Collecting historical data...")
            from backtest.data_collector import collect_historical_data
            collect_historical_data(PAIRS, days=365)
            logging.info("Historical data collection completed")

        # Run backtest
        logging.info("Running backtest...")
        from backtest.engine import run_backtest
        run_backtest()

        # Analyze results
        logging.info("Analyzing backtest results...")
        from backtest.analyzer import analyze_backtest
        analyze_backtest()

        logging.info("Backtest mode completed, exiting")

    else:
        # LIVE MODE
        logging.info("Starting in LIVE mode")
        scheduler.start()
