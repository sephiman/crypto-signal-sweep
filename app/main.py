import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import TIMEFRAMES, RUN_AT_START
from app.config import tf_to_minutes
from app.jobs import run_analysis_job, run_midnight_summary_job, run_hit_polling_job
from app.exception_notifier import setup_exception_notification

logging.basicConfig(
    level=logging.INFO,
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

if __name__ == "__main__":
    scheduler.start()
