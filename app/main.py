import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from app.config import TIMEFRAMES
from app.jobs import run_analysis_job, run_midnight_summary_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

scheduler = BlockingScheduler()

for tf in TIMEFRAMES:
    minutes = int(tf[:-1]) if tf.endswith("m") else int(tf[:-1]) * 60
    scheduler.add_job(lambda tf=tf: run_analysis_job(tf), 'interval', minutes=minutes, id=f"run_analysis_{tf}")
    run_analysis_job(tf)

scheduler.add_job(run_midnight_summary_job, 'cron', hour=0, minute=0, id="run_midnight_summary")

if __name__ == "__main__":
    scheduler.start()
