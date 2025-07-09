# app/telegram_bot.py
import logging

import requests

from app.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_alerts(signals):
    if not signals:
        return

    if len(signals) == 1 and "summary" in signals[0]:
        text = (
            "📊 *Daily Summary*\n"
            f"{signals[0]['summary']}"
        )
    else:
        lines = ["📡 *Signal Alert*"]
        for s in signals:
            if not all(k in s for k in ("pair", "timeframe", "side")):
                continue

            lines.append(
                f"\n*Pair:* {s['pair']}"
                f"\n*Timeframe:* {s['timeframe']}"
                f"\n*Side:* {s['side']}"
                f"\n*Price:* {s['price']:.4f}"
                f"\n*SL:* {s['stop_loss']:.4f}"
                f"\n*TP:* {s['take_profit']:.4f}"
                f"\n*Time:* {s['timestamp']:%Y-%m-%d %H:%M:%S} UTC"
                f"\n*Score:* {s.get('score', '?')}/{s.get('required_score', 5)}"
                f"\n*Momentum OK:* {s['momentum_ok']} | *Trend Confirmed:* {s['trend_confirmed']} | *HTF Confirmed:* {s['higher_tf_confirmed']} | *Signal confirmed:* {s['confirmed']}\n"
            )
        text = "\n".join(lines)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info(f"[Telegram] Message sent successfully: {resp.status_code}")
    except Exception as e:
        logger.error(f"[Telegram] Failed to send message: {e}")
