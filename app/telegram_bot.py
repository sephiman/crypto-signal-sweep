# app/telegram_bot.py
import logging

import requests

from app.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_alerts(signals):
    if not signals:
        return

    if len(signals) == 1 and "summary" in signals[0]:
        text = f"📊 *Daily Summary*\n{signals[0]['summary']}"
    else:
        lines = ["🚨 *Signal Alert*"]
        for s in signals:
            if not all(k in s for k in ("pair", "timeframe", "side")):
                continue

            # Calculate risk-reward ratio
            risk = abs(s['price'] - s['stop_loss'])
            reward = abs(s['take_profit'] - s['price'])
            rr_ratio = reward / risk if risk > 0 else 0

            confidence_emoji = "🔥" if s.get('confidence') == 'HIGH' else "⚡"

            lines.append(
                f"\n{confidence_emoji} *{s['pair']}* | {s['timeframe']} | *{s['side']}*"
                f"\n💰 *Entry:* {s['price']:.4f}"
                f"\n🛑 *SL:* {s['stop_loss']:.4f} | 🎯 *TP:* {s['take_profit']:.4f}"
                f"\n📊 *RR:* {rr_ratio:.1f}:1 | *Score:* {s.get('score', '?')}/{s.get('required_score', '?')}"
                f"\n📈 *RSI:* {s.get('rsi', 0):.1f} | *ADX:* {s.get('adx', 0):.1f}"
                f"\n🔄 *Volume:* {s.get('volume_ratio', 1.0):.1f}x | *Confidence:* {s.get('confidence', 'MEDIUM')}"
                f"\n⏰ {s['timestamp']:%H:%M UTC}\n"
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
