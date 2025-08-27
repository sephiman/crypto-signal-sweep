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

            # Calculate risk-reward ratios for both TP levels
            risk = abs(s['price'] - s['stop_loss'])
            reward_tp1 = abs(s['take_profit_1'] - s['price'])
            reward_tp2 = abs(s['take_profit_2'] - s['price'])
            rr_ratio_tp1 = reward_tp1 / risk if risk > 0 else 0
            rr_ratio_tp2 = reward_tp2 / risk if risk > 0 else 0

            confidence_emoji = "🔥" if s.get('confidence') == 'HIGH' else "⚡"

            tp_text = f"🎯 *TP1:* {s['take_profit_1']:.6f} | *TP2:* {s['take_profit_2']:.6f}"
            rr_text = f"📊 *RR:* {rr_ratio_tp1:.1f}:1 / {rr_ratio_tp2:.1f}:1"
            strategy_note = "\n💡 *Strategy:* Partial profit at TP1, SL to BE"

            lines.append(
                f"\n{confidence_emoji} *{s['pair']}* | {s['timeframe']} | *{s['side']}*"
                f"\n💰 *Entry:* {s['price']:.6f}"
                f"\n🛑 *SL:* {s['stop_loss']:.6f} | {tp_text}"
                f"\n{rr_text} | *Score:* {s.get('score', '?')}/{s.get('required_score', '?')}"
                f"\n📈 *RSI:* {s.get('rsi', 0):.1f} | *ADX:* {s.get('adx', 0):.1f}"
                f"\n🔄 *Volume:* {s.get('volume_ratio', 1.0):.1f}x | *Confidence:* {s.get('confidence', 'MEDIUM')}{strategy_note}"
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


def send_tp1_alerts(hit_updates):
    """Send alerts when TP1 is hit and SL is moved to breakeven"""
    if not hit_updates:
        return
    
    # Filter for TP1 hits only
    tp1_hits = [update for update in hit_updates if update.get('hit') == 'TP1_HIT']
    if not tp1_hits:
        return

    lines = ["🎯 *TP1 Hit - SL Moved to Breakeven*"]
    
    for update in tp1_hits:
        lines.append(
            f"\n⚡ *{update['pair']}* | {update['timeframe']} | *{update['side']}*"
            f"\n💰 *Current Price:* {update['price']:.6f}"
            f"\n{update['action']}"
            f"\n⏰ {update['hit_timestamp']:%H:%M UTC}\n"
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
        logger.info(f"[Telegram] TP1 alert sent successfully: {resp.status_code}")
    except Exception as e:
        logger.error(f"[Telegram] Failed to send TP1 alert: {e}")


def send_signal_outcome_alerts(hit_updates):
    """Send alerts for final signal outcomes (SUCCESS, FAILURE, BREAKEVEN)"""
    if not hit_updates:
        return
    
    # Filter for final outcomes
    final_outcomes = [update for update in hit_updates if update.get('hit') in ['SUCCESS', 'FAILURE', 'BREAKEVEN']]
    if not final_outcomes:
        return

    lines = ["📊 *Signal Updates*"]
    
    for update in final_outcomes:
        outcome_emoji = {
            'SUCCESS': '✅',
            'FAILURE': '❌', 
            'BREAKEVEN': '⚖️'
        }.get(update['hit'], '📊')
        
        lines.append(
            f"\n{outcome_emoji} *{update['pair']}* | {update['timeframe']} | *{update['side']}*"
            f"\n💰 *Final Price:* {update['price']:.6f}"
            f"\n📝 *Outcome:* {update['hit']}"
            f"\n{update['action']}"
            f"\n⏰ {update['hit_timestamp']:%H:%M UTC}\n"
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
        logger.info(f"[Telegram] Signal outcome alerts sent successfully: {resp.status_code}")
    except Exception as e:
        logger.error(f"[Telegram] Failed to send signal outcome alerts: {e}")
