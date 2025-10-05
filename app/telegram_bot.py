# app/telegram_bot.py
import logging

import requests

from app.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_MARKET_CHAT_ID
from app.db.tracker import get_pair_winrate

logger = logging.getLogger(__name__)


def send_alerts(signals, chat_id=None):
    if not signals:
        return

    # Use market chat ID for summary messages, default chat ID otherwise
    if chat_id is None:
        if len(signals) == 1 and "summary" in signals[0]:
            chat_id = TELEGRAM_MARKET_CHAT_ID
        else:
            chat_id = TELEGRAM_CHAT_ID

    if len(signals) == 1 and "summary" in signals[0]:
        # Summary message - send as single message
        _send_telegram_message(chat_id, signals[0]['summary'])
    else:
        # Signal alerts - batch to avoid hitting 4096 character limit
        MAX_MESSAGE_LENGTH = 3500  # Safe buffer under Telegram's 4096 limit

        batches = []
        current_batch = []
        current_length = len("🚨 *Signal Alert*\n")

        for s in signals:
            if not all(k in s for k in ("pair", "timeframe", "side")):
                continue

            # Build signal text
            risk = abs(s['price'] - s['stop_loss'])
            reward_tp1 = abs(s['take_profit_1'] - s['price'])
            reward_tp2 = abs(s['take_profit_2'] - s['price'])
            rr_ratio_tp1 = reward_tp1 / risk if risk > 0 else 0
            rr_ratio_tp2 = reward_tp2 / risk if risk > 0 else 0

            confidence_emoji = "🔥" if s.get('confidence') == 'HIGH' else "⚡"
            winrate = get_pair_winrate(s['pair'])
            winrate_text = f" | *WR:* {winrate:.1f}%" if winrate is not None else ""

            tp_text = f"🎯 *TP1:* {s['take_profit_1']:.6f} | *TP2:* {s['take_profit_2']:.6f}"
            rr_text = f"📊 *RR:* {rr_ratio_tp1:.1f}:1 / {rr_ratio_tp2:.1f}:1"
            strategy_note = "\n💡 *Strategy:* Partial profit at TP1, SL to BE"

            signal_text = (
                f"\n{confidence_emoji} *{s['pair']}* | {s['timeframe']} | *{s['side']}*"
                f"\n💰 *Entry:* {s['price']:.6f}"
                f"\n🛑 *SL:* {s['stop_loss']:.6f} | {tp_text}"
                f"\n{rr_text} | *Score:* {s.get('score', '?')}/{s.get('required_score', '?')}{winrate_text}"
                f"\n📈 *RSI:* {s.get('rsi', 0):.1f} | *ADX:* {s.get('adx', 0):.1f}"
                f"\n🔄 *Volume:* {s.get('volume_ratio', 1.0):.1f}x | *Confidence:* {s.get('confidence', 'MEDIUM')}{strategy_note}"
                f"\n⏰ {s['timestamp']:%H:%M UTC}"
                f"\n🆔 `{s.get('signal_uuid', 'N/A')}`\n"
            )

            signal_length = len(signal_text)

            # Check if adding this signal would exceed the limit
            if current_length + signal_length > MAX_MESSAGE_LENGTH and current_batch:
                # Save current batch and start a new one
                batches.append(current_batch)
                current_batch = [signal_text]
                current_length = len("🚨 *Signal Alert*\n") + signal_length
            else:
                # Add to current batch
                current_batch.append(signal_text)
                current_length += signal_length

        # Add the last batch if it has signals
        if current_batch:
            batches.append(current_batch)

        # Send each batch as a separate message
        for i, batch in enumerate(batches):
            header = f"🚨 *Signal Alert* ({i+1}/{len(batches)})" if len(batches) > 1 else "🚨 *Signal Alert*"
            text = header + "\n" + "\n".join(batch)
            _send_telegram_message(chat_id, text)


def _send_telegram_message(chat_id, text):
    """Helper function to send a single Telegram message"""
    payload = {
        "chat_id": chat_id,
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
            f"\n⏰ {update['hit_timestamp']:%H:%M UTC}"
            f"\n🆔 `{update.get('signal_uuid', 'N/A')}`\n"
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
            f"\n⏰ {update['hit_timestamp']:%H:%M UTC}"
            f"\n🆔 `{update.get('signal_uuid', 'N/A')}`\n"
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
