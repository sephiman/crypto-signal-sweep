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
        # Summary message - check length and split if needed
        summary_text = signals[0]['summary']
        MAX_MESSAGE_LENGTH = 4000  # Telegram's limit is 4096, use 4000 for safety

        if len(summary_text) <= MAX_MESSAGE_LENGTH:
            # Send as single message
            _send_telegram_message(chat_id, summary_text)
        else:
            # Split into multiple messages if too long
            lines = summary_text.split('\n')
            current_message = ""
            message_count = 0

            for line in lines:
                # Check if adding this line would exceed the limit
                if len(current_message) + len(line) + 1 > MAX_MESSAGE_LENGTH:
                    # Send current message and start new one
                    if current_message:
                        message_count += 1
                        _send_telegram_message(chat_id, current_message)
                        current_message = line + '\n'
                else:
                    current_message += line + '\n'

            # Send remaining message
            if current_message:
                message_count += 1
                _send_telegram_message(chat_id, current_message)
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

    MAX_MESSAGE_LENGTH = 3500  # Safe buffer under Telegram's 4096 limit

    batches = []
    current_batch = []
    current_length = len("🎯 *TP1 Hit - SL Moved to Breakeven*\n")

    for update in tp1_hits:
        update_text = (
            f"\n⚡ *{update['pair']}* | {update['timeframe']} | *{update['side']}*"
            f"\n💰 *Current Price:* {update['price']:.6f}"
            f"\n{update['action']}"
            f"\n⏰ {update['hit_timestamp']:%H:%M UTC}"
            f"\n🆔 `{update.get('signal_uuid', 'N/A')}`\n"
        )

        update_length = len(update_text)

        # Check if adding this update would exceed the limit
        if current_length + update_length > MAX_MESSAGE_LENGTH and current_batch:
            # Save current batch and start a new one
            batches.append(current_batch)
            current_batch = [update_text]
            current_length = len("🎯 *TP1 Hit - SL Moved to Breakeven*\n") + update_length
        else:
            # Add to current batch
            current_batch.append(update_text)
            current_length += update_length

    # Add the last batch if it has updates
    if current_batch:
        batches.append(current_batch)

    # Send each batch as a separate message
    for i, batch in enumerate(batches):
        header = f"🎯 *TP1 Hit - SL Moved to Breakeven* ({i+1}/{len(batches)})" if len(batches) > 1 else "🎯 *TP1 Hit - SL Moved to Breakeven*"
        text = header + "\n" + "\n".join(batch)
        _send_telegram_message(TELEGRAM_CHAT_ID, text)


def send_signal_outcome_alerts(hit_updates):
    """Send alerts for final signal outcomes (SUCCESS, FAILURE, BREAKEVEN)"""
    if not hit_updates:
        return

    # Filter for final outcomes
    final_outcomes = [update for update in hit_updates if update.get('hit') in ['SUCCESS', 'FAILURE', 'BREAKEVEN']]
    if not final_outcomes:
        return

    MAX_MESSAGE_LENGTH = 3500  # Safe buffer under Telegram's 4096 limit

    batches = []
    current_batch = []
    current_length = len("📊 *Signal Updates*\n")

    for update in final_outcomes:
        outcome_emoji = {
            'SUCCESS': '✅',
            'FAILURE': '❌',
            'BREAKEVEN': '⚖️'
        }.get(update['hit'], '📊')

        update_text = (
            f"\n{outcome_emoji} *{update['pair']}* | {update['timeframe']} | *{update['side']}*"
            f"\n💰 *Final Price:* {update['price']:.6f}"
            f"\n📝 *Outcome:* {update['hit']}"
            f"\n{update['action']}"
            f"\n⏰ {update['hit_timestamp']:%H:%M UTC}"
            f"\n🆔 `{update.get('signal_uuid', 'N/A')}`\n"
        )

        update_length = len(update_text)

        # Check if adding this update would exceed the limit
        if current_length + update_length > MAX_MESSAGE_LENGTH and current_batch:
            # Save current batch and start a new one
            batches.append(current_batch)
            current_batch = [update_text]
            current_length = len("📊 *Signal Updates*\n") + update_length
        else:
            # Add to current batch
            current_batch.append(update_text)
            current_length += update_length

    # Add the last batch if it has updates
    if current_batch:
        batches.append(current_batch)

    # Send each batch as a separate message
    for i, batch in enumerate(batches):
        header = f"📊 *Signal Updates* ({i+1}/{len(batches)})" if len(batches) > 1 else "📊 *Signal Updates*"
        text = header + "\n" + "\n".join(batch)
        _send_telegram_message(TELEGRAM_CHAT_ID, text)


def send_backtest_summary(run_id, mode, start_date, end_date, pairs, timeframes,
                          total_trades, winners, losers, tp1_wins, win_rate,
                          total_pnl, avg_pnl, execution_time, config_snapshot):
    """
    Send backtest completion summary with results and configuration snapshot.

    Args:
        run_id: Backtest run ID
        mode: Backtest mode (sequential/parallel/one_pair_at_a_time)
        start_date: Start date of backtest
        end_date: End date of backtest
        pairs: List of pairs tested
        timeframes: List of timeframes tested
        total_trades: Total number of trades
        winners: Number of TP2 hits
        losers: Number of SL hits
        tp1_wins: Number of TP1 hits
        win_rate: Win rate percentage
        total_pnl: Total PnL percentage
        avg_pnl: Average PnL per trade
        execution_time: Execution time string (e.g., "1:23:45")
        config_snapshot: Dict with configuration parameters
    """
    import json

    # Format mode name
    mode_display = {
        'sequential': 'Sequential (All Pairs)',
        'parallel': f'Parallel ({config_snapshot.get("parallel_workers", "?")} workers)',
        'one_pair_at_a_time': 'One-Pair-at-a-Time'
    }.get(mode, mode.title())

    # Build results section
    lines = [
        "🎯 *BACKTEST COMPLETED*",
        "",
        "*Results:*",
        f"Mode: {mode_display}",
        f"Run ID: `{run_id}`",
        f"Period: {start_date} to {end_date}",
        f"Pairs: {len(pairs)} | Timeframes: {', '.join(timeframes)}",
        f"Total Trades: {total_trades}",
        f"  ✅ TP2: {winners} | 🎯 TP1: {tp1_wins} | ❌ SL: {losers}",
        f"Win Rate: {win_rate:.1f}%",
        f"Total PnL: {total_pnl:.2f}%",
        f"Avg PnL/Trade: {avg_pnl:.2f}%",
        f"Execution Time: {execution_time}",
        "",
        "*Configuration Snapshot:*"
    ]

    # Add key configuration parameters
    cfg = config_snapshot

    # TP/SL Settings
    lines.extend([
        "",
        "*TP/SL Settings:*",
        f"ATR Period: {cfg.get('atr_period', 'N/A')}",
        f"SL Multiplier: {cfg.get('atr_sl_multiplier', 'N/A')}",
        f"TP Multiplier: {cfg.get('atr_tp_multiplier', 'N/A')}"
    ])

    # RSI Settings
    lines.extend([
        "",
        "*RSI Settings:*",
        f"Period: {cfg.get('rsi_period', 'N/A')}",
        f"Oversold/Overbought: {cfg.get('rsi_oversold', 'N/A')}/{cfg.get('rsi_overbought', 'N/A')}",
        f"Momentum: {cfg.get('rsi_momentum', 'N/A')}",
        f"Trending Mode: {cfg.get('rsi_trending_mode', 'N/A')}",
    ])

    if cfg.get('rsi_trending_mode'):
        lines.extend([
            f"Trending OS/OB: {cfg.get('rsi_trending_oversold', 'N/A')}/{cfg.get('rsi_trending_overbought', 'N/A')}",
            f"Pullback L/S: {cfg.get('rsi_trending_pullback_long', 'N/A')}/{cfg.get('rsi_trending_pullback_short', 'N/A')}"
        ])

    # MACD Settings
    lines.extend([
        "",
        "*MACD Settings:*",
        f"Fast/Slow/Signal: {cfg.get('macd_fast', 'N/A')}/{cfg.get('macd_slow', 'N/A')}/{cfg.get('macd_signal', 'N/A')}",
        f"Min Diff: {cfg.get('macd_min_diff_pct', 'N/A')}% (Enabled: {cfg.get('macd_min_diff_enabled', False)})"
    ])

    # EMA Settings
    lines.extend([
        "",
        "*EMA Settings:*",
        f"Fast/Slow: {cfg.get('ema_fast', 'N/A')}/{cfg.get('ema_slow', 'N/A')}",
        f"Min Diff Enabled: {cfg.get('ema_min_diff_enabled', False)}"
    ])

    # ADX Settings
    lines.extend([
        "",
        "*ADX Settings:*",
        f"Period: {cfg.get('adx_period', 'N/A')}",
        f"Threshold: {cfg.get('adx_threshold', 'N/A')}",
        f"RSI Mode: {cfg.get('adx_rsi_mode', 'N/A')}"
    ])

    # Optional Indicators
    if cfg.get('stoch_enabled'):
        lines.extend([
            "",
            "*Stochastic:*",
            f"K/D Period: {cfg.get('stoch_k_period', 'N/A')}/{cfg.get('stoch_d_period', 'N/A')}",
            f"OS/OB: {cfg.get('stoch_oversold', 'N/A')}/{cfg.get('stoch_overbought', 'N/A')}"
        ])

    if cfg.get('bb_enabled'):
        lines.extend([
            "",
            "*Bollinger Bands:*",
            f"Period/StdDev: {cfg.get('bb_period', 'N/A')}/{cfg.get('bb_std_dev', 'N/A')}",
            f"Min Width: {cfg.get('bb_width_min', 'N/A')}"
        ])

    # Filters and Confirmations
    lines.extend([
        "",
        "*Filters & Confirmations:*",
        f"Min ATR Ratio: {cfg.get('min_atr_ratio', 'N/A')}",
        f"Volume Confirm: {cfg.get('volume_confirmation_enabled', False)} (Min: {cfg.get('min_volume_ratio', 'N/A')}x)",
        f"Higher TF Confirm: {cfg.get('use_higher_tf_confirm', False)}",
        f"Trend Filter: {cfg.get('use_trend_filter', False)} (MA Period: {cfg.get('trend_ma_period', 'N/A')})"
    ])

    # Scoring
    lines.extend([
        "",
        "*Scoring:*",
        f"Send Unconfirmed: {cfg.get('send_unconfirmed', False)}",
        f"Dynamic Score: {cfg.get('dynamic_score_enabled', False)}",
        f"Min Score Default: {cfg.get('min_score_default', 'N/A')}",
        f"Min Score Trending/Ranging: {cfg.get('min_score_trending', 'N/A')}/{cfg.get('min_score_ranging', 'N/A')}"
    ])

    # Time Filter
    if cfg.get('time_filter_enabled'):
        lines.extend([
            "",
            "*Time Filter:*",
            f"Timezone: {cfg.get('time_filter_timezone', 'N/A')}",
            f"Avoid Hours: {cfg.get('avoid_hours_start', 'N/A')}-{cfg.get('avoid_hours_end', 'N/A')}"
        ])

    # Join all lines
    summary_text = "\n".join(lines)

    # Send via Telegram (use market chat ID for summaries)
    send_alerts([{"summary": summary_text}], chat_id=TELEGRAM_MARKET_CHAT_ID)
