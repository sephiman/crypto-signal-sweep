import logging
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.config import DB_ENABLED, PAIRS, TIMEFRAMES, get_adx_threshold_for_timeframe
from app.db.init_db import engine
from app.db.models import MarketAnalysis
from app.telegram_bot import send_alerts

logger = logging.getLogger(__name__)


@dataclass
class TrendData:
    """Container for trend analysis data"""
    direction: str  # "bullish", "bearish", "ranging", "volatile"
    emoji: str
    rsi: float
    score: int
    timestamp: datetime.datetime


@dataclass
class PairSummary:
    """Container for pair analysis summary"""
    pair: str
    overall_bias: str
    timeframes: Dict[str, TrendData]
    signal_readiness: str
    readiness_emoji: str
    latest_timestamp: datetime.datetime


def get_trend_direction(ema_fast: float, ema_slow: float, adx: float, timeframe: str) -> Tuple[str, str]:
    """
    Classify trend direction based on EMA and ADX values.
    Returns (direction, emoji) tuple.
    """
    adx_threshold = get_adx_threshold_for_timeframe(timeframe)

    if adx >= adx_threshold:
        # Trending market
        if ema_fast > ema_slow:
            return "bullish", "🟢"
        else:
            return "bearish", "🔴"
    else:
        # Non-trending market - check for volatility
        if adx >= (adx_threshold * 0.7):  # Moderate ADX with conflicting signals
            return "volatile", "🟡"
        else:
            return "ranging", "⚪"


def get_signal_readiness(score: int, required_score: int) -> Tuple[str, str]:
    """
    Determine signal readiness based on score.
    Returns (status, emoji) tuple.
    """
    if score >= required_score:
        return "READY", "🎯"
    elif score >= (required_score - 1):
        return "BUILDING", "⏳"
    else:
        return "NO SETUP", "❌"


def get_latest_market_analysis() -> List[PairSummary]:
    """
    Get the latest market analysis data for all pairs and timeframes.
    Returns list of PairSummary objects.
    """
    if not DB_ENABLED:
        logger.warning("Database disabled, cannot generate market summary")
        return []

    # Use configured timeframes from user input
    available_timeframes = TIMEFRAMES
    summaries = []

    # Calculate dynamic cutoff based on longest timeframe
    # Add buffer time (1.5x the longest timeframe) to ensure we have recent data
    def tf_to_hours(tf: str) -> float:
        """Convert timeframe string to hours."""
        if tf.endswith('m'):
            return int(tf[:-1]) / 60.0
        elif tf.endswith('h'):
            return int(tf[:-1])
        elif tf.endswith('d'):
            return int(tf[:-1]) * 24
        elif tf.endswith('w'):
            return int(tf[:-1]) * 24 * 7
        else:
            return 2.0  # Default fallback

    # Find the longest timeframe and calculate appropriate cutoff
    max_tf_hours = max([tf_to_hours(tf) for tf in available_timeframes])
    cutoff_hours = max(max_tf_hours * 1.5, 2.0)  # At least 2 hours, or 1.5x longest timeframe
    cutoff_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=cutoff_hours)


    with Session(engine) as session:
        for pair in PAIRS:
            timeframe_data = {}
            latest_timestamp = None

            for tf in available_timeframes:
                # Get most recent analysis for this pair/timeframe
                analysis = session.query(MarketAnalysis) \
                    .filter(
                        MarketAnalysis.pair == pair,
                        MarketAnalysis.timeframe == tf,
                        MarketAnalysis.timestamp >= cutoff_time
                    ) \
                    .order_by(desc(MarketAnalysis.timestamp)) \
                    .first()

                if analysis:
                    direction, emoji = get_trend_direction(
                        analysis.ema_fast,
                        analysis.ema_slow,
                        analysis.adx,
                        tf
                    )

                    # Use the higher score between long and short
                    score = max(analysis.long_score, analysis.short_score)

                    # Handle timezone-naive timestamps from database
                    timestamp = analysis.timestamp
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

                    timeframe_data[tf] = TrendData(
                        direction=direction,
                        emoji=emoji,
                        rsi=analysis.rsi,
                        score=score,
                        timestamp=timestamp
                    )

                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                else:
                    # No recent data available
                    timeframe_data[tf] = TrendData(
                        direction="unknown",
                        emoji="❓",
                        rsi=50.0,
                        score=0,
                        timestamp=cutoff_time
                    )

            if not timeframe_data:
                continue

            # Determine overall bias (weighted by timeframe importance)
            # 4h carries most weight, 1h medium, 15m least
            overall_bias = _calculate_overall_bias(timeframe_data)

            # Calculate signal readiness based on best timeframe score
            best_score = max([tf_data.score for tf_data in timeframe_data.values()])
            # Use a representative required score (can be refined)
            representative_required_score = 5
            readiness, readiness_emoji = get_signal_readiness(best_score, representative_required_score)

            summaries.append(PairSummary(
                pair=pair,
                overall_bias=overall_bias,
                timeframes=timeframe_data,
                signal_readiness=readiness,
                readiness_emoji=readiness_emoji,
                latest_timestamp=latest_timestamp or cutoff_time
            ))

    return summaries


def _calculate_overall_bias(timeframe_data: Dict[str, TrendData]) -> str:
    """
    Calculate overall market bias with dynamic timeframe weighting.
    Longer timeframes get higher weights.
    """
    # Define weights for different timeframes (longer = higher weight)
    weight_map = {
        "1m": 0.05,
        "5m": 0.1,
        "15m": 0.2,
        "1h": 0.3,
        "4h": 0.5,
        "1d": 0.7,
        "1w": 0.9
    }

    # Calculate weights only for available timeframes
    available_timeframes = list(timeframe_data.keys())
    total_weight = sum([weight_map.get(tf, 0.25) for tf in available_timeframes])

    # Normalize weights to sum to 1.0
    normalized_weights = {}
    for tf in available_timeframes:
        normalized_weights[tf] = weight_map.get(tf, 0.25) / total_weight

    bullish_weight = 0
    bearish_weight = 0

    for tf, weight in normalized_weights.items():
        if tf in timeframe_data:
            direction = timeframe_data[tf].direction
            if direction == "bullish":
                bullish_weight += weight
            elif direction == "bearish":
                bearish_weight += weight
            # ranging/volatile don't contribute to directional bias

    if bullish_weight > bearish_weight:
        if bullish_weight >= 0.6:
            return "STRONGLY BULLISH"
        else:
            return "BULLISH"
    elif bearish_weight > bullish_weight:
        if bearish_weight >= 0.6:
            return "STRONGLY BEARISH"
        else:
            return "BEARISH"
    else:
        return "NEUTRAL"


def format_market_summary_message(summaries: List[PairSummary]) -> str:
    """
    Format market summary data into a Telegram message.
    """
    if not summaries:
        return "📊 *MARKET SUMMARY*\n❌ No recent market data available"

    now = datetime.datetime.now(datetime.UTC)

    # Find the most recent data timestamp and handle timezone issues
    latest_data_time = max([s.latest_timestamp for s in summaries])

    # Ensure both timestamps are timezone-aware for comparison
    if latest_data_time.tzinfo is None:
        # If database timestamp is naive, assume it's UTC
        latest_data_time = latest_data_time.replace(tzinfo=datetime.timezone.utc)

    data_age_minutes = (now - latest_data_time).total_seconds() / 60

    # Header
    lines = [
        f"📊 *MARKET SUMMARY* - {now:%H:%M UTC}",
    ]

    # Data freshness warning
    if data_age_minutes > 30:
        lines.append(f"⚠️ Data is {data_age_minutes:.0f} minutes old")

    lines.append("")  # Empty line

    # Sort pairs by signal readiness (Ready first, then Building, then No Setup)
    readiness_order = {"READY": 0, "BUILDING": 1, "NO SETUP": 2}
    sorted_summaries = sorted(summaries, key=lambda x: (readiness_order.get(x.signal_readiness, 3), x.pair))

    # Format pairs - one per line
    for summary in sorted_summaries:
        # Remove "/USDT" suffix from pair name
        pair_name = summary.pair.replace("/USDT", "")

        # Compact timeframe details with timeframe labels
        tf_details = []
        for tf in sorted(summary.timeframes.keys(), key=lambda x: ['1m', '5m', '15m', '1h', '4h', '1d', '1w'].index(x) if x in ['1m', '5m', '15m', '1h', '4h', '1d', '1w'] else 999):
            if tf in summary.timeframes:
                tf_data = summary.timeframes[tf]
                tf_details.append(f"{tf}: {tf_data.emoji}{tf_data.rsi:.0f}")

        # Calculate best score for display
        best_score = max([tf_data.score for tf_data in summary.timeframes.values()])

        # Format: PAIR:  timeframe: trend|timeframe: trend  status(score)
        pair_line = f"*{pair_name}*:  {' | '.join(tf_details)} {summary.readiness_emoji}({best_score})"
        lines.append(pair_line)

    # Legend
    lines.extend([
        "",  # Empty line before legend
        "*Legend:*",
        "Format: TF: Trend RSI | TF: Trend RSI  Status(Score)",
        "Example: 15m: ⚪ 64 = 15min Ranging RSI 64",
        "Trend: 🟢Bull 🔴Bear ⚪Range 🟡Vol",
        "Status: 🎯Ready ⏳Building ❌None"
    ])

    return "\n".join(lines)


def generate_and_send_market_summary():
    """
    Generate market summary and send via Telegram.
    Main function to be called by scheduler.
    """
    try:
        logger.info("Generating hourly market summary...")

        # Get market analysis data
        summaries = get_latest_market_analysis()

        if not summaries:
            logger.warning("No market data available for summary")
            return

        # Format message
        message = format_market_summary_message(summaries)

        # Send via Telegram
        send_alerts([{"summary": message}])

        logger.info(f"Market summary sent successfully for {len(summaries)} pairs")

    except Exception as e:
        logger.error(f"Failed to generate market summary: {e}")
        # Send error notification
        error_message = f"⚠️ *Market Summary Error*\nFailed to generate hourly summary: {str(e)}"
        try:
            send_alerts([{"summary": error_message}])
        except:
            logger.error("Failed to send error notification as well")


if __name__ == "__main__":
    # For testing
    generate_and_send_market_summary()