import logging
import traceback
from datetime import datetime
from typing import Optional

import requests

from app.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_exception_notification(exception: Exception, context: str = "Application", details: Optional[str] = None):
    """
    Send a Telegram notification when an exception occurs.

    Args:
        exception: The exception that occurred
        context: Context where the exception occurred (e.g., function name, job name)
        details: Additional details about the exception
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured, skipping exception notification")
        return

    try:
        # Format the exception message
        exc_type = type(exception).__name__
        exc_message = str(exception)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Get traceback info
        tb_lines = traceback.format_exc().split('\n')
        # Take last few lines of traceback for brevity
        relevant_tb = '\n'.join(tb_lines[-8:]) if len(tb_lines) > 8 else traceback.format_exc()

        # Build the message
        text_lines = [
            "🚨 *Exception Alert*",
            "",
            f"🔴 *Type:* {exc_type}",
            f"📍 *Context:* {context}",
            f"⏰ *Time:* {timestamp}",
            "",
            f"💬 *Message:*",
            f"```",
            exc_message,
            "```"
        ]

        if details:
            text_lines.extend([
                "",
                f"📝 *Details:* {details}"
            ])

        text_lines.extend([
            "",
            f"📋 *Traceback:*",
            f"```",
            relevant_tb,
            "```"
        ])

        text = "\n".join(text_lines)

        # Telegram has a message length limit
        if len(text) > 4000:
            text = text[:3900] + "\n...\n```\n[Message truncated due to length]"

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info(f"[ExceptionNotifier] Exception notification sent successfully: {resp.status_code}")

    except Exception as e:
        # Avoid recursive exception notifications for Telegram failures
        logger.error(f"[ExceptionNotifier] Failed to send exception notification: {e}")
        # Do not re-raise or send another notification to prevent infinite loops


class ExceptionNotifierHandler(logging.Handler):
    """
    Custom logging handler that sends exceptions to Telegram.
    """

    def emit(self, record):
        """
        Emit a log record. If it's an exception, send a Telegram notification.
        """
        try:
            if record.levelno >= logging.ERROR and record.exc_info:
                exc_type, exc_value, exc_traceback = record.exc_info
                if exc_value:
                    context = f"{record.name}:{record.funcName}" if record.funcName else record.name
                    send_exception_notification(exc_value, context, record.getMessage())
        except Exception:
            # Silently fail to avoid recursive issues
            pass


def setup_exception_notification():
    """
    Set up the exception notification system by adding the handler to the root logger.
    """
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        handler = ExceptionNotifierHandler()
        handler.setLevel(logging.ERROR)

        # Add to root logger to catch all exceptions
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        logger.info("Exception notification system initialized")
    else:
        logger.warning("Telegram credentials not configured, exception notifications disabled")