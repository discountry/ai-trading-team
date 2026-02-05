"""Telegram notification client."""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send Telegram messages via Bot API."""

    def __init__(self, bot_token: str, chat_id: str, account_label: str = "") -> None:
        self._bot_token = bot_token.strip()
        self._chat_id = chat_id.strip()
        self._account_label = account_label.strip()
        self._enabled = bool(self._bot_token and self._chat_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def account_label(self) -> str:
        return self._account_label

    async def send_message(self, message: str) -> None:
        if not self._enabled:
            return
        try:
            await asyncio.to_thread(self._post_message, message)
        except Exception as exc:
            logger.warning("Telegram message failed: %s", exc)

    def _post_message(self, message: str) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = urllib.parse.urlencode(
            {
                "chat_id": self._chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
