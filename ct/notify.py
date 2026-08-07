"""Notification delivery — the outbox's real sender.

Until 2026-07-30 `_send_notification` logged and returned True, so every
"loud" notification — halts, quarantines, drawdown breaches, arming changes —
drained into a log file nobody watches at 3 a.m. This module is the swap-in.

Design rules:
* **Unconfigured is not an error.** With no token or chat id the sender logs
  the message and reports success, so the outbox drains instead of growing a
  retry backlog on a box that simply has no Telegram wired. It says it is
  falling back, once per boot, rather than silently.
* **Failure returns False**, and the outbox's existing backoff/retry/give-up
  machinery does the rest. No retry logic here — one owner per concern.
* **Plain text, no parse_mode.** Markdown escaping bugs turn an alert about
  `qty_0.5*price` into a silent 400 exactly when the message matters most.
* The token is read from the secret file at first use, stripped of the
  trailing newline that `cat > file` always leaves (that exact newline cost a
  debugging round on the exchange keys — see WORKLOG 2026-07-29).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramSender:
    def __init__(self, token: str | None, chat_id: str | None, *,
                 post: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None,
                 ) -> None:
        self.token = (token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self._post = post or self._post_aiohttp
        self._warned_unconfigured = False

    @classmethod
    def from_env(cls) -> TelegramSender:
        token = ""
        path = os.environ.get("QD_TELEGRAM_BOT_TOKEN_FILE", "")
        if path:
            try:
                token = Path(path).read_text(encoding="utf-8").strip()
            except OSError:
                token = ""
        return cls(token, os.environ.get("QD_TELEGRAM_CHAT_ID", ""))

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, channel: str, subject: str | None, body: str) -> bool:
        """True = delivered (or deliberately dropped to the log); False = the
        outbox should retry with backoff."""
        if not self.configured:
            if not self._warned_unconfigured:
                log.warning("telegram is not configured (token file or "
                            "QD_TELEGRAM_CHAT_ID missing); notifications go "
                            "to this log only")
                self._warned_unconfigured = True
            log.info("NOTIFY [%s] %s: %s", channel, subject or "", body)
            return True
        text = f"{subject}\n{body}" if subject else body
        try:
            ok = await self._post(
                _API.format(token=self.token),
                {"chat_id": self.chat_id, "text": text[:4000],
                 "disable_web_page_preview": True})
        except Exception as exc:
            log.warning("telegram send failed (%s); outbox will retry",
                        type(exc).__name__)
            return False
        if not ok:
            log.warning("telegram refused the message; outbox will retry")
        return ok

    @staticmethod
    async def _post_aiohttp(url: str, payload: dict[str, Any]) -> bool:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=10)
        async with (aiohttp.ClientSession(timeout=timeout) as session,
                    session.post(url, json=payload) as resp):
            if resp.status == 200:
                return True
            log.warning("telegram HTTP %s: %s", resp.status,
                        (await resp.text())[:200])
            return False


__all__ = ["TelegramSender"]
