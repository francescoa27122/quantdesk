"""Notification sender tests. The load-bearing ones are the degradations:
unconfigured must drain (not backlog), failure must return False (the
outbox's backoff owns retries), and the token must survive the trailing
newline that `cat > file` always writes."""

from __future__ import annotations

import asyncio

from ct.notify import TelegramSender


def _sender(token="123:abc", chat="42", **kw):
    return TelegramSender(token, chat, **kw)


def test_unconfigured_drains_instead_of_backlogging(caplog):
    s = TelegramSender("", "")
    assert asyncio.run(s.send("telegram", "subject", "body")) is True
    assert not s.configured


def test_configured_posts_to_the_bot_api():
    calls = []

    async def post(url, payload):
        calls.append((url, payload))
        return True

    s = _sender(post=post)
    assert asyncio.run(s.send("telegram", "Halted", "drawdown 21%")) is True
    url, payload = calls[0]
    assert url == "https://api.telegram.org/bot123:abc/sendMessage"
    assert payload["chat_id"] == "42"
    assert payload["text"] == "Halted\ndrawdown 21%"


def test_network_failure_returns_false_for_the_outbox_to_retry():
    async def post(url, payload):
        raise ConnectionError("api.telegram.org unreachable")

    s = _sender(post=post)
    assert asyncio.run(s.send("telegram", None, "body")) is False


def test_http_refusal_returns_false():
    async def post(url, payload):
        return False

    assert asyncio.run(_sender(post=post).send("t", "s", "b")) is False


def test_token_survives_the_trailing_newline(tmp_path, monkeypatch):
    """The classic `cat > file` newline: the exchange keys shipped with one
    and it cost a debugging round (WORKLOG 2026-07-29). Never again."""
    f = tmp_path / "token"
    f.write_text("123:abc\n", encoding="utf-8")
    monkeypatch.setenv("QD_TELEGRAM_BOT_TOKEN_FILE", str(f))
    monkeypatch.setenv("QD_TELEGRAM_CHAT_ID", "42\n")
    s = TelegramSender.from_env()
    assert s.token == "123:abc" and s.chat_id == "42"
    assert s.configured


def test_missing_token_file_degrades_not_raises(monkeypatch):
    monkeypatch.setenv("QD_TELEGRAM_BOT_TOKEN_FILE", "/nonexistent/token")
    monkeypatch.setenv("QD_TELEGRAM_CHAT_ID", "42")
    assert TelegramSender.from_env().configured is False


def test_engine_vol_matches_the_backtester_exactly():
    """Backtest/live parity for the sizer's vol input — the identity that
    makes a paper record meaningful."""
    import numpy as np

    from ct.backtest.engine import BacktestEngine
    from ct.engine import Allocation, TradingEngine
    from ct.execution.broker import PaperBroker
    from ct.strategies import REGISTRY

    closes = list(100.0 + np.cumsum(
        np.random.default_rng(7).standard_normal(60)))
    bt = BacktestEngine(REGISTRY["ema_cross"](), timeframe="4h")
    e = TradingEngine([Allocation(REGISTRY["ema_cross"](), ("BTC/USD",), "4h")],
                      PaperBroker(), db=None)
    live = e._compute_vol(closes, "4h")
    back = bt._realized_ann_vol(np.array(closes), len(closes) - 1)
    assert abs(live - back) < 1e-12, (live, back)
    assert live > 0
