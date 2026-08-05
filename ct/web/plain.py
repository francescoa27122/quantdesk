"""The Home page's plain-English summary.

Written for an operator who is not a quant and should not need to become one
to know whether the system is fine. Rules for every sentence produced here:

* Answer the five questions a non-expert actually has — is everything OK,
  what is it doing, is it making money (and is the money real), what happens
  next, do I need to do anything — in that order.
* No jargon without an inline explanation the first time it appears.
* Never dress a number up as better than it is: practice money is called
  practice money in every sentence that mentions an amount.
* "Nothing happened" must read as normal, not as a fault: a 4-hour-bar
  trend strategy trading a few times a month is the system working.

This module only *phrases*; every number comes from `Queries`. Keep it free of
SQL so the sentences stay testable against a fake.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

#: How long the promotion gate wants a strategy to practice before it may even
#: be considered for real money (ARCHITECTURE §8.3).
PAPER_TARGET_DAYS = 90

#: One-sentence lay descriptions. Fallback below covers unknown names, so a
#: new strategy degrades to a generic sentence rather than a KeyError.
STRATEGY_PLAIN: dict[str, str] = {
    "donchian_breakout": ("a trend-follower — it buys when the price breaks "
                          "above its recent range and exits when the trend "
                          "fades"),
    "ema_cross": ("a trend-follower — it buys when the short-term average "
                  "price rises above the long-term average"),
    "macd_momentum": ("a momentum system — it trades when price swings gain "
                      "speed in one direction"),
    "supertrend": ("a trend-follower — it rides a trend and exits when price "
                   "crosses a volatility-based line"),
    "bollinger_reversion": ("a mean-reverter — it bets that unusually "
                            "stretched prices snap back toward normal"),
    "keltner_squeeze": ("a breakout system — it waits for a quiet market to "
                        "pick a direction, then follows it"),
    "zscore_reversion": ("a mean-reverter — it bets that statistically "
                         "extreme moves partially reverse"),
}

GLOSSARY: list[tuple[str, str]] = [
    ("Paper trading", "practicing with fake money against real market prices. "
     "Every trade is simulated; nothing real is bought or sold."),
    ("Strategy", "a fixed set of rules that decides when to buy and sell. "
     "No person picks the trades."),
    ("Candle / 4h bar", "the market's price summarised over 4 hours. This "
     "system decides at most once per candle, so it acts a few times a day "
     "at the very most — usually far less."),
    ("Drawdown", "how far the account has fallen from its best value so far. "
     "The system halts itself if this gets too big."),
    ("Halted", "the engine stopped opening new positions — either by itself "
     "for safety, or because someone asked. Closing positions still works."),
]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def strategy_sentence(name: str) -> str:
    return STRATEGY_PLAIN.get(name, "an automated trading rule")


def verdict(engine: dict[str, Any], attention_events: list[dict[str, Any]],
            ) -> dict[str, str]:
    """One traffic light and one sentence saying why.

    Ordered by severity, first match wins. `stale` outranks everything after
    the kill switch because a silent engine can no longer report its own
    problems — absence of bad news is not good news.
    """
    if engine.get("kill"):
        return {"level": "bad", "headline": "Stopped by the kill switch",
                "detail": "Someone engaged the emergency stop. The system "
                          "sends no orders at all until it is released."}
    if not engine.get("seen") or engine.get("stale"):
        return {"level": "bad", "headline": "The engine has gone quiet",
                "detail": "No heartbeat has arrived recently, so nothing on "
                          "this page can be trusted to be current. Check the "
                          "System page or ask Claude to look at the NAS."}
    if engine.get("halted"):
        reason = engine.get("halt_reason") or "no reason recorded"
        return {"level": "bad", "headline": "Trading is halted",
                "detail": f"The engine stopped opening new positions "
                          f"({reason}). Existing positions can still close. "
                          f"This usually needs a decision from you."}
    if any(e.get("severity") in ("critical", "warn") for e in attention_events):
        worst = attention_events[0]
        return {"level": "warn", "headline": "Running, but check one thing",
                "detail": f"The system is trading normally, but it flagged: "
                          f"{worst.get('detail') or worst.get('code')}"}
    return {"level": "ok", "headline": "Everything is fine",
            "detail": "The engine is running, market data is flowing, and "
                      "nothing needs your attention."}


def build(*, engine: dict[str, Any], perf: dict[str, Any],
          positions: list[dict[str, Any]], risk_events: list[dict[str, Any]],
          strategies: list[dict[str, Any]], fills_count: int, mode: str,
          initial_equity: float = 10_000.0,
          now: datetime | None = None) -> dict[str, Any]:
    """Everything `home.html` renders, with the sentences already built."""
    now = now or datetime.now(timezone.utc)

    attention = [e for e in risk_events
                 if e.get("severity") in ("critical", "warn")]
    fyi = [e for e in risk_events if e not in attention]

    paper = [s for s in strategies if s.get("status") == "paper"]
    live = [s for s in strategies if s.get("status") == "live"]
    running = paper + live

    # ── what it's doing ──────────────────────────────────────────────────
    if mode == "paper":
        money_word = "practice money"
        doing_intro = ("QuantDesk is practicing with fake money (called "
                       "“paper trading”) against real, live market "
                       "prices.")
    else:
        money_word = "real money"
        doing_intro = ("QuantDesk is in LIVE mode: within the limits you "
                       "armed, its trades use real money.")

    if running:
        strat_bits = [f"{s['name']} — {strategy_sentence(s['name'])}"
                      for s in running]
        doing = (f"{doing_intro} {_plural(len(running), 'strategy')} "
                 f"{'is' if len(running) == 1 else 'are'} active: "
                 f"{'; '.join(strat_bits)}. It watches 4-hour candles, so it "
                 f"decides at most a few times a day and typically trades a "
                 f"few times a month. Long quiet stretches are normal and "
                 f"are not a fault.")
    else:
        doing = (f"{doing_intro} No strategy is enabled right now, so it "
                 f"watches the market and records data but will not trade.")

    # ── the money ────────────────────────────────────────────────────────
    if perf.get("has_data"):
        equity = float(perf["equity"])
        change = equity - initial_equity
        pct = change / initial_equity * 100.0
        direction = ("up" if change > 0 else "down" if change < 0 else "flat")
        money = (f"It started with ${initial_equity:,.0f} of {money_word} "
                 f"and now has ${equity:,.2f} — {direction} "
                 f"${abs(change):,.2f} ({pct:+.2f}%).")
    else:
        money = (f"It starts with ${initial_equity:,.0f} of {money_word}. "
                 f"No value has been recorded yet — that begins with the "
                 f"engine's first cycles.")
    if mode == "paper":
        money += (" No real money is at risk: switching to real money takes "
                  "several deliberate steps that only you can perform.")

    if fills_count == 0:
        activity = ("It has not made any trades yet. For this kind of "
                    "strategy that can easily stay true for days after "
                    "enabling — it is waiting for its rules to trigger.")
    else:
        trade_word = "simulated trade" if mode == "paper" else "trade"
        activity = (f"It has made {_plural(fills_count, trade_word)} so far, "
                    f"and currently holds "
                    f"{_plural(len(positions), 'open position')}.")

    # ── what happens next ────────────────────────────────────────────────
    paper_days = None
    if paper:
        started = [s["updated_at"] for s in paper if s.get("updated_at")]
        if started:
            paper_days = max(1, (now - min(started)).days + 1)
    if paper_days is not None:
        nxt = (f"The system is on day {paper_days} of a "
               f"{PAPER_TARGET_DAYS}-day practice period. After "
               f"{PAPER_TARGET_DAYS} days with enough trades and good "
               f"enough results, a strategy can be *considered* for real "
               f"money — that step is never automatic and always yours.")
    elif live:
        nxt = ("A strategy is marked live. Real-money trading additionally "
               "requires the arming step only you can perform.")
    else:
        nxt = ("Once you enable a strategy, it must practice for "
               f"{PAPER_TARGET_DAYS} days before real money can even be "
               "discussed. Enabling is one command; ask Claude or see the "
               "README.")

    return {
        "verdict": verdict(engine, attention),
        "doing": doing,
        "money": money,
        "activity": activity,
        "next": nxt,
        "attention": attention,
        "fyi": fyi,
        "paper_days": paper_days,
        "paper_target_days": PAPER_TARGET_DAYS,
        "glossary": GLOSSARY,
    }


def positions_lede(positions: list[Any], mode: str) -> str:
    """One sentence above the Positions tables, in the Home page's voice."""
    money = "practice money" if mode == "paper" else "real money"
    if not positions:
        return (f"The system holds nothing right now — all its {money} is in "
                f"cash, waiting for a strategy's rules to trigger. Days of "
                f"this are normal.")
    return (f"The system currently holds "
            f"{_plural(len(positions), 'position')}, opened with {money}. "
            f"Each row shows what it bought, at what price, and where its "
            f"automatic exit levels sit.")


def trades_lede(n_trades: int, mode: str) -> str:
    """One sentence above the Trades table."""
    word = "practice trade" if mode == "paper" else "trade"
    if n_trades == 0:
        return ("No trades yet — every trade the system ever makes will "
                "appear here, newest first, traced from the signal that "
                "started it to the profit or loss that ended it.")
    return (f"{_plural(n_trades, word)} so far, newest first — each traced "
            f"from the signal that started it to the profit or loss that "
            f"ended it.")


__all__ = ["GLOSSARY", "PAPER_TARGET_DAYS", "STRATEGY_PLAIN", "build",
           "positions_lede", "strategy_sentence", "trades_lede", "verdict"]
