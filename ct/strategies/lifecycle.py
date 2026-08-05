"""Strategy lifecycle — Phase 2 item 7.

## The defect this exists to fix

The engine kept quarantined strategies in `RuntimeState.paused_strategies`, a
`set` built fresh on every boot. So a strategy that raised, got quarantined and
was deliberately stopped came back **alive on the next container restart** —
and `restart: unless-stopped` means restarts are routine, not exceptional. A
safety decision that a restart undoes is not a safety decision.

State now lives in the `strategies` table, which until this had no writer
anywhere in the codebase.

## The states

    proposed ──enable──▶ paper ──promote──▶ live
        ▲                 │  ▲                │
        │                 │  └──── demote ────┘
     retire            pause/quarantine
        │                 │
        └───── retired ◀──┴──▶ paused ──expiry──▶ (previous)
                              quarantined ──human──▶ (previous)

## The asymmetry, which is the whole design

**Transitions that reduce risk are automatic. Transitions that increase it need
a person.** That is the same rule as the arming charter (ARCHITECTURE §8.4),
applied one level down:

* the engine may **pause**, **quarantine** and **demote** (live → paper) on its
  own judgement, at any time, without asking;
* the engine may **resume from a pause** once its timer has expired, because
  the pause itself was time-boxed by a human-set policy;
* the engine may **never** enable, promote, or clear a quarantine. A crash
  needs somebody to read the traceback, and a promotion puts capital behind a
  strategy.

`live` is a statement about the promotion gate, **not** a permission to trade
real money. That still requires `QD_MODE=live` *and* a human-rooted arming row.
This FSM cannot widen the money boundary because it is not on that path.

Pure and I/O-free: the repository in `ct/store/strategies.py` persists it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class StrategyState(str, Enum):
    """Stored verbatim in `strategies.status`."""

    PROPOSED = "proposed"          # exists; never validated; cannot trade
    PAPER = "paper"                # accumulating the 90-day record (§8.3)
    LIVE = "live"                  # passed the promotion gate
    PAUSED = "paused"              # loss breaker; time-boxed
    QUARANTINED = "quarantined"    # raised an exception; needs a human
    RETIRED = "retired"            # deliberately stopped


#: States in which the engine will call `on_bar` at all.
RUNNING = frozenset({StrategyState.PAPER, StrategyState.LIVE})

#: States a human must move out of. The engine may enter them, never leave.
NEEDS_HUMAN = frozenset({StrategyState.QUARANTINED, StrategyState.RETIRED,
                         StrategyState.PROPOSED})


class Actor(str, Enum):
    ENGINE = "engine"
    HUMAN = "human"


class IllegalTransition(Exception):
    """Refused. The message names the state, the target and who asked."""


# (from, to) -> who may make it. Absent means nobody may.
_TRANSITIONS: dict[tuple[StrategyState, StrategyState], frozenset[Actor]] = {
    # ── risk-increasing: human only ──────────────────────────────────────
    (StrategyState.PROPOSED, StrategyState.PAPER): frozenset({Actor.HUMAN}),
    (StrategyState.PAPER, StrategyState.LIVE): frozenset({Actor.HUMAN}),
    (StrategyState.QUARANTINED, StrategyState.PAPER): frozenset({Actor.HUMAN}),
    (StrategyState.QUARANTINED, StrategyState.LIVE): frozenset({Actor.HUMAN}),
    (StrategyState.RETIRED, StrategyState.PAPER): frozenset({Actor.HUMAN}),
    (StrategyState.RETIRED, StrategyState.PROPOSED): frozenset({Actor.HUMAN}),

    # ── risk-reducing: either may ────────────────────────────────────────
    (StrategyState.PAPER, StrategyState.PAUSED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.LIVE, StrategyState.PAUSED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.PAPER, StrategyState.QUARANTINED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.LIVE, StrategyState.QUARANTINED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.PAUSED, StrategyState.QUARANTINED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    # Demotion on a promotion-gate regression: the engine may always step DOWN.
    (StrategyState.LIVE, StrategyState.PAPER):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.PAPER, StrategyState.RETIRED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.LIVE, StrategyState.RETIRED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.PAUSED, StrategyState.RETIRED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.QUARANTINED, StrategyState.RETIRED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.PROPOSED, StrategyState.RETIRED):
        frozenset({Actor.ENGINE, Actor.HUMAN}),

    # ── resuming a time-boxed pause ──────────────────────────────────────
    # The engine may, but `resume_due` gates it on the timer having expired.
    (StrategyState.PAUSED, StrategyState.PAPER):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
    (StrategyState.PAUSED, StrategyState.LIVE):
        frozenset({Actor.ENGINE, Actor.HUMAN}),
}


@dataclass
class StrategyLifecycle:
    """One strategy's state, and the rules for changing it."""

    name: str
    state: StrategyState = StrategyState.PROPOSED
    reason: str = ""
    paused_until: datetime | None = None
    # What it was doing before it was paused, so an expiry restores the right
    # state rather than guessing `paper`.
    resume_to: StrategyState | None = None
    updated_at: datetime | None = None
    history: list[tuple[datetime, StrategyState, StrategyState, str, str]] = \
        field(default_factory=list)

    # ── queries ──────────────────────────────────────────────────────────

    @property
    def may_run(self) -> bool:
        return self.state in RUNNING

    @property
    def may_trade_live(self) -> bool:
        """True only in `live`. Still not sufficient — `QD_MODE=live` and a
        human-rooted arming row are the actual gate (§8.4)."""
        return self.state is StrategyState.LIVE

    def resume_due(self, now: datetime | None = None) -> bool:
        """Has a time-boxed pause expired?"""
        if self.state is not StrategyState.PAUSED or self.paused_until is None:
            return False
        return (now or _utc()) >= self.paused_until

    def can(self, target: StrategyState, actor: Actor) -> bool:
        return actor in _TRANSITIONS.get((self.state, target), frozenset())

    # ── transitions ──────────────────────────────────────────────────────

    def to(self, target: StrategyState, *, actor: Actor, reason: str = "",
           now: datetime | None = None) -> StrategyLifecycle:
        """Apply a transition or raise `IllegalTransition`.

        Refusing loudly rather than ignoring: a transition the caller believed
        happened and did not is how a quarantined strategy quietly keeps
        trading.
        """
        now = now or _utc()
        if target is self.state:
            return self                       # idempotent, and not an error
        allowed = _TRANSITIONS.get((self.state, target), frozenset())
        if not allowed:
            raise IllegalTransition(
                f"{self.name}: {self.state.value} -> {target.value} is not a "
                f"legal transition")
        if actor not in allowed:
            raise IllegalTransition(
                f"{self.name}: only {'/'.join(sorted(a.value for a in allowed))}"
                f" may move {self.state.value} -> {target.value}; "
                f"{actor.value} asked. Risk-increasing transitions need a "
                f"person (see the module docstring).")

        self.history.append((now, self.state, target, actor.value, reason))
        previous = self.state
        self.state = target
        self.reason = reason
        self.updated_at = now
        if target is StrategyState.PAUSED:
            self.resume_to = previous if previous in RUNNING else StrategyState.PAPER
        else:
            self.paused_until = None
            self.resume_to = None
        return self

    # ── the transitions the engine actually makes ────────────────────────

    def quarantine(self, reason: str, *, now: datetime | None = None
                   ) -> StrategyLifecycle:
        """A strategy raised. Stop it and keep it stopped **across restarts**
        until a person has read the traceback."""
        return self.to(StrategyState.QUARANTINED, actor=Actor.ENGINE,
                       reason=reason, now=now)

    def pause(self, reason: str, *, hours: float = 24.0,
              now: datetime | None = None) -> StrategyLifecycle:
        """Loss breaker. Time-boxed, so it heals itself."""
        now = now or _utc()
        self.to(StrategyState.PAUSED, actor=Actor.ENGINE, reason=reason, now=now)
        self.paused_until = now + timedelta(hours=hours)
        return self

    def resume_if_due(self, now: datetime | None = None
                      ) -> StrategyLifecycle:
        """Return to whatever it was doing, once the pause has expired."""
        now = now or _utc()
        if not self.resume_due(now):
            return self
        target = self.resume_to or StrategyState.PAPER
        return self.to(target, actor=Actor.ENGINE,
                       reason="pause expired", now=now)

    def demote(self, reason: str, *, now: datetime | None = None
               ) -> StrategyLifecycle:
        """live -> paper on a promotion-gate regression. Always permitted: it
        is a reduction, and reductions never wait for permission."""
        if self.state is not StrategyState.LIVE:
            return self
        return self.to(StrategyState.PAPER, actor=Actor.ENGINE, reason=reason,
                       now=now)

    def __str__(self) -> str:
        extra = f" until {self.paused_until:%Y-%m-%d %H:%M}Z" \
            if self.paused_until else ""
        return f"{self.name}: {self.state.value}{extra} ({self.reason or '—'})"


def _utc() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "NEEDS_HUMAN",
    "RUNNING",
    "Actor",
    "IllegalTransition",
    "StrategyLifecycle",
    "StrategyState",
]
