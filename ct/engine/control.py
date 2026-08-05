"""Control plane — every way a human stops the engine.

Three independent paths, deliberately not sharing a mechanism. If they did,
one bug would take them all out at once, and the whole point is that *something*
still works when something else is broken:

  1. **A file on disk** (`/app/state/KILL`). Needs no database, no network, no
     web UI, no working engine logic beyond `os.path.exists`. This is the path
     that works when everything else is down, and it is checked first.
  2. **`control_commands` rows**, written by `qd-web` and polled here. Survives
     a dashboard restart because the intent is durable, not a live call.
  3. **SIGTERM**, from `docker stop`. Graceful: cancel resting orders, persist,
     exit inside the 30 s `stop_grace_period`.

**Stopping is never gated.** Every role may halt, the kill switch needs no
authentication beyond filesystem access, and no drawdown or arming state can
prevent a halt from taking effect. Actions that reduce risk do not need
permission — that asymmetry is deliberate and load-bearing.

`halt` and `kill` are different on purpose:

  * **HALT** — stop opening; exits still work. "Stop taking new risk."
  * **KILL** — stop everything, exits included. "I do not trust this system."
    To get *out* of positions, use `flatten_all`, which is a third thing again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ct.store.db import Database

log = logging.getLogger(__name__)

VALID_COMMANDS = frozenset({
    "halt", "resume", "flatten_symbol", "flatten_all",
    "pause_strategy", "resume_strategy", "reload_config",
    "resolve_risk_event",
})


@dataclass
class KillSwitch:
    """Presence of a file. Nothing else — that is the feature.

    No database, no network, no auth, no engine logic. `touch` it over SSH when
    the dashboard is down and the exchange is on fire.
    """
    path: Path

    def __init__(self, path: str | Path = "/app/state/KILL") -> None:
        self.path = Path(path)

    @property
    def engaged(self) -> bool:
        try:
            return self.path.exists()
        except OSError:
            # A filesystem that cannot be read is not a reason to keep
            # trading. Fail closed: an unreadable kill switch counts as ON.
            log.exception("cannot read kill switch at %s; treating as ENGAGED",
                          self.path)
            return True

    def engage(self, reason: str = "manual") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(reason, encoding="utf-8")

    def release(self) -> None:
        self.path.unlink(missing_ok=True)

    def reason(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip() or "manual"
        except OSError:
            return "unknown"


@dataclass
class ControlCommand:
    id: int
    command: str
    args: dict[str, Any] = field(default_factory=dict)
    requested_by: str = "unknown"


class ControlPlane:
    """Polls `control_commands`. The ONLY table `qd-web` may write.

    The engine remains the single writer of trading state; the dashboard writes
    *intent* and the engine decides. That keeps one process authoritative over
    orders and positions while still giving the UI a red button.
    """

    def __init__(self, db: Database, account_id: int) -> None:
        self.db = db
        self.account_id = account_id
        self.applied = 0
        self.rejected = 0

    async def pending(self, limit: int = 20) -> list[ControlCommand]:
        rows = await self.db.fetch(
            """SELECT id, command, args, requested_by FROM control_commands
               WHERE applied_at IS NULL
                 AND (account_id = $1 OR account_id IS NULL)
               ORDER BY id LIMIT $2""",
            self.account_id, limit)
        out: list[ControlCommand] = []
        for r in rows:
            args = r["args"]
            if isinstance(args, str):
                import json
                args = json.loads(args)
            out.append(ControlCommand(int(r["id"]), r["command"], args or {},
                                      r["requested_by"]))
        return out

    async def mark_applied(self, command_id: int, result: str) -> None:
        await self.db.execute(
            """UPDATE control_commands SET applied_at = now(), result = $2
               WHERE id = $1""", command_id, result[:500])

    async def request(self, command: str, *, requested_by: str,
                      args: dict[str, Any] | None = None) -> int:
        """Used by the CLI and by tests. `qd-web` inserts directly."""
        if command not in VALID_COMMANDS:
            raise ValueError(
                f"unknown control command {command!r}; valid: "
                f"{sorted(VALID_COMMANDS)}")
        import json
        return int(await self.db.fetchval(
            """INSERT INTO control_commands (account_id, command, args,
                                             requested_by)
               VALUES ($1,$2,$3::jsonb,$4) RETURNING id""",
            self.account_id, command, json.dumps(args or {}), requested_by))


@dataclass
class RuntimeState:
    """What the engine currently believes about its own permission to trade."""

    halted: bool = True                 # boots halted; reconciliation lifts it
    halt_reason: str = "not yet reconciled"
    kill_engaged: bool = False
    armed: bool = False
    mode: str = "paper"
    paused_strategies: set[str] = field(default_factory=set)
    flatten_requested: set[str] = field(default_factory=set)
    flatten_all: bool = False
    reconciled: bool = False

    @property
    def may_open(self) -> bool:
        """Opening needs everything to be true at once."""
        return (self.reconciled and not self.halted and not self.kill_engaged
                and self.armed)

    @property
    def may_close(self) -> bool:
        """Closing is blocked ONLY by the kill switch.

        Not by halt, not by drawdown, not by disarming. Refusing to let a
        position close because the account is in trouble is how a small loss
        becomes a large one.
        """
        return not self.kill_engaged

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        log.warning("engine HALTED: %s", reason)

    def resume(self) -> bool:
        """Refuses while unreconciled. Resuming into a state where we disagree
        with the exchange about what we own is worse than staying down."""
        if not self.reconciled:
            log.warning("resume refused: reconciliation has not passed")
            return False
        self.halted = False
        self.halt_reason = ""
        log.info("engine RESUMED")
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "halted": self.halted,
            "halt_reason": self.halt_reason, "kill": self.kill_engaged,
            "armed": self.armed, "reconciled": self.reconciled,
            "may_open": self.may_open, "may_close": self.may_close,
            "paused_strategies": sorted(self.paused_strategies),
            "flatten_all": self.flatten_all,
            "flatten_symbols": sorted(self.flatten_requested),
        }


__all__ = [
    "VALID_COMMANDS",
    "ControlCommand",
    "ControlPlane",
    "KillSwitch",
    "RuntimeState",
]
