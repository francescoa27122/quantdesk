"""Auth, sessions, CSRF and rate limiting for `qd-web`.

The threat model this is built against, stated plainly: the dashboard is on
Tailscale, so the attacker is not the open internet — it is a stolen laptop, a
borrowed browser tab, or a page on another site trying to POST to the halt
endpoint. Everything below is aimed at those three.

**`qd-web` holds no exchange credentials.** It authenticates people, reads the
database and writes `control_commands`. A total compromise of this process
cannot place an order, because nothing here can reach an exchange.

## Sessions

A signed cookie, not a server-side store. One operator, one container, and a
session table would add a write path to a service whose whole point is that it
is read-mostly. The cookie carries `user | role | issued | expires | csrf`,
HMAC-SHA256 over the secret from `QD_SESSION_SECRET_FILE`, compared in constant
time. It cannot be forged without the secret and cannot be extended by editing,
because the expiry is inside the signed payload.

## Roles

`viewer` (read) · `operator` (halt, resume, flatten, pause) · `admin` (all of
that plus arming-adjacent views). Passwords are argon2id hashes supplied as env
vars — there is no user table, because one person does not need one.

**Halt is available to every role, including `viewer`.** That asymmetry is
deliberate and load-bearing: actions that reduce risk are never gated
(ARCHITECTURE.md §9). If you can see the dashboard you can stop it.

## What is deliberately NOT built

TOTP. ARCHITECTURE.md §9 lists it as optional, and a second factor without an
enrolment flow, recovery codes and a re-key path is security theatre that locks
you out at the worst moment. The surface is already Tailscale-only. Left as a
stated gap rather than a half-built one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SESSION_COOKIE = "qd_session"
SESSION_TTL_S = 12 * 3600
# Argon2 verification of a wrong password takes ~50 ms by design. Five tries
# per window is generous for a human and useless for a script.
MAX_ATTEMPTS = 5
LOCKOUT_S = 900

ROLES = ("viewer", "operator", "admin")
# What each role may do. `halt` is absent from every list on purpose — see
# `may()`: stopping is never gated.
PRIVILEGES: dict[str, frozenset[str]] = {
    "viewer": frozenset(),
    "operator": frozenset({"resume", "flatten", "pause_strategy",
                           "resume_strategy", "resolve_event"}),
    "admin": frozenset({"resume", "flatten", "pause_strategy",
                        "resume_strategy", "reload_config", "disarm",
                        "resolve_event"}),
}

# A real argon2id hash of a random string, used to burn the same CPU on an
# unknown username as on a known one. Without it, login timing tells an
# attacker which usernames exist.
_DUMMY_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$"
    "c29tZXNhbHRzb21lc2FsdA$0Nc7RjPBNP0KGWLQqNMWFq5PErNw9jQ1sT+7RRlLZ0M")


class AuthError(Exception):
    """Login refused. The message is deliberately identical for every cause."""


@dataclass(frozen=True, slots=True)
class Session:
    user: str
    role: str
    issued: int
    expires: int
    csrf: str

    @property
    def stale(self) -> bool:
        """Older than a quarter of its life — worth re-issuing on activity."""
        return time.time() - self.issued > SESSION_TTL_S / 4

    def may(self, privilege: str) -> bool:
        # Halt is the exception that defines the rule: every authenticated
        # role may stop the system. Refusing to let somebody stop trading
        # because their account is the wrong shape is how a small loss becomes
        # a large one.
        if privilege == "halt":
            return True
        return privilege in PRIVILEGES.get(self.role, frozenset())


def session_secret() -> bytes:
    """From a Docker secret file, or `QD_SESSION_SECRET`, or refuse.

    No development fallback that invents a key. A generated-at-boot secret
    would silently log everyone out on every restart *and* would mean the
    signature guarantees nothing about who issued it.
    """
    path = os.environ.get("QD_SESSION_SECRET_FILE")
    if path and Path(path).is_file():
        raw = Path(path).read_bytes().strip()
        if raw:
            return raw
    raw_env = os.environ.get("QD_SESSION_SECRET", "").strip()
    if raw_env:
        return raw_env.encode()
    raise RuntimeError(
        "no session secret: set QD_SESSION_SECRET_FILE (Docker secret) or "
        "QD_SESSION_SECRET. Generate one with `openssl rand -base64 48`.")


def known_users() -> dict[str, tuple[str, str]]:
    """`{username: (argon2_hash, role)}` from the environment or a secret file.

    Only hashes ever reach this process — the plaintext exists solely in the
    operator's head and in whatever they typed into `qd hash-password`.

    Usernames are the role names: `admin`, `operator`, `viewer`. One person does
    not need a user table.

    ## Prefer `QD_<ROLE>_PASSWORD_HASH_FILE`

    **An argon2 hash always contains `$`, and Docker Compose interpolates `$` in
    environment values.** Putting the hash in `.env` as the README described
    silently mangled it:

        $argon2id$v=19$m=65536,t=3,p=4$mkLF...   (what you generate)
        =19=65536,t=3,p=4+2DEg                   (what the container received)

    Every login then failed 401 with correct credentials. Found by running the
    real compose stack; nothing short of that would have caught it, because the
    hash is well-formed everywhere except inside a container Compose started.

    The file route sidesteps interpolation entirely and matches how every other
    secret here is delivered. If you do use the environment variable, every `$`
    must be doubled to `$$` in `.env`.
    """
    users: dict[str, tuple[str, str]] = {}
    for role in ROLES:
        env = f"QD_{role.upper()}_PASSWORD_HASH"
        candidates: list[tuple[str, str]] = []

        raw = os.environ.get(env, "").strip()
        if raw:
            candidates.append((env, raw))

        path = os.environ.get(f"{env}_FILE", "").strip()
        if path:
            try:
                from_file = Path(path).read_text(encoding="utf-8").strip()
                if from_file:
                    candidates.append((f"{env}_FILE", from_file))
            except OSError as exc:
                # Not fatal: compose sets the _FILE variable unconditionally, so
                # a deployment using the env route legitimately has no file.
                log.debug("cannot read %s_FILE (%s): %s", env, path, exc)

        # First WELL-FORMED candidate wins, env before file. Order matters and
        # so does the "well-formed" part: an env value mangled by Compose must
        # not shadow a perfectly good secret file, which an env-first-wins rule
        # would have let it do.
        for source, digest in candidates:
            if digest.startswith("$argon2"):
                users[role] = (digest, role)
                break
            log.error(
                "%s does not look like an argon2 hash (starts with %r) and was "
                "ignored. If it came from a Compose .env, the $ characters were "
                "interpolated away — double them to $$, or use %s_FILE. "
                "Regenerate with `qd hash-password`.", source, digest[:12], env)
    return users


def verify_password(user: str, password: str) -> str:
    """Return the role, or raise `AuthError`. Constant work either way."""
    from argon2 import PasswordHasher
    from argon2.exceptions import (
        InvalidHashError,
        VerificationError,
        VerifyMismatchError,
    )

    users = known_users()
    if not users:
        raise AuthError(
            "no users are configured: set QD_ADMIN_PASSWORD_HASH "
            "(`qd hash-password`) and restart qd-web")
    digest, role = users.get(user, (_DUMMY_HASH, ""))
    try:
        PasswordHasher().verify(digest, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError) as exc:
        # A malformed hash (InvalidHashError) is an operator mistake, not a bad
        # password, and used to be indistinguishable from one: the catch was
        # `(…, Exception)`, so a typo'd QD_ADMIN_PASSWORD_HASH presented as
        # "invalid credentials" forever with nothing in the log. It still fails
        # closed — it must — but now it says so where you can find it.
        #
        # The old `if isinstance(exc, KeyboardInterrupt): raise` guard was dead
        # code: KeyboardInterrupt derives from BaseException, so `except
        # Exception` never caught it in the first place. Narrowing the tuple
        # makes that guarantee structural rather than incidental.
        if isinstance(exc, InvalidHashError):
            log.error("configured password hash for %r is malformed — "
                      "regenerate it with `qd hash-password`", user)
        raise AuthError("invalid credentials") from None
    if not role:
        # The password happened to match the dummy hash, which cannot happen
        # for a random 48-byte secret — but never trust "cannot happen" on the
        # authentication path.
        raise AuthError("invalid credentials")
    return role


# ── cookie signing ───────────────────────────────────────────────────────


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue(user: str, role: str, secret: bytes, *, now: float | None = None,
          ) -> tuple[str, Session]:
    now_i = int(now if now is not None else time.time())
    sess = Session(user=user, role=role, issued=now_i,
                   expires=now_i + SESSION_TTL_S,
                   csrf=secrets.token_urlsafe(24))
    payload = _b64e(json.dumps({
        "u": sess.user, "r": sess.role, "i": sess.issued,
        "e": sess.expires, "c": sess.csrf}, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(secret, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}", sess


def read(cookie: str | None, secret: bytes, *, now: float | None = None,
         ) -> Session | None:
    """Verify and decode. Returns None for anything that is not a live,
    correctly signed session — never raises, never explains which check failed."""
    if not cookie or "." not in cookie:
        return None
    payload, _, sig = cookie.rpartition(".")
    expected = _b64e(hmac.new(secret, payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(_b64d(payload))
        sess = Session(user=str(data["u"]), role=str(data["r"]),
                       issued=int(data["i"]), expires=int(data["e"]),
                       csrf=str(data["c"]))
    except (ValueError, KeyError, TypeError):
        return None
    if sess.expires <= int(now if now is not None else time.time()):
        return None
    if sess.role not in ROLES:
        return None
    return sess


# ── login rate limiting ──────────────────────────────────────────────────


class RateLimiter:
    """Per-key sliding window, in memory.

    In memory is the right call here and not laziness: one container, one
    process, and a shared store would put a *write* on the login path of a
    service defined by not writing. A restart forgives outstanding lockouts —
    acceptable, because a restart is not something an attacker can trigger.
    """

    def __init__(self, max_attempts: int = MAX_ATTEMPTS,
                 window: int = LOCKOUT_S) -> None:
        self.max_attempts = max_attempts
        self.window = window
        self._hits: dict[str, list[float]] = {}

    def _prune(self, key: str, now: float) -> list[float]:
        hits = [t for t in self._hits.get(key, []) if now - t < self.window]
        self._hits[key] = hits
        return hits

    def check(self, key: str, *, now: float | None = None) -> int:
        """Seconds to wait, or 0 if allowed."""
        now = now if now is not None else time.time()
        hits = self._prune(key, now)
        if len(hits) < self.max_attempts:
            return 0
        return max(1, int(self.window - (now - hits[0])))

    def fail(self, key: str, *, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self._prune(key, now).append(now)

    def clear(self, key: str) -> None:
        self._hits.pop(key, None)


__all__ = [
    "MAX_ATTEMPTS",
    "PRIVILEGES",
    "ROLES",
    "SESSION_COOKIE",
    "SESSION_TTL_S",
    "AuthError",
    "RateLimiter",
    "Session",
    "issue",
    "known_users",
    "read",
    "session_secret",
    "verify_password",
]
