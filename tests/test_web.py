"""Dashboard tests.

Three things are worth testing here and the rest is decoration:

1. **Every page renders against an empty database.** That is this system's
   actual state today, and a dashboard that 500s because nothing has happened
   yet is worse than no dashboard. Most of these tests assert exactly that.
2. **The authorisation boundary.** Who may halt (everyone), who may resume
   (not a viewer), what a missing CSRF token does, and that an unauthenticated
   browser is sent to the login page rather than shown a page.
3. **The honest-output rules.** A missing number renders as an em dash, not
   zero. A flatten request says it does not liquidate. A NULL maker ratio is
   not 0%.

No database and no network: `ct.web.app.state.db` is a `FakeDB` that answers by
SQL substring and records what was written. The queries themselves are exercised
against real Postgres in `tests/test_store.py`.
"""

from __future__ import annotations

import os

import pytest

# The app reads its session secret at lifespan start; set it before import so
# a bare `import ct.web.app` in another test cannot pick up a missing one.
os.environ.setdefault("QD_SESSION_SECRET", "test-secret-not-used-anywhere-real")

from fastapi.testclient import TestClient

from ct.web import app as webapp
from ct.web import charts, security
from ct.web import format as fmt

PASSWORD = "correct-horse-battery-staple"


@pytest.fixture(scope="module")
def admin_hash() -> str:
    from argon2 import PasswordHasher

    return PasswordHasher().hash(PASSWORD)


class FakeDB:
    """Answers by SQL substring; records every statement."""

    def __init__(self, rows: dict | None = None, vals: dict | None = None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.rows = rows or {}
        self.vals = vals or {}

    def _match(self, table: dict, sql: str):
        flat = " ".join(sql.split())
        for needle, value in table.items():
            if needle in flat:
                return value
        return None

    def wrote(self, needle: str) -> list[tuple[str, tuple]]:
        return [c for c in self.calls if needle in " ".join(c[0].split())]

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return self._match(self.rows, sql) or []

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        got = self._match(self.rows, sql)
        return got[0] if got else None

    async def fetchval(self, sql, *args):
        self.calls.append((sql, args))
        got = self._match(self.vals, sql)
        return 0 if got is None and "count" in sql.lower() else got

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        return "OK"

    async def close(self) -> None:
        """The lifespan closes the pool on shutdown; a fake without this makes
        every test fail during teardown for a reason unrelated to the test."""


@pytest.fixture
def db(monkeypatch, admin_hash):
    monkeypatch.setenv("QD_SESSION_SECRET", "test-secret-not-used-anywhere-real")
    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH", admin_hash)
    monkeypatch.delenv("QD_DB_DSN", raising=False)
    fake = FakeDB(vals={"INSERT INTO control_commands": 7})
    monkeypatch.setattr(webapp.state, "db", fake)
    monkeypatch.setattr(webapp.state, "health_path", "/nonexistent/metrics.json")
    monkeypatch.setattr(webapp.state, "limiter", security.RateLimiter())
    return fake


@pytest.fixture
def client(db):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def auth(client, db):
    """A signed-in admin."""
    res = client.post("/login", data={"username": "admin", "password": PASSWORD},
                      follow_redirects=False)
    assert res.status_code == 303, res.text
    return client


def csrf_of(client) -> str:
    sess = security.read(client.cookies.get(security.SESSION_COOKIE),
                         webapp.state.secret)
    assert sess is not None
    return sess.csrf


PAGES = ["/", "/overview", "/positions", "/trades", "/strategies", "/risk",
         "/system", "/activity"]


# ── the empty database, which is the real one ────────────────────────────


@pytest.mark.parametrize("path", PAGES)
def test_every_page_renders_with_no_data(auth, path):
    """The state this system is actually in. A dashboard that breaks when
    nothing has happened is broken exactly when you first open it."""
    res = auth.get(path)
    assert res.status_code == 200, res.text[:600]
    assert "QuantDesk" in res.text


def test_overview_says_why_it_is_empty(auth):
    res = auth.get("/overview")
    assert "No equity marks recorded yet" in res.text
    # Not "0.00": a number the system does not have must never render as zero.
    assert "not because anything failed" in res.text


def test_partials_render(auth):
    for path in ["/partials/statusbar", "/partials/equity", "/partials/positions",
                 "/partials/signals"]:
        assert auth.get(path).status_code == 200, path


def test_engine_never_seen_is_said_plainly(auth):
    res = auth.get("/partials/statusbar")
    assert "NEVER SEEN" in res.text


# ── auth ─────────────────────────────────────────────────────────────────


def test_healthz_needs_no_session(client):
    res = client.get("/healthz")
    assert res.status_code == 200 and res.text == "ok"


def test_browser_is_redirected_to_login(client):
    res = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/login?next=/"


def test_api_gets_json_401_not_a_redirect(client):
    res = client.get("/api/v1/status")
    assert res.status_code == 401
    assert res.json()["error"]


def test_bad_password_is_refused_without_a_cookie(client):
    res = client.post("/login", data={"username": "admin", "password": "wrong"},
                      follow_redirects=False)
    assert res.status_code == 401
    assert security.SESSION_COOKIE not in res.cookies


def test_unknown_user_and_bad_password_are_indistinguishable(client):
    a = client.post("/login", data={"username": "admin", "password": "wrong"})
    b = client.post("/login", data={"username": "nobody", "password": "wrong"})
    assert a.status_code == b.status_code == 401
    # Same message: login must not tell an attacker which usernames exist.
    assert "invalid credentials" in a.text and "invalid credentials" in b.text


def test_login_is_rate_limited(client):
    for _ in range(security.MAX_ATTEMPTS):
        client.post("/login", data={"username": "admin", "password": "wrong"})
    res = client.post("/login", data={"username": "admin", "password": "wrong"})
    assert res.status_code == 429
    assert "Too many attempts" in res.text


def test_rate_limit_is_not_bypassable_with_x_forwarded_for(client, monkeypatch):
    """A rotating X-Forwarded-For must not mint a fresh lockout bucket.

    Regression: `_client_key` read the header unconditionally from any peer, so
    an attacker changing it per request was never rate limited at all — 7 of 7
    attempts allowed where 2 should have been blocked. `qd-web` publishes 8090
    directly and `qd-proxy` is profile-gated off, so nothing strips the header.
    """
    monkeypatch.delenv("QD_TRUSTED_PROXIES", raising=False)
    codes = []
    for i in range(security.MAX_ATTEMPTS + 3):
        res = client.post("/login", data={"username": "admin", "password": "wrong"},
                          headers={"X-Forwarded-For": f"10.0.0.{i}"})
        codes.append(res.status_code)
    assert 429 in codes, f"lockout never fired with a rotating header: {codes}"
    assert codes[-1] == 429


def test_x_forwarded_for_is_honoured_from_a_trusted_proxy(monkeypatch):
    """The header is still usable — but only when the peer is allowlisted."""
    from ct.web.app import _client_key

    class _Req:
        def __init__(self, peer, xff):
            self.headers = {"X-Forwarded-For": xff} if xff else {}
            self.client = type("C", (), {"host": peer})()

    monkeypatch.setenv("QD_TRUSTED_PROXIES", "127.0.0.1, 10.9.9.9")
    assert _client_key(_Req("10.9.9.9", "203.0.113.7")) == "203.0.113.7"
    assert _client_key(_Req("127.0.0.1", "203.0.113.7, 1.2.3.4")) == "203.0.113.7"
    # Untrusted peer: the header is ignored and the socket address wins.
    assert _client_key(_Req("198.51.100.4", "203.0.113.7")) == "198.51.100.4"
    # Trusted peer that sends no header falls back to the peer.
    assert _client_key(_Req("127.0.0.1", None)) == "127.0.0.1"
    monkeypatch.delenv("QD_TRUSTED_PROXIES")
    assert _client_key(_Req("198.51.100.4", "203.0.113.7")) == "198.51.100.4"


def _argon2(plain="pw"):
    from argon2 import PasswordHasher

    return PasswordHasher().hash(plain)


def test_password_hash_can_come_from_a_secret_file(monkeypatch, tmp_path):
    """The `_FILE` route exists because Compose eats `$` — see below."""
    h = _argon2()
    f = tmp_path / "admin_password_hash"
    f.write_text(h, encoding="utf-8")
    monkeypatch.delenv("QD_ADMIN_PASSWORD_HASH", raising=False)
    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH_FILE", str(f))
    assert security.known_users()["admin"][0] == h


def test_a_compose_mangled_env_hash_does_not_shadow_a_good_file(monkeypatch, tmp_path, caplog):
    """The defect this guards, found only by running the real compose stack.

    An argon2 hash always contains `$`, and Compose interpolates `$` in .env
    values. `QD_ADMIN_PASSWORD_HASH=$argon2id$v=19$m=65536,t=3,p=4$mkLF...`
    reached the container as `=19=65536,t=3,p=4+2DEg`, so every login returned
    401 with correct credentials — following the README exactly.

    Two things must hold: the mangled value is refused rather than passed to
    PasswordHasher (where it is indistinguishable from a wrong password), and it
    must not shadow a valid secret file, which a plain env-wins rule would allow.
    """
    import logging

    good = _argon2()
    f = tmp_path / "admin_password_hash"
    f.write_text(good, encoding="utf-8")
    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH", "=19=65536,t=3,p=4+2DEg")
    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH_FILE", str(f))

    with caplog.at_level(logging.ERROR, logger="ct.web.security"):
        users = security.known_users()
    assert users["admin"][0] == good, "the mangled env var shadowed the file"
    # getMessage(), not `message % args` — caplog has already interpolated.
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "does not look like an argon2 hash" in msg
    assert "$$" in msg, "the log should say how to fix it"


def test_a_mangled_hash_with_no_file_leaves_nobody_configured(monkeypatch, caplog):
    """Fail closed and say why, rather than 401-ing forever with no explanation."""
    import logging

    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH", "=19=65536,t=3,p=4+2DEg")
    monkeypatch.delenv("QD_ADMIN_PASSWORD_HASH_FILE", raising=False)
    for role in ("operator", "viewer"):
        monkeypatch.delenv(f"QD_{role.upper()}_PASSWORD_HASH", raising=False)
        monkeypatch.delenv(f"QD_{role.upper()}_PASSWORD_HASH_FILE", raising=False)
    with caplog.at_level(logging.ERROR, logger="ct.web.security"):
        assert security.known_users() == {}
    assert any("argon2" in r.message for r in caplog.records)


def test_env_hash_still_works_when_it_is_intact(monkeypatch):
    """The env route stays supported — you just have to double the `$`."""
    h = _argon2()
    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH", h)
    monkeypatch.delenv("QD_ADMIN_PASSWORD_HASH_FILE", raising=False)
    assert security.known_users()["admin"][0] == h


def test_an_unreadable_hash_file_is_not_fatal(monkeypatch, tmp_path):
    """Compose sets the _FILE variable unconditionally, so a deployment using
    the env route legitimately has no file there."""
    h = _argon2()
    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH", h)
    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH_FILE", str(tmp_path / "absent"))
    assert security.known_users()["admin"][0] == h


def test_malformed_password_hash_fails_closed_and_is_logged(monkeypatch, caplog):
    """An operator typo must not read as a wrong password with a silent log.

    The rejection now happens in `known_users()` rather than inside
    `PasswordHasher.verify` — earlier and with a more actionable message, since
    at that point we can say *which* variable is wrong and how to fix it. The
    outcome that matters is unchanged: AuthError, and a log line naming the
    problem.
    """
    import logging

    monkeypatch.setenv("QD_ADMIN_PASSWORD_HASH", "definitely-not-an-argon2-hash")
    monkeypatch.delenv("QD_ADMIN_PASSWORD_HASH_FILE", raising=False)
    with (caplog.at_level(logging.ERROR, logger="ct.web.security"),
          pytest.raises(security.AuthError)):
        security.verify_password("admin", "anything")
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "argon2" in msg and "QD_ADMIN_PASSWORD_HASH" in msg, caplog.text


def test_successful_login_sets_a_session_and_audits_it(client, db):
    res = client.post("/login", data={"username": "admin", "password": PASSWORD},
                      follow_redirects=False)
    assert res.status_code == 303
    cookie = client.cookies.get(security.SESSION_COOKIE)
    assert cookie
    sess = security.read(cookie, webapp.state.secret)
    assert sess and sess.role == "admin"
    assert db.wrote("INSERT INTO audit_log")


def test_next_is_not_an_open_redirect(client):
    res = client.post("/login", data={"username": "admin", "password": PASSWORD,
                                      "next": "//evil.example/x"},
                      follow_redirects=False)
    assert res.headers["location"] == "/"


# ── control ──────────────────────────────────────────────────────────────


def test_control_without_csrf_is_refused(auth, db):
    res = auth.post("/control/halt", json={})
    assert res.status_code == 403
    assert not db.wrote("INSERT INTO control_commands")


def test_halt_queues_intent_and_audits_it(auth, db):
    res = auth.post("/control/halt", json={}, headers={"X-CSRF-Token": csrf_of(auth)})
    assert res.status_code == 200
    body = res.json()
    assert body["queued"] is True and body["command"] == "halt"
    queued = db.wrote("INSERT INTO control_commands")
    assert len(queued) == 1 and queued[0][1][1] == "halt"
    assert db.wrote("INSERT INTO audit_log")


def test_flatten_note_says_queued_and_where_completion_shows(auth):
    """Flatten is real since 2026-07-29, but queued is still not done: the
    note must say the engine closes on its next cycle and where completion
    shows up, not claim positions are already flat."""
    res = auth.post("/control/flatten_all", json={},
                    headers={"X-CSRF-Token": csrf_of(auth)})
    assert res.status_code == 200
    note = res.json()["note"]
    assert "next cycle" in note and "FLATTEN_COMPLETE" in note
    assert "does not liquidate" not in note


def test_unknown_command_is_rejected(auth, db):
    res = auth.post("/control/sell_everything", json={},
                    headers={"X-CSRF-Token": csrf_of(auth)})
    assert res.status_code == 400
    assert not db.wrote("INSERT INTO control_commands")


def test_the_dashboard_cannot_arm(auth):
    """Arming is not reachable from here at all — there is no route, not merely
    a role check. The phrase can only be typed at a terminal."""
    assert auth.post("/control/arm_live", json={},
                     headers={"X-CSRF-Token": csrf_of(auth)}).status_code == 400
    assert "/control/arm-live" not in [r.path for r in webapp.app.routes
                                       if hasattr(r, "path")]


# ── roles ────────────────────────────────────────────────────────────────


def test_every_role_may_halt():
    """Stopping is never gated (ARCHITECTURE.md §9). A viewer who can see the
    system on fire can stop it."""
    for role in security.ROLES:
        sess = security.Session("u", role, 0, 9e9, "c")
        assert sess.may("halt") is True


def test_a_viewer_may_not_resume_or_flatten():
    viewer = security.Session("u", "viewer", 0, 9e9, "c")
    assert not viewer.may("resume")
    assert not viewer.may("flatten")
    operator = security.Session("u", "operator", 0, 9e9, "c")
    assert operator.may("resume") and operator.may("flatten")
    assert not operator.may("reload_config")


def test_role_is_enforced_on_the_endpoint(client, db, admin_hash, monkeypatch):
    monkeypatch.setenv("QD_VIEWER_PASSWORD_HASH", admin_hash)
    client.post("/login", data={"username": "viewer", "password": PASSWORD})
    token = csrf_of(client)
    assert client.post("/control/halt", json={},
                       headers={"X-CSRF-Token": token}).status_code == 200
    denied = client.post("/control/resume", json={},
                         headers={"X-CSRF-Token": token})
    assert denied.status_code == 403


# ── headers ──────────────────────────────────────────────────────────────


def test_security_headers_allow_nothing_third_party(auth):
    res = auth.get("/")
    csp = res.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "http://" not in csp and "https://" not in csp
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["x-content-type-options"] == "nosniff"


def test_no_page_loads_a_third_party_asset(auth):
    """The corollary of the CSP: nothing to block, because nothing is asked for."""
    for path in PAGES:
        body = auth.get(path).text
        assert "cdn." not in body and "https://unpkg" not in body
        assert "//cdnjs" not in body


# ── metrics ──────────────────────────────────────────────────────────────


def test_metrics_needs_a_session_from_a_non_loopback_client(client):
    # TestClient presents as "testclient", not 127.0.0.1.
    assert client.get("/metrics").status_code == 401


def test_metrics_renders_prometheus_text(auth):
    res = auth.get("/metrics")
    assert res.status_code == 200
    assert "qd_web_up 1" in res.text
    assert "# TYPE qd_engine_halted gauge" in res.text


# ── json api ─────────────────────────────────────────────────────────────


def test_api_status_shape(auth):
    body = auth.get("/api/v1/status").json()
    assert set(body) == {"engine", "performance", "arming", "positions"}
    assert body["engine"]["seen"] is False
    assert body["performance"]["has_data"] is False


def test_api_endpoints_return_json(auth):
    for path in ["/api/v1/positions", "/api/v1/equity", "/api/v1/trades",
                 "/api/v1/events", "/api/v1/risk"]:
        res = auth.get(path)
        assert res.status_code == 200, path
        assert isinstance(res.json(), dict)


# ── populated pages ──────────────────────────────────────────────────────


@pytest.fixture
def seeded(db):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    db.rows.update({
        "payload->>'component' = 'engine'": [{
            "ts": now - timedelta(seconds=30),
            "payload": {"component": "engine", "mode": "paper", "halted": False,
                        "kill": False, "armed": True, "reconciled": True,
                        "may_open": True, "may_close": True, "cycles": 412,
                        "signals": 9, "orders": 2, "open_books": 1,
                        "paused_strategies": []}}],
        # Needles must be distinctive: `equity_latest`'s statement is a
        # substring of `performance`'s CTE, and a loose match hands one
        # query's row to the other (KeyError three frames away).
        "SELECT ts, equity, cash, positions_value": [{
            "ts": now, "equity": 10420.5, "cash": 9000.0,
            "positions_value": 1420.5, "gross_exposure": 0.137,
            "drawdown": 0.021}],
        "AS day_ago": [{
            "equity": 10420.5, "ts": now, "day_ago": 10300.0,
            "week_ago": 10000.0, "first_equity": 10000.0, "peak": 10500.0,
            "max_dd": 0.021, "marks": 4210}],
        "FROM positions WHERE account_id = $1 AND qty <> 0": [{
            "strategy": "donchian_breakout", "symbol": "BTC/USD",
            "qty": 0.0123, "avg_entry": 63500.0, "realized_pnl": 12.5,
            "fees_paid": 6.0, "stop_price": 61000.0, "target_price": None,
            "trail_atr_mult": 2.5, "opened_at": now, "updated_at": now}],
        "FROM signals WHERE account_id = $1 ORDER BY ts DESC": [{
            "ts": now, "strategy": "donchian_breakout", "symbol": "BTC/USD",
            "direction": 1, "strength": 1.0, "ref_price": 63500.0,
            "accepted": False, "reject_code": "EDGE_BELOW_FEES",
            "reject_detail": "edge 12 bps < 60 bps", "sized_qty": None}],
        "GROUP BY 1 ORDER BY n DESC": [
            {"code": "EDGE_BELOW_FEES", "n": 14, "last_ts": now},
            {"code": None, "n": 3, "last_ts": now}],
        "FROM candles WHERE exchange = $1": [{
            "symbol": "BTC/USD", "timeframe": "4h", "bars": 168,
            "earliest": now - timedelta(days=28), "latest": now,
            "imported": 0}],
    })
    return db


def test_overview_renders_real_numbers(auth, seeded):
    res = auth.get("/overview")
    assert res.status_code == 200, res.text[:600]
    assert "10,420.50" in res.text          # equity, tabular
    assert "donchian_breakout" in res.text  # the open book
    assert "EDGE_BELOW_FEES" in res.text    # rejection histogram
    # The fee gate firing is the system working, and the page says so.
    assert "working as designed" in res.text


def test_a_rejection_with_no_recorded_reason_is_labelled_not_dropped(auth, seeded):
    """A NULL reject_code is a real rejection whose reason was never written.
    Dropping it would quietly shrink the histogram it belongs in."""
    assert "UNRECORDED" in auth.get("/overview").text


def test_status_bar_shows_trading_when_the_engine_says_so(auth, seeded):
    assert "TRADING" in auth.get("/partials/statusbar").text


# ── SSE ──────────────────────────────────────────────────────────────────


def test_stream_ends_on_its_own_deadline(auth, monkeypatch):
    """An unbounded generator holds a task, a connection and a database handle
    until the browser tab closes — and `is_disconnected()` is not reliable on
    every transport. This hung the suite until the deadline existed."""
    monkeypatch.setenv("QD_SSE_MAX_SECONDS", "0.2")
    res = auth.get("/stream")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.text.startswith("retry:")


# ── charts (pure, no browser) ────────────────────────────────────────────


def test_equity_chart_refuses_to_draw_one_point():
    svg = charts.equity_chart([{"ts": "t", "equity": 100.0, "drawdown": 0.0}])
    assert "chart--empty" in svg and "No equity marks yet" in svg


def test_equity_chart_draws_a_path_and_ships_hover_points():
    series = [{"ts": f"2026-07-{d:02d}", "equity": 100.0 + d, "drawdown": d / 1000}
              for d in range(1, 15)]
    svg = charts.equity_chart(series)
    assert "<path" in svg and "eq-line--up" in svg
    assert "data-points=" in svg and "chart-hit" in svg


def test_equity_chart_survives_a_perfectly_flat_series():
    """A flat account is the normal paper-trading state; a zero-height range
    would divide by zero and take the whole page down."""
    series = [{"ts": str(i), "equity": 10_000.0, "drawdown": 0.0} for i in range(20)]
    svg = charts.equity_chart(series)
    assert "chart--equity" in svg and "nan" not in svg.lower()


def test_donut_separates_unknown_from_zero():
    assert "—" in charts.donut(None)
    assert "0%" in charts.donut(0.0) and "—" not in charts.donut(0.0)


def test_meter_colours_by_fraction_of_limit():
    assert "meter--ok" in charts.meter(0.1, 1.0)
    assert "meter--warn" in charts.meter(0.75, 1.0)
    assert "meter--danger" in charts.meter(0.95, 1.0)
    assert "OVER" in charts.meter(1.4, 1.0)


def test_sparkline_handles_a_series_too_short_to_draw():
    assert "spark--empty" in charts.sparkline([])
    assert "spark--empty" in charts.sparkline([1.0])
    assert "<path" in charts.sparkline([1.0, 2.0, 1.5])


def test_charts_escape_their_labels():
    svg = charts.hbars([('<script>alert(1)</script>', 3)])
    assert "<script>" not in svg and "&lt;script&gt;" in svg


# ── formatting ───────────────────────────────────────────────────────────


def test_a_missing_number_is_never_zero():
    assert fmt.money(None) == "—"
    assert fmt.pct(None) == "—"
    assert fmt.qty(None) == "—"
    assert fmt.signed_pct(None) == "—"
    assert fmt.money(0) == "0.00"          # a real zero still reads as zero


def test_qty_keeps_precision_a_money_format_would_destroy():
    """0.00005 BTC is a legal Kraken order (`ordermin`). Rendering it as 0.00
    would show a real position as empty."""
    assert fmt.qty(0.00005) == "0.00005"
    assert fmt.money(0.00005) == "0.00"


def test_tone_maps_sign_to_colour():
    assert fmt.tone(1) == "up" and fmt.tone(-1) == "down"
    assert fmt.tone(0) == "flat" and fmt.tone(None) == "flat"


def test_event_label_drops_the_namespace():
    assert fmt.event_label("execution.order_filled") == "order filled"
    assert fmt.event_group("execution.order_filled") == "execution"


# ── security unit ────────────────────────────────────────────────────────


def test_session_round_trip_and_tamper_detection():
    secret = b"a-secret"
    cookie, sess = security.issue("admin", "admin", secret)
    assert security.read(cookie, secret).user == sess.user
    payload, _, sig = cookie.rpartition(".")
    assert security.read(f"{payload}.{sig[:-2]}xx", secret) is None
    assert security.read(cookie, b"another-secret") is None
    assert security.read("garbage", secret) is None
    assert security.read(None, secret) is None


def test_an_expired_session_is_not_read():
    secret = b"a-secret"
    cookie, _ = security.issue("admin", "admin", secret, now=0)
    assert security.read(cookie, secret, now=security.SESSION_TTL_S + 1) is None


def test_a_forged_role_does_not_survive_the_signature():
    import base64
    import json as _json

    secret = b"a-secret"
    cookie, _ = security.issue("v", "viewer", secret)
    payload, _, sig = cookie.rpartition(".")
    data = _json.loads(base64.urlsafe_b64decode(payload + "=="))
    data["r"] = "admin"
    forged = base64.urlsafe_b64encode(
        _json.dumps(data).encode()).decode().rstrip("=")
    assert security.read(f"{forged}.{sig}", secret) is None


def test_session_secret_refuses_to_invent_one(monkeypatch):
    monkeypatch.delenv("QD_SESSION_SECRET", raising=False)
    monkeypatch.delenv("QD_SESSION_SECRET_FILE", raising=False)
    with pytest.raises(RuntimeError, match="no session secret"):
        security.session_secret()


def test_rate_limiter_window():
    limiter = security.RateLimiter(max_attempts=2, window=100)
    assert limiter.check("ip", now=0) == 0
    limiter.fail("ip", now=0)
    limiter.fail("ip", now=1)
    assert limiter.check("ip", now=2) > 0
    assert limiter.check("ip", now=200) == 0     # window slid past


# ── queries helpers ──────────────────────────────────────────────────────


def test_host_health_absent_file_is_not_a_healthy_idle_machine():
    from ct.web.queries import host_health

    out = host_health("/definitely/not/here.json")
    assert out["available"] is False and "not mounted" in out["reason"]
    # Crucially not zeros, which would render as a calm, idle host.
    assert "cpu_pct" not in out


def test_freshness_is_measured_in_bars_not_seconds():
    from ct.web.queries import freshness

    assert freshness(60, "4h") == "ok"          # one minute into a 4h bar
    assert freshness(6 * 3600, "4h") == "ok"    # 1.5 bars is still current
    assert freshness(8 * 3600, "4h") == "slow"  # 2 bars
    assert freshness(20 * 3600, "4h") == "stale"
    assert freshness(None, "4h") == "stale"


def test_worst_freshness_scans_every_series():
    """Regression: this was computed with `{% set %}` inside a Jinja `{% for %}`,
    whose scope does not escape the loop — so it reported the first series'
    state as if it were the worst."""
    coverage = [{"age_s": 10, "timeframe": "4h"},
                {"age_s": 20 * 3600, "timeframe": "4h"},
                {"age_s": 10, "timeframe": "4h"}]
    assert webapp._worst_freshness(coverage) == "stale"
    assert webapp._worst_freshness([{"age_s": 10, "timeframe": "4h"}]) == "ok"


def test_rejection_bars_label_the_unrecorded_bucket():
    assert webapp._rejection_bars([{"code": None, "n": 2}]) == [("UNRECORDED", 2.0)]


# ── the Home page (2026-07-30): plain English for a non-quant operator ───
# The root path IS the plain page — typing the bare address must never land
# a non-expert on a wall of metrics.


def test_home_is_the_root_and_renders_on_an_empty_database(auth):
    res = auth.get("/")
    assert res.status_code == 200
    assert "paper trading" in res.text
    assert "No strategy is enabled" in res.text
    # With no heartbeat ever recorded, the honest verdict is bad, not green.
    assert "The engine has gone quiet" in res.text


def test_home_with_a_running_paper_strategy_says_so_in_sentences(auth, seeded):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    seeded.rows["status_reason, updated_at FROM strategies"] = [{
        "name": "donchian_breakout", "status": "paper",
        "status_reason": "enabled by cli",
        "updated_at": now - timedelta(days=2)}]
    res = auth.get("/")
    assert res.status_code == 200
    assert "Everything is fine" in res.text
    assert "trend-follower" in res.text, "the strategy gets a lay description"
    assert "Practice day 3 of 90" in res.text
    assert "practice money" in res.text
    assert "No real money is at risk" in res.text


def test_home_shows_a_halt_as_needing_attention(auth, seeded):
    seeded.rows["payload->>'component' = 'engine'"][0]["payload"].update(
        halted=True, halt_reason="reconciliation mismatch")
    res = auth.get("/")
    assert "Trading is halted" in res.text
    assert "reconciliation mismatch" in res.text


def test_plain_verdict_priorities():
    """Kill outranks stale outranks halt outranks warnings outranks fine."""
    from ct.web import plain

    base = {"seen": True, "stale": False, "halted": False, "kill": False}
    assert plain.verdict({**base, "kill": True}, [])["level"] == "bad"
    assert plain.verdict({**base, "stale": True}, [])["level"] == "bad"
    assert plain.verdict({**base, "halted": True, "halt_reason": "x"},
                         [])["level"] == "bad"
    warn = [{"severity": "warn", "code": "X", "detail": "d"}]
    assert plain.verdict(base, warn)["level"] == "warn"
    assert plain.verdict(base, [])["level"] == "ok"
    # Info events are not attention: a completed flatten is not a warning.
    assert plain.verdict(base, [])["headline"] == "Everything is fine"


def test_plain_build_counts_paper_days_from_the_enable_timestamp():
    from datetime import datetime, timedelta, timezone

    from ct.web import plain

    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    out = plain.build(
        engine={"seen": True, "stale": False, "halted": False, "kill": False},
        perf={"has_data": True, "equity": 10100.0},
        positions=[], risk_events=[],
        strategies=[{"name": "donchian_breakout", "status": "paper",
                     "updated_at": now - timedelta(days=9)}],
        fills_count=3, mode="paper", now=now)
    assert out["paper_days"] == 10
    assert "up $100.00" in out["money"]
    assert "3 simulated trades" in out["activity"]
    # An info-severity risk event lands in FYI, never in attention.
    out2 = plain.build(
        engine={"seen": True, "stale": False, "halted": False, "kill": False},
        perf={"has_data": False}, positions=[],
        risk_events=[{"severity": "info", "code": "FLATTEN_COMPLETE"}],
        strategies=[], fills_count=0, mode="paper", now=now)
    assert out2["attention"] == []
    assert out2["fyi"] and out2["verdict"]["level"] == "ok"


def test_engine_query_asks_for_the_heartbeat_type_the_engine_writes(auth, db):
    """Regression, found by a human reading the Home page. The query said
    'system.heartbeat' while the engine publishes EventType.HEARTBEAT
    ('ops.heartbeat'), so the dashboard NEVER saw a heartbeat and called a
    healthy engine "gone quiet". The FakeDB matches this query by a different
    substring, so only pinning the literal here catches the drift."""
    from ct.events.types import EventType

    auth.get("/")
    heartbeat_sql = [sql for sql, _ in db.calls if "payload->>'component'" in sql]
    assert heartbeat_sql, "the engine query ran"
    for sql in heartbeat_sql:
        assert f"'{EventType.HEARTBEAT.value}'" in sql
        assert "system.heartbeat" not in sql


def test_staleness_threshold_tolerates_the_normal_beat_gap():
    """The engine beats every ~300 s; a threshold equal to the interval calls
    ordinary jitter 'quiet'. Stale must mean a whole missed beat."""
    from ct.web import queries

    assert queries.HEARTBEAT_STALE_S >= 2 * 300, (
        "threshold must exceed twice the beat interval")


# ── resolving a risk event (2026-07-30) ──────────────────────────────────
# Found in production: the Risk page SHOWED open events and offered no way to
# resolve one — he typed CONFIRM into the only modal there (Resume) three
# times, which sent harmless no-op resumes and cleared nothing. The button
# now exists and queues `resolve_risk_event`; the engine stays the single
# writer of risk_events.


def _open_event(now):
    return {"id": 42, "ts": now, "severity": "info", "code": "FLATTEN_COMPLETE",
            "detail": "all positions closed by flatten request",
            "payload": {}, "resolved_at": None}


def test_risk_page_offers_resolve_on_open_events(auth, db):
    from datetime import datetime, timezone
    db.rows["FROM risk_events"] = [_open_event(datetime.now(timezone.utc))]
    res = auth.get("/risk")
    assert 'data-control="resolve_risk_event"' in res.text
    assert 'data-event-id="42"' in res.text
    assert "does not change any trading state" in res.text


def test_resolve_risk_event_queues_a_command_with_the_id(auth, db):
    res = auth.post("/control/resolve_risk_event", json={"id": "42"},
                    headers={"X-CSRF-Token": csrf_of(auth)})
    assert res.status_code == 200
    call = db.wrote("INSERT INTO control_commands")[0]
    assert call[1][1] == "resolve_risk_event"
    assert "42" in call[1][2], "the event id must reach the engine"


def test_a_viewer_may_not_resolve_events():
    viewer = security.Session("u", "viewer", 0, 9e9, "c")
    assert not viewer.may("resolve_event")
    operator = security.Session("u", "operator", 0, 9e9, "c")
    assert operator.may("resolve_event")


def test_overview_banner_ignores_info_severity_events(auth, db):
    """An acknowledgement record (a completed flatten) must not render as a
    red 'unresolved risk event' alarm — that is what sent a non-quant
    operator hunting for a resolve button that did not exist."""
    from datetime import datetime, timezone
    db.rows["FROM risk_events"] = [_open_event(datetime.now(timezone.utc))]
    res = auth.get("/overview")
    assert "unresolved risk event" not in res.text
    critical = dict(_open_event(datetime.now(timezone.utc)), severity="critical")
    db.rows["FROM risk_events"] = [critical]
    res = auth.get("/overview")
    assert "unresolved risk event" in res.text


def test_system_page_survives_a_database_without_schema_version(auth, db):
    """Regression, found by the operator: /system returned 500 on the production
    database. `schema_version` exists only when `bootstrap()` created the
    schema; the NAS database was initialised by the Postgres entrypoint mount
    and has no marker. The FakeDB's None-for-unmatched behaviour hid this —
    the real database RAISES, so this fake must too."""
    from ct.store.db import DatabaseError

    original = db.fetchrow

    async def raising_fetchrow(sql, *args):
        if "schema_version" in sql:
            raise DatabaseError('relation "schema_version" does not exist')
        return await original(sql, *args)

    db.fetchrow = raising_fetchrow
    res = auth.get("/system")
    assert res.status_code == 200, res.text[:600]
    assert "not stamped" in res.text


# ── two-tier navigation (2026-07-30): "a lot of bloat" — the operator ───────
# Progressive disclosure: Home/Positions/Trades stay primary; the expert
# pages fold behind a Details group. Folded, never removed.


def test_nav_puts_expert_pages_behind_a_details_group(auth):
    res = auth.get("/")
    body = res.text
    assert 'id="nav-advanced"' in body
    adv = body.split('id="nav-advanced"')[1]
    for href in ("/overview", "/strategies", "/risk", "/system", "/activity"):
        assert f'href="{href}"' in adv, f"{href} must stay reachable"
    main = body.split('id="nav-advanced"')[0]
    for href in ('href="/positions"', 'href="/trades"'):
        assert href in main


def test_details_group_opens_when_an_advanced_page_is_active(auth):
    assert '<details class="nav-adv" id="nav-advanced" open>' in auth.get("/risk").text
    assert '<details class="nav-adv" id="nav-advanced" open>' not in auth.get("/").text


def test_positions_page_leads_with_a_plain_sentence(auth):
    res = auth.get("/positions")
    assert "holds nothing right now" in res.text
    assert "Days of this are normal" in res.text


def test_trades_page_leads_with_a_plain_sentence(auth):
    res = auth.get("/trades")
    assert "No trades yet" in res.text


def test_position_and_trade_ledes_count_correctly():
    from ct.web import plain

    two = plain.positions_lede([{"symbol": "BTC/USD"}, {"symbol": "ETH/USD"}],
                               "paper")
    assert "2 positions" in two and "practice money" in two
    assert "3 practice trades" in plain.trades_lede(3, "paper")
    assert "1 trade " in plain.trades_lede(1, "live")
