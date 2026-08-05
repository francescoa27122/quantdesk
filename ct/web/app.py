"""`qd-web` — the dashboard and JSON API.

    uvicorn ct.web.app:app --host 0.0.0.0 --port 8090

## The constraint that shapes everything

**`qd-engine` is the single writer of trading state (ARCHITECTURE.md §1).** This
process reads that state and writes exactly one kind of row: `control_commands`,
which is *intent*. The engine polls it and decides. Nothing here can place an
order, cancel one, or touch `orders`, `fills` or `positions` — not because it is
careful, but because no such code path exists. It also holds no exchange
credentials, so a compromise of this container costs you a dashboard.

The one other table it appends to is `audit_log`, because an audit log that
does not record who pressed the button is not an audit log.

## Why server-rendered

Every page is complete HTML when it arrives. HTMX swaps fragments for the live
regions and SSE pushes the header, so the common case is a few hundred bytes
over the wire instead of a JSON payload plus a client-side render. On a 320 MB
container serving one person that is the right shape — and it means the page
still works with JavaScript disabled, which is a property worth having in the
thing you open when something is broken.

## Availability, stated honestly

The database is a hard dependency: if `qd-db` is down, pages return 503 with
the reason rather than an empty dashboard that looks like a calm, flat account.
`/healthz` is the *liveness* of this process only and deliberately does not
check the database — a health check that fails because a dependency is down
gets the wrong container restarted.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ct.web import charts, security
from ct.web.format import FILTERS
from ct.web.queries import Queries, freshness, host_health, timedelta_words

log = logging.getLogger("ct.web")

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
STATIC = HERE / "static"

# Every source is 'self'. No CDN, no inline script, no third-party origin —
# which is only possible because the charts are server-rendered SVG.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; "
       "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

# Two tiers (2026-07-30, at the operator's request — "a lot of bloat"). The
# research-backed shape is progressive disclosure: the pages a non-expert
# uses daily stay front and center; the expert layer collapses behind one
# click but is NEVER removed — hiding capability from the person debugging
# at 3 a.m. would be the opposite failure.
NAV_MAIN = [
    ("/", "Home", "home"),
    ("/positions", "Positions", "positions"),
    ("/trades", "Trades", "trades"),
]
NAV_ADVANCED = [
    ("/overview", "Overview", "overview"),
    ("/strategies", "Strategies", "strategies"),
    ("/risk", "Risk", "risk"),
    ("/system", "System", "system"),
    ("/activity", "Activity", "activity"),
]
NAV = NAV_MAIN + NAV_ADVANCED
ADVANCED_KEYS = {k for _h, _l, k in NAV_ADVANCED}


# ── application state ────────────────────────────────────────────────────


class WebState:
    """Everything the app needs that is not a request."""

    def __init__(self) -> None:
        self.db: Any = None
        self.secret: bytes = b""
        self.limiter = security.RateLimiter()
        self.account_id = int(os.environ.get("QD_ACCOUNT_ID", "1"))
        self.health_path = os.environ.get("QD_HOST_HEALTH",
                                          "/hosthealth/metrics.json")
        self.started = time.time()
        self.mode = os.environ.get("QD_MODE", "paper")

    def queries(self) -> Queries:
        if self.db is None:
            raise HTTPException(503, "database is not connected")
        return Queries(self.db, self.account_id)


state = WebState()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state.secret = security.session_secret()
    dsn = os.environ.get("QD_DB_DSN")
    if dsn:
        from ct.store import Database

        state.db = Database(dsn, max_size=6)
        try:
            await state.db.connect(retries=10, backoff=1.5)
        except Exception:
            # Start anyway and serve 503s with the reason. A dashboard that
            # refuses to boot because the database is slow to come up is a
            # dashboard you cannot use to find out why.
            log.exception("database unavailable at boot; serving degraded")
    else:
        log.warning("QD_DB_DSN not set — running without a database")
    try:
        yield
    finally:
        if state.db is not None:
            await state.db.close()


templates = Jinja2Templates(directory=str(TEMPLATES))
templates.env.filters.update(FILTERS)
templates.env.globals.update(
    charts=charts, nav=NAV, nav_main=NAV_MAIN, nav_advanced=NAV_ADVANCED,
    advanced_keys=ADVANCED_KEYS, timedelta_words=timedelta_words,
    freshness=freshness)


# ── auth plumbing ────────────────────────────────────────────────────────


def current_session(request: Request) -> security.Session:
    sess = security.read(request.cookies.get(security.SESSION_COOKIE),
                         state.secret)
    if sess is None:
        raise HTTPException(401, "not signed in")
    return sess


def require_csrf(request: Request, sess: security.Session) -> None:
    """Double-submit on every state-changing request.

    `SameSite=Strict` already blocks the cross-site POST, and this is the
    second lock on the same door: browsers have relaxed SameSite semantics
    before, and the door in question queues commands to a trading engine.
    """
    token = request.headers.get("X-CSRF-Token", "")
    # compare_digest, not `!=`: the token is a 24-byte secret and every other
    # secret comparison in this codebase is already constant-time.
    if not token or not hmac.compare_digest(token, sess.csrf):
        raise HTTPException(403, "csrf token missing or stale — reload the page")


def _trusted_proxies() -> frozenset[str]:
    """Peer addresses whose `X-Forwarded-For` we believe. Empty by default."""
    raw = os.environ.get("QD_TRUSTED_PROXIES", "")
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def _client_key(request: Request) -> str:
    """The login rate-limit bucket. NEVER trust `X-Forwarded-For` blindly.

    This read the header unconditionally, from any peer. `qd-web` publishes
    8090 directly and `qd-proxy` is `profiles: [tls]` — off unless you ask for
    it — so in the shipped configuration nothing sanitises that header. An
    attacker rotating it got a fresh bucket per request and the 5-attempt /
    900-second lockout never fired once. Verified before the fix: 7 of 7
    attempts allowed with a rotating header, 2 of 7 blocked with a fixed one.

    Uvicorn's own `--proxy-headers` trust list does not help, because this
    function reads the raw header rather than `request.client`.

    Falling back to the socket peer is the safe default: behind a real proxy
    every request buckets together (strict, and the operator sets
    QD_TRUSTED_PROXIES to fix it); with no proxy it is the true client.
    """
    peer = request.client.host if request.client else "unknown"
    if peer in _trusted_proxies():
        forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return peer


async def _page_context(request: Request, sess: security.Session,
                        active: str) -> dict[str, Any]:
    """The header every page shares: what the engine believes, right now."""
    q = state.queries()
    engine = await q.engine()
    arming = await q.arming()
    return {
        "request": request, "session": sess, "active": active,
        "engine": engine, "arming": arming, "mode": state.mode,
        "csrf": sess.csrf, "now": datetime.now(timezone.utc),
    }


async def _limits(q: Queries) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """`(effective, defaults, configured)`.

    The engine falls back to the `RiskLimits` dataclass when `risk_config` is
    empty, so a page showing only the stored row would show nothing while the
    engine was enforcing plenty. Effective limits are what is actually applied;
    `configured` exists so the page can mark which ones were overridden.
    """
    from ct.risk.engine import RiskLimits

    defaults = {f.name: getattr(RiskLimits(), f.name)
                for f in RiskLimits.__dataclass_fields__.values()}
    configured = (await q.risk_config()).get("limits", {})
    return {**defaults, **configured}, defaults, configured


def _rejection_bars(rejections: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """A NULL `reject_code` is a rejection whose reason was never recorded —
    labelled, not dropped, because a silent bucket is how a category of
    rejection disappears from the histogram it belongs in."""
    return [(r["code"] or "UNRECORDED", float(r["n"])) for r in rejections]


def _worst_freshness(coverage: list[dict[str, Any]]) -> str:
    """The worst state across every stored series.

    Computed here rather than in the template: Jinja's `{% set %}` inside a
    `{% for %}` does not escape the loop, so the template version silently
    reported the *first* series' state as if it were the worst.
    """
    order = {"ok": 0, "slow": 1, "stale": 2}
    worst = "ok"
    for c in coverage:
        state_ = freshness(c.get("age_s"), c.get("timeframe", "4h"))
        if order[state_] > order[worst]:
            worst = state_
    return worst


# ── routers ──────────────────────────────────────────────────────────────

public = APIRouter()
pages = APIRouter(dependencies=[Depends(current_session)])
api = APIRouter(prefix="/api/v1", dependencies=[Depends(current_session)])


@public.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    """This process only. Deliberately does not touch the database — see the
    module docstring."""
    return "ok"


@public.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "/") -> Response:
    if security.read(request.cookies.get(security.SESSION_COOKIE), state.secret):
        return RedirectResponse(next or "/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {
        "error": None, "next": next,
        "configured": bool(security.known_users())})


@public.post("/login", response_class=HTMLResponse)
async def login(request: Request) -> Response:
    """Decodes the body with `urllib.parse` rather than a form library.

    Both FastAPI's `Form(...)` and Starlette's `request.form()` import
    `python-multipart` — a parser for a content type this app never receives,
    since an HTML login form posts `application/x-www-form-urlencoded`. Three
    lines of stdlib avoid a dependency in the image that also runs the trading
    engine. Measured, not assumed: the import error is what sent me here.
    """
    from urllib.parse import parse_qs

    raw = (await request.body()).decode("utf-8", "replace")
    form = parse_qs(raw, keep_blank_values=True)
    username = (form.get("username") or [""])[0]
    password = (form.get("password") or [""])[0]
    next = (form.get("next") or ["/"])[0] or "/"
    key = _client_key(request)
    wait = state.limiter.check(key)
    if wait:
        return templates.TemplateResponse(
            request, "login.html",
            {"error": f"Too many attempts. Try again in {wait // 60 + 1} min.",
             "next": next, "configured": True}, status_code=429)
    try:
        role = security.verify_password(username.strip(), password)
    except security.AuthError as exc:
        state.limiter.fail(key)
        log.warning("failed login for %r from %s", username[:32], key)
        return templates.TemplateResponse(
            request, "login.html",
            {"error": str(exc), "next": next, "configured": True},
            status_code=401)

    state.limiter.clear(key)
    cookie, sess = security.issue(username.strip(), role, state.secret)
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    resp = RedirectResponse(target, status_code=303)
    resp.set_cookie(security.SESSION_COOKIE, cookie, httponly=True,
                    samesite="strict", max_age=security.SESSION_TTL_S,
                    secure=request.url.scheme == "https", path="/")
    with contextlib.suppress(Exception):
        await state.db.execute(
            """INSERT INTO audit_log (actor, action, target)
               VALUES ($1, 'login', $2)""", sess.user, key)
    return resp


@public.post("/logout")
async def logout() -> Response:
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(security.SESSION_COOKIE, path="/")
    return resp


# ── pages ────────────────────────────────────────────────────────────────


@pages.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Response:
    """The plain-English page — the one a non-quant reads first, and
    therefore the root: typing the bare address must land here, not on a
    wall of metrics. Sentences, not numbers: `ct/web/plain.py` phrases what
    `Queries` reads. The Overview (now /overview) and everything after it is
    the detail layer for whoever wants the numbers.
    """
    from ct.web import plain

    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "home")
    ctx["plain"] = plain.build(
        engine=ctx["engine"],
        perf=await q.performance(),
        positions=await q.positions(),
        risk_events=await q.risk_events(unresolved_only=True, limit=10),
        strategies=await q.strategy_states(),
        fills_count=await q.fills_count(),
        mode=state.mode,
    )
    return templates.TemplateResponse(request, "home.html", ctx)


@pages.get("/overview", response_class=HTMLResponse)
async def overview(request: Request, hours: int = 168) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "overview")
    limits, _defaults, configured = await _limits(q)
    rejections = await q.rejections(7)
    ctx.update(
        perf=await q.performance(),
        series=await q.equity_series(hours),
        hours=hours,
        positions=await q.positions(),
        exposure=await q.symbol_exposure(),
        latest=await q.equity_latest(),
        strategies=await q.strategy_stats(),
        signals=await q.recent_signals(12),
        rejections=rejections,
        rejection_bars=_rejection_bars(rejections),
        execution=await q.execution_quality(30),
        # The banner is an alarm; info-severity records (e.g. a completed
        # flatten) are acknowledgements, not alarms — same split as Home.
        risk_events=[e for e in await q.risk_events(unresolved_only=True,
                                                    limit=10)
                     if e.get("severity") in ("warn", "critical")][:5],
        coverage=await q.data_coverage(),
        limits=limits,
        has_config=bool(configured),
    )
    return templates.TemplateResponse(request, "overview.html", ctx)


@pages.get("/positions", response_class=HTMLResponse)
async def positions_page(request: Request) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "positions")
    from ct.web import plain
    positions = await q.positions()
    ctx.update(positions=positions, exposure=await q.symbol_exposure(),
               open_orders=await q.open_orders(),
               orders=await q.recent_orders(60), fills=await q.recent_fills(60),
               execution=await q.execution_quality(30),
               lede=plain.positions_lede(positions, state.mode))
    return templates.TemplateResponse(request, "positions.html", ctx)


@pages.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request, strategy: str | None = None,
                      symbol: str | None = None, closed: int = 0) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "trades")
    from ct.web import plain
    lifecycles = await q.lifecycles(limit=200, closed_only=bool(closed),
                                    strategy=strategy, symbol=symbol)
    ctx.update(
        lifecycles=lifecycles,
        filter_strategy=strategy, filter_symbol=symbol, closed_only=bool(closed),
        strategies=await q.strategy_stats(),
        lede=plain.trades_lede(len(lifecycles), state.mode))
    return templates.TemplateResponse(request, "trades.html", ctx)


@pages.get("/trades/{correlation_id}", response_class=HTMLResponse)
async def trade_detail(request: Request, correlation_id: str) -> Response:
    """The causal chain for one trade, straight from the event log.

    `causation_id` points at the event that produced this one, so the timeline
    is a chain of *because*, not a list of *and then*."""
    sess = current_session(request)
    q = state.queries()
    events = await q.lifecycle_events(correlation_id)
    if not events:
        raise HTTPException(404, "no events for that correlation id")
    ctx = await _page_context(request, sess, "trades")
    ctx.update(correlation_id=correlation_id, events=events)
    return templates.TemplateResponse(request, "trade_detail.html", ctx)


@pages.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "strategies")
    rejections = await q.rejections(30)
    ctx.update(strategies=await q.strategy_stats(), promotion=await q.promotion(),
               rejections=rejections, rejection_bars=_rejection_bars(rejections),
               signals=await q.recent_signals(40))
    return templates.TemplateResponse(request, "strategies.html", ctx)


@pages.get("/risk", response_class=HTMLResponse)
async def risk_page(request: Request) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "risk")
    limits, defaults, configured = await _limits(q)
    coverage = await q.data_coverage()
    ctx.update(
        limits=limits,
        defaults=defaults,
        configured=configured,
        has_config=bool(configured),
        latest=await q.equity_latest(),
        perf=await q.performance(),
        exposure=await q.symbol_exposure(),
        positions=await q.positions(),
        rejections=await q.rejections(30),
        risk_events=await q.risk_events(unresolved_only=False, limit=50),
        commands=await q.control_commands(20),
        coverage=coverage,
        worst_freshness=_worst_freshness(coverage),
    )
    return templates.TemplateResponse(request, "risk.html", ctx)


@pages.get("/system", response_class=HTMLResponse)
async def system_page(request: Request) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "system")
    ctx.update(host=host_health(state.health_path), db=await q.db_stats(),
               coverage=await q.data_coverage(),
               notifications=await q.notifications(),
               accounts=await q.accounts(),
               uptime_s=time.time() - state.started)
    return templates.TemplateResponse(request, "system.html", ctx)


@pages.get("/activity", response_class=HTMLResponse)
async def activity_page(request: Request, severity: str | None = None,
                        kind: str | None = None) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "activity")
    ctx.update(events=await q.events(limit=150, severity=severity, kind=kind),
               audit=await q.audit(40), commands=await q.control_commands(20),
               filter_severity=severity, filter_kind=kind)
    return templates.TemplateResponse(request, "activity.html", ctx)


# ── HTMX partials ────────────────────────────────────────────────────────


@pages.get("/partials/statusbar", response_class=HTMLResponse)
async def partial_statusbar(request: Request) -> Response:
    sess = current_session(request)
    ctx = await _page_context(request, sess, "")
    return templates.TemplateResponse(request, "partials/statusbar.html", ctx)


@pages.get("/partials/equity", response_class=HTMLResponse)
async def partial_equity(request: Request, hours: int = 168) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "")
    ctx.update(series=await q.equity_series(hours), perf=await q.performance(),
               hours=hours)
    return templates.TemplateResponse(request, "partials/equity.html", ctx)


@pages.get("/partials/positions", response_class=HTMLResponse)
async def partial_positions(request: Request) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "")
    ctx.update(positions=await q.positions(), latest=await q.equity_latest())
    return templates.TemplateResponse(request, "partials/positions.html", ctx)


@pages.get("/partials/signals", response_class=HTMLResponse)
async def partial_signals(request: Request) -> Response:
    sess = current_session(request)
    q = state.queries()
    ctx = await _page_context(request, sess, "")
    ctx.update(signals=await q.recent_signals(12))
    return templates.TemplateResponse(request, "partials/signals.html", ctx)


# ── control ──────────────────────────────────────────────────────────────


CONTROL_PRIVILEGE = {
    "halt": "halt", "resume": "resume", "flatten_all": "flatten",
    "flatten_symbol": "flatten", "pause_strategy": "pause_strategy",
    "resume_strategy": "resume_strategy", "reload_config": "reload_config",
    "resolve_risk_event": "resolve_event",
}


@pages.post("/control/{command}")
async def control(request: Request, command: str) -> JSONResponse:
    """Queue intent. The engine decides, applies and reports back.

    This endpoint never mutates trading state — it inserts a row. The response
    says "queued", never "done", because the difference is real: the engine
    applies it on its next cycle and writes the result back to the same row.
    """
    sess = current_session(request)
    require_csrf(request, sess)
    privilege = CONTROL_PRIVILEGE.get(command)
    if privilege is None:
        raise HTTPException(400, f"unknown command {command!r}")
    if not sess.may(privilege):
        raise HTTPException(403, f"role {sess.role!r} may not {command}")

    body: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        body = await request.json()
    args = {k: v for k, v in body.items() if k in ("symbol", "strategy", "id")}

    from ct.engine.control import ControlPlane

    cmd_id = await ControlPlane(state.db, state.account_id).request(
        command, requested_by=f"web:{sess.user}", args=args)
    await state.db.execute(
        """INSERT INTO audit_log (actor, action, target, after)
           VALUES ($1, $2, $3, $4::jsonb)""",
        f"web:{sess.user}", f"control:{command}", f"account:{state.account_id}",
        json.dumps({"command_id": cmd_id, "args": args, "role": sess.role}))

    note = ""
    if command in ("flatten_all", "flatten_symbol"):
        # The same honesty the CLI prints: queued is not done. The engine
        # submits reduce-only market closes on its next cycle and clears the
        # request only when the book is actually flat.
        note = ("Queued — the engine submits market closes on its next cycle "
                "(~20 s) and reports FLATTEN_COMPLETE in risk events once "
                "the book is actually flat.")
    return JSONResponse({"queued": True, "id": cmd_id, "command": command,
                         "note": note})


# ── JSON API ─────────────────────────────────────────────────────────────


@api.get("/status")
async def api_status() -> dict[str, Any]:
    q = state.queries()
    return {"engine": await q.engine(), "performance": await q.performance(),
            "arming": await q.arming(), "positions": await q.positions()}


@api.get("/positions")
async def api_positions() -> dict[str, Any]:
    q = state.queries()
    return {"books": await q.positions(), "by_symbol": await q.symbol_exposure(),
            "open_orders": await q.open_orders()}


@api.get("/equity")
async def api_equity(hours: int = 168) -> dict[str, Any]:
    return {"series": await state.queries().equity_series(hours)}


@api.get("/trades")
async def api_trades(limit: int = 100) -> dict[str, Any]:
    return {"lifecycles": await state.queries().lifecycles(limit=limit)}


@api.get("/events")
async def api_events(limit: int = 100, after_seq: int | None = None,
                     severity: str | None = None) -> dict[str, Any]:
    return {"events": await state.queries().events(
        limit=limit, after_seq=after_seq, severity=severity)}


@api.get("/risk")
async def api_risk() -> dict[str, Any]:
    q = state.queries()
    return {"events": await q.risk_events(unresolved_only=False),
            "rejections": await q.rejections(30),
            "config": await q.risk_config()}


# ── server-sent events ───────────────────────────────────────────────────


@pages.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    """One-way push: new events, plus a periodic state frame for the header.

    SSE rather than websockets (PHASE2-AUDIT §7.8): the dashboard's data flows
    one way, SSE reconnects itself with no application code, and it carries no
    inbound channel that would need authorising per message.

    Polling the `events` table rather than `LISTEN`/`NOTIFY` because the engine
    does not `NOTIFY` — adding that would put a dashboard concern inside the
    order path's transaction, which is the coupling §1 exists to prevent. At
    ~3k events/day the poll is free.
    """
    q = state.queries()
    last_seq = await state.db.fetchval("SELECT coalesce(max(seq), 0) FROM events")
    # A stream that never ends holds a connection, a task and a database handle
    # for as long as a browser tab is left open — and `is_disconnected()` is not
    # reliable across every transport, so "the client will close it" is not a
    # lifetime policy. Ending deliberately every few minutes lets EventSource
    # reconnect (that is what `retry:` is for) and bounds the leak at one cycle.
    max_seconds = float(os.environ.get("QD_SSE_MAX_SECONDS", "600"))
    deadline = time.monotonic() + max_seconds

    async def gen() -> AsyncIterator[bytes]:
        nonlocal last_seq
        ticks = 0
        # Tell the client how long to wait before reconnecting, once.
        yield b"retry: 3000\n\n"
        while time.monotonic() < deadline:
            if await request.is_disconnected():
                return
            try:
                rows = await q.events(limit=60, after_seq=last_seq)
                for row in reversed(rows):
                    last_seq = max(last_seq, int(row["seq"]))
                    payload = json.dumps({
                        "seq": row["seq"], "ts": str(row["ts"]),
                        "type": row["type"], "severity": row["severity"],
                        "source": row["source"],
                        "correlation_id": row["correlation_id"],
                        "payload": row["payload"]}, default=str)
                    yield f"event: qd\ndata: {payload}\n\n".encode()
                if ticks % 5 == 0:
                    engine = await q.engine()
                    perf = await q.performance()
                    frame = json.dumps({"engine": engine, "perf": perf},
                                       default=str)
                    yield f"event: state\ndata: {frame}\n\n".encode()
                ticks += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A stream that dies silently looks like a quiet system. Say so
                # and keep the connection, so the client's dot goes amber
                # rather than the page going still.
                log.warning("stream poll failed: %s", exc)
                yield (b"event: qd-error\ndata: "
                       + json.dumps({"error": str(exc)}).encode() + b"\n\n")
            await asyncio.sleep(2.0)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
        "Connection": "keep-alive"})


# ── metrics ──────────────────────────────────────────────────────────────


@public.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request) -> str:
    """Prometheus text format, from day one (ARCHITECTURE.md §1, merge 4).

    Unauthenticated **only from loopback**, so a future Prometheus sidecar can
    scrape it without a credential while it stays off the Tailscale surface.
    Nothing here is a secret, but "not a secret" and "public" are different.
    """
    host = request.client.host if request.client else ""
    local = host in ("127.0.0.1", "::1", "localhost")
    if not local and not security.read(
            request.cookies.get(security.SESSION_COOKIE), state.secret):
        raise HTTPException(401, "not signed in")

    lines = [
        "# HELP qd_web_up qd-web process is serving",
        "# TYPE qd_web_up gauge",
        "qd_web_up 1",
        "# HELP qd_web_uptime_seconds seconds since this process started",
        "# TYPE qd_web_uptime_seconds counter",
        f"qd_web_uptime_seconds {time.time() - state.started:.0f}",
    ]
    if state.db is not None:
        with contextlib.suppress(Exception):
            q = state.queries()
            engine = await q.engine()
            perf = await q.performance()
            lines += [
                "# HELP qd_engine_heartbeat_age_seconds age of the newest engine heartbeat",
                "# TYPE qd_engine_heartbeat_age_seconds gauge",
                f"qd_engine_heartbeat_age_seconds {engine['age_s'] or -1:.0f}",
                "# HELP qd_engine_halted 1 when the engine is halted",
                "# TYPE qd_engine_halted gauge",
                f"qd_engine_halted {int(engine['halted'])}",
                "# HELP qd_engine_armed 1 when live arming is valid",
                "# TYPE qd_engine_armed gauge",
                f"qd_engine_armed {int(engine['armed'])}",
                "# HELP qd_open_books number of non-flat position books",
                "# TYPE qd_open_books gauge",
                f"qd_open_books {engine['open_books']}",
            ]
            if perf.get("has_data"):
                lines += [
                    "# HELP qd_equity account equity, quote currency",
                    "# TYPE qd_equity gauge",
                    f"qd_equity {perf['equity']:.8f}",
                    "# HELP qd_max_drawdown worst recorded drawdown, fraction",
                    "# TYPE qd_max_drawdown gauge",
                    f"qd_max_drawdown {perf['max_drawdown']:.8f}",
                ]
    return "\n".join(lines) + "\n"


# ── assembly ─────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    application = FastAPI(title="QuantDesk", version="0.1.0", lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)
    application.mount("/static", StaticFiles(directory=str(STATIC)),
                      name="static")
    application.include_router(public)
    application.include_router(pages)
    application.include_router(api)

    @application.middleware("http")
    async def secure_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = \
            "geolocation=(), microphone=(), camera=(), interest-cohort=()"
        return response

    @application.exception_handler(HTTPException)
    async def on_http_error(request: Request, exc: HTTPException) -> Response:
        wants_html = "text/html" in request.headers.get("accept", "")
        if exc.status_code == 401 and wants_html:
            nxt = request.url.path
            return RedirectResponse(f"/login?next={nxt}", status_code=303)
        if wants_html:
            return templates.TemplateResponse(
                request, "error.html",
                {"request": request, "code": exc.status_code,
                 "detail": exc.detail}, status_code=exc.status_code)
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return application


app = create_app()

__all__ = ["app", "create_app", "state"]
