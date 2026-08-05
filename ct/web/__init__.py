"""`qd-web` — the dashboard and JSON API.

    app.py       FastAPI application, routes, SSE, /metrics
    queries.py   every read, in one place
    charts.py    server-rendered SVG — no charting library, no CDN
    security.py  sessions, roles, CSRF, login rate limiting
    format.py    presentation filters; a missing number never renders as zero

Read-mostly by construction: the only rows this package writes are
`control_commands` (intent the engine polls) and `audit_log` (who asked).
It holds no exchange credentials, so a compromise here cannot trade.
"""

__all__: list[str] = []
