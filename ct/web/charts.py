"""Charts, rendered server-side as SVG.

**Deviation from ARCHITECTURE.md §2, stated rather than smuggled.** The design
said Chart.js. This renders SVG in Python instead, for three reasons that only
became concrete once the page existed:

1. **A control plane must not need the internet to draw its own equity curve.**
   A CDN `<script>` makes the chart — the thing you look at when something is
   wrong — dependent on a network path that has nothing to do with the NAS. The
   alternative, vendoring 200 KB of minified third-party JavaScript into a
   repository whose whole security posture is "small and auditable", is worse.
2. **It lets the Content-Security-Policy be `script-src 'self'` with no
   exceptions.** No `unsafe-inline`, no `unsafe-eval`, no third-party origin.
3. **The charts here are simple.** A line, a shaded drawdown, sparklines and
   segmented meters. That is ~200 lines of path arithmetic, and it renders on
   the *server*, so the browser does no layout maths and the page is complete
   the moment it arrives — including with JavaScript disabled.

The cost, honestly: no pan or zoom, and a redraw is a request. For a
single-operator dashboard on a LAN that is a good trade. If interactive
zooming is ever wanted, that is the moment to reconsider — not before.

Everything here is a pure function of its inputs, so charts are unit-testable
without a browser, which is the other thing a JS charting library takes away.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Sequence
from typing import Any

# Geometry. The palette lives in CSS custom properties so the chart inherits
# the page's theme instead of hardcoding a second one that drifts from it.
PAD_L, PAD_R, PAD_T, PAD_B = 52, 14, 14, 26


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Human-looking axis values: 1, 2, 2.5, 5 × a power of ten."""
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, count)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mag * mult:
            step = mag * mult
            break
    else:
        step = mag * 10
    start = math.ceil(lo / step) * step
    ticks, value = [], start
    while value <= hi + step * 0.001 and len(ticks) < 12:
        ticks.append(round(value, 10))
        value += step
    return ticks or [lo]


def _fmt_money(value: float) -> str:
    a = abs(value)
    if a >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if a >= 10_000:
        return f"{value / 1000:.1f}k"
    if a >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def empty_chart(width: int, height: int, message: str, hint: str = "") -> str:
    """The no-data state, drawn deliberately.

    Most of this system's life so far *is* the empty state, and a blank
    rectangle reads as "broken". A framed panel that says what is missing and
    what would fill it reads as "waiting", which is the truth.
    """
    parts = [
        f'<svg class="chart chart--empty" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_esc(message)}" preserveAspectRatio="none">',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" '
        f'rx="10" class="chart-empty-frame"/>',
        f'<text x="{width / 2}" y="{height / 2 - (6 if hint else 0)}" '
        f'text-anchor="middle" class="chart-empty-text">{_esc(message)}</text>',
    ]
    if hint:
        parts.append(
            f'<text x="{width / 2}" y="{height / 2 + 14}" text-anchor="middle" '
            f'class="chart-empty-hint">{_esc(hint)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def equity_chart(series: Sequence[dict[str, Any]], *, width: int = 960,
                 height: int = 260, show_drawdown: bool = True) -> str:
    """Equity line with a gradient fill and the drawdown shaded beneath it.

    Drawdown is drawn on its own inverted scale along the bottom, not on a
    second y-axis: a dual axis lets you imply any correlation you like by
    choosing the scales, and this chart's job is to make the drawdown's *shape*
    obvious, not to invite a comparison of two units.
    """
    points = [p for p in series if p.get("equity") is not None]
    if len(points) < 2:
        return empty_chart(
            width, height, "No equity marks yet",
            "The engine writes one per cycle once it is running and reconciled")

    values = [float(p["equity"]) for p in points]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:                     # a flat line still deserves a chart
        pad = max(abs(hi) * 0.001, 1.0)
        lo, hi = lo - pad, hi + pad
    span = hi - lo
    lo -= span * 0.08
    hi += span * 0.08
    span = hi - lo

    plot_w = width - PAD_L - PAD_R
    plot_h = height - PAD_T - PAD_B
    n = len(points)

    def x_of(i: int) -> float:
        return PAD_L + (plot_w * i / (n - 1))

    def y_of(v: float) -> float:
        return PAD_T + plot_h * (1 - (v - lo) / span)

    coords = [(x_of(i), y_of(v)) for i, v in enumerate(values)]
    line = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    area = (line + f" L{coords[-1][0]:.2f},{PAD_T + plot_h:.2f}"
                   f" L{coords[0][0]:.2f},{PAD_T + plot_h:.2f} Z")

    rising = values[-1] >= values[0]
    tone = "up" if rising else "down"

    out = [
        f'<svg class="chart chart--equity" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" role="img" '
        f'aria-label="Equity curve, {n} points, '
        f'{_fmt_money(values[0])} to {_fmt_money(values[-1])}">',
        f'<defs><linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" class="eqfill-top eqfill--{tone}"/>'
        f'<stop offset="100%" class="eqfill-bottom"/></linearGradient></defs>',
    ]

    for tick in _nice_ticks(lo, hi):
        y = y_of(tick)
        if not (PAD_T - 1 <= y <= PAD_T + plot_h + 1):
            continue
        out.append(f'<line x1="{PAD_L}" y1="{y:.2f}" x2="{width - PAD_R}" '
                   f'y2="{y:.2f}" class="grid"/>')
        out.append(f'<text x="{PAD_L - 8}" y="{y + 3.5:.2f}" text-anchor="end" '
                   f'class="axis">{_fmt_money(tick)}</text>')

    if show_drawdown and any(p.get("drawdown") for p in points):
        dds = [float(p.get("drawdown") or 0.0) for p in points]
        worst = max(dds) or 1.0
        band = plot_h * 0.28
        dd_path = ("M" + " L".join(
            f"{x_of(i):.2f},{PAD_T + plot_h - band * (d / worst):.2f}"
            for i, d in enumerate(dds))
            + f" L{x_of(n - 1):.2f},{PAD_T + plot_h:.2f}"
              f" L{PAD_L:.2f},{PAD_T + plot_h:.2f} Z")
        out.append(f'<path d="{dd_path}" class="dd-area"/>')

    out.append(f'<path d="{area}" fill="url(#eqfill)" class="eq-area"/>')
    out.append(f'<path d="{line}" class="eq-line eq-line--{tone}" fill="none"/>')
    out.append(f'<circle cx="{coords[-1][0]:.2f}" cy="{coords[-1][1]:.2f}" '
               f'r="3.5" class="eq-dot eq-dot--{tone}"/>')

    # Data island for the crosshair. Attributes, not a <script> block, so the
    # CSP can forbid inline script outright.
    payload = json.dumps([
        {"x": round(x, 1), "y": round(y, 1), "v": round(v, 2),
         "d": round(float(p.get("drawdown") or 0.0), 6),
         "t": str(p.get("ts", ""))}
        for (x, y), v, p in zip(coords, values, points)], separators=(",", ":"))
    out.append(f'<g class="crosshair" hidden>'
               f'<line class="crosshair-line" y1="{PAD_T}" '
               f'y2="{PAD_T + plot_h}"/><circle class="crosshair-dot" r="4"/></g>')
    out.append(f'<rect class="chart-hit" x="{PAD_L}" y="{PAD_T}" '
               f'width="{plot_w}" height="{plot_h}" fill="transparent" '
               f'data-points="{_esc(payload)}"/>')
    out.append("</svg>")
    return "".join(out)


def sparkline(values: Sequence[float], *, width: int = 120, height: int = 28,
              tone: str = "auto") -> str:
    """A shape, not a chart: no axes, no labels, nothing to read precisely."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return (f'<svg class="spark spark--empty" viewBox="0 0 {width} {height}" '
                f'aria-hidden="true"><line x1="0" y1="{height / 2}" x2="{width}" '
                f'y2="{height / 2}" class="spark-flat"/></svg>')
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    step = width / (len(vals) - 1)
    pts = " L".join(
        f"{i * step:.1f},{height - 2 - (height - 4) * (v - lo) / span:.1f}"
        for i, v in enumerate(vals))
    if tone == "auto":
        tone = "up" if vals[-1] >= vals[0] else "down"
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="none" aria-hidden="true">'
            f'<path d="M{pts}" class="spark-line spark-line--{tone}" '
            f'fill="none"/></svg>')


def meter(value: float, limit: float, *, label: str = "", warn: float = 0.7,
          danger: float = 0.9, invert: bool = False) -> str:
    """A segmented budget bar: how much of a limit is spent.

    Segmented rather than a smooth bar because a risk budget is read at a
    glance for *roughly where* it is, and ten discrete blocks are faster to
    judge than a continuous edge. The colour steps at fixed fractions of the
    limit, so two meters with different units still mean the same thing.
    """
    limit = float(limit) or 1.0
    frac = max(0.0, min(1.5, abs(float(value)) / abs(limit)))
    state = ("danger" if frac >= danger else
             "warn" if frac >= warn else "ok")
    if invert:
        state = {"ok": "danger", "danger": "ok", "warn": "warn"}[state]
    filled = round(min(1.0, frac) * 10)
    blocks = "".join(
        f'<span class="meter-seg{" is-on" if i < filled else ""}"></span>'
        for i in range(10))
    over = ' <span class="meter-over">OVER</span>' if frac > 1.0 else ""
    return (f'<span class="meter meter--{state}" role="img" '
            f'aria-label="{_esc(label)} {frac:.0%} of limit">{blocks}{over}</span>')


def hbars(rows: Sequence[tuple[str, float]], *, width: int = 320) -> str:
    """Horizontal bars for counts — rejection reasons, event types.

    Bars, not a pie: humans compare lengths accurately and angles badly, and
    the question here is always "which one dominates", never "what fraction".

    **The bar is an SVG `<rect>`, not a div with `style="width:…"`.** The
    Content-Security-Policy is `style-src 'self'`, which blocks inline style
    *attributes* — so the first version silently dropped every width and drew
    five identical full-length bars claiming five different numbers. A chart
    that is confidently wrong is worse than no chart, and this one was
    misreporting the rejection mix. An SVG width is a presentation attribute,
    not CSS, so nothing strips it.
    """
    if not rows:
        return '<div class="hbars hbars--empty">nothing recorded</div>'
    top = max(float(v) for _, v in rows) or 1.0
    out = ['<div class="hbars">']
    for label, value in rows:
        pct = 100.0 * float(value) / top
        out.append(
            f'<div class="hbar"><span class="hbar-label" title="{_esc(label)}">'
            f'{_esc(label)}</span>'
            f'<svg class="hbar-track" viewBox="0 0 100 7" preserveAspectRatio="none" '
            f'aria-hidden="true">'
            f'<rect x="0" y="0" width="100" height="7" rx="3" class="hbar-bg"/>'
            f'<rect x="0" y="0" width="{pct:.2f}" height="7" rx="3" '
            f'class="hbar-fill"/></svg>'
            f'<span class="hbar-value">{int(value):,}</span></div>')
    out.append("</div>")
    _ = width
    return "".join(out)


def donut(fraction: float | None, *, size: int = 64, label: str = "") -> str:
    """One ratio, drawn once. Used for the maker ratio, which is the single
    number that says whether the maker-first edge is real (§0.1).

    `None` renders as an explicit unknown rather than 0%: "no fills yet" and
    "every fill was taker" are different facts and the difference matters."""
    r = size / 2 - 5
    circ = 2 * math.pi * r
    if fraction is None:
        return (f'<svg class="donut donut--unknown" viewBox="0 0 {size} {size}" '
                f'role="img" aria-label="{_esc(label)}: no data">'
                f'<circle cx="{size/2}" cy="{size/2}" r="{r}" class="donut-track"/>'
                f'<text x="{size/2}" y="{size/2 + 4}" text-anchor="middle" '
                f'class="donut-text">—</text></svg>')
    frac = max(0.0, min(1.0, float(fraction)))
    return (f'<svg class="donut" viewBox="0 0 {size} {size}" role="img" '
            f'aria-label="{_esc(label)}: {frac:.0%}">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" class="donut-track"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" class="donut-value" '
            f'stroke-dasharray="{circ * frac:.2f} {circ:.2f}" '
            f'transform="rotate(-90 {size/2} {size/2})"/>'
            f'<text x="{size/2}" y="{size/2 + 4}" text-anchor="middle" '
            f'class="donut-text">{frac * 100:.0f}%</text></svg>')


__all__ = [
    "donut",
    "empty_chart",
    "equity_chart",
    "hbars",
    "meter",
    "sparkline",
]
