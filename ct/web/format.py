"""Presentation helpers, registered as Jinja filters.

One rule runs through all of them: **a number the system does not have must not
render as zero.** `None` becomes an em dash, never `0.00` — because "no fills
recorded" and "zero fill rate" look identical once a missing value is coerced,
and this system currently has far more missing values than real ones.

Every figure is rendered with tabular figures in the CSS, so digits line up
column-wise and a change of magnitude is visible without reading the number.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DASH = "—"


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def money(value: Any, digits: int = 2) -> str:
    v = _num(value)
    return DASH if v is None else f"{v:,.{digits}f}"


def compact(value: Any) -> str:
    """1,234,567 → 1.23M. For counts where the magnitude is the message."""
    v = _num(value)
    if v is None:
        return DASH
    a = abs(v)
    if a >= 1_000_000_000:
        return f"{v / 1e9:.2f}B"
    if a >= 1_000_000:
        return f"{v / 1e6:.2f}M"
    if a >= 10_000:
        return f"{v / 1e3:.1f}k"
    return f"{v:,.0f}"


def qty(value: Any) -> str:
    """Crypto quantities need precision that money formatting throws away —
    0.00005 BTC is a legal Kraken order (`ordermin`), and rendering it as 0.00
    would show a real position as empty."""
    v = _num(value)
    if v is None:
        return DASH
    if v == 0:
        return "0"
    a = abs(v)
    digits = 8 if a < 0.001 else 6 if a < 1 else 4 if a < 1000 else 2
    return f"{v:,.{digits}f}".rstrip("0").rstrip(".")


def pct(value: Any, digits: int = 1) -> str:
    v = _num(value)
    return DASH if v is None else f"{v * 100:.{digits}f}%"


def signed_pct(value: Any, digits: int = 2) -> str:
    v = _num(value)
    return DASH if v is None else f"{v * 100:+.{digits}f}%"


def signed(value: Any, digits: int = 2) -> str:
    v = _num(value)
    return DASH if v is None else f"{v:+,.{digits}f}"


def bps(value: Any) -> str:
    v = _num(value)
    return DASH if v is None else f"{v * 10_000:,.0f} bps"


def tone(value: Any) -> str:
    """`up` / `down` / `flat` — the CSS class that colours a number."""
    v = _num(value)
    if v is None or v == 0:
        return "flat"
    return "up" if v > 0 else "down"


def ago(value: Any) -> str:
    """Relative time, coarse on purpose. The exact stamp is in the tooltip."""
    if value is None:
        return "never"
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, datetime):
        when = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        seconds = (datetime.now(timezone.utc) - when).total_seconds()
    else:
        return str(value)
    if seconds < 0:
        return "in " + ago(-seconds)
    if seconds < 10:
        return "just now"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def stamp(value: Any) -> str:
    """Absolute UTC, to the second. Everything in this system is UTC and the
    `Z` is shown so nobody has to wonder which zone a fill was in."""
    if not isinstance(value, datetime):
        return str(value or DASH)
    when = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def severity_class(value: Any) -> str:
    return {"critical": "sev-critical", "warning": "sev-warn",
            "warn": "sev-warn", "info": "sev-info",
            "debug": "sev-debug"}.get(str(value).lower(), "sev-info")


def event_label(event_type: Any) -> str:
    """`execution.order_filled` → `order filled`. The namespace is a filter
    dimension, not something to read forty times down a page."""
    text = str(event_type or "")
    return text.split(".", 1)[-1].replace("_", " ")


def event_group(event_type: Any) -> str:
    return str(event_type or "").split(".", 1)[0]


FILTERS = {
    "money": money,
    "compact": compact,
    "qty": qty,
    "pct": pct,
    "signed_pct": signed_pct,
    "signed": signed,
    "bps": bps,
    "tone": tone,
    "ago": ago,
    "stamp": stamp,
    "severity_class": severity_class,
    "event_label": event_label,
    "event_group": event_group,
}

__all__ = ["DASH", "FILTERS", *sorted(FILTERS)]
