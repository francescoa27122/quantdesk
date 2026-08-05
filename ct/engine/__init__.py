"""The trading engine — single writer of orders, fills and positions.

    loop.py       TradingEngine: boot -> reconcile -> cycle
    execution.py  OrderExecutor: persist-then-submit, atomic fills
    control.py    kill switch, control commands, RuntimeState

Boots halted and stays halted until reconciliation passes clean. See
`loop.py`'s module docstring for why the cycle order is what it is.
"""

from ct.engine.control import (
    VALID_COMMANDS,
    ControlCommand,
    ControlPlane,
    KillSwitch,
    RuntimeState,
)
from ct.engine.execution import OrderExecutor
from ct.engine.loop import Allocation, TradingEngine

__all__ = [
    "VALID_COMMANDS",
    "Allocation",
    "ControlCommand",
    "ControlPlane",
    "KillSwitch",
    "OrderExecutor",
    "RuntimeState",
    "TradingEngine",
]
