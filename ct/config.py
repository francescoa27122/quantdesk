"""Typed configuration: environment and Docker secrets via `pydantic-settings`,
strategy parameters via versioned YAML in `config/`.

ARCHITECTURE.md §2 has specified this since the design was written —
"pydantic-settings, env + `config/*.yml`. Secrets from env / Docker secrets;
strategy params from versioned YAML that is hashed into every backtest record
for reproducibility" — and `docker-compose.yml` has mounted `./config:/app/
config:ro` into four services all along. Neither existed. Docker silently
creates a missing bind-mount source, so the mount failed as an empty directory
rather than an error, and the gap stayed invisible.

## Why two mechanisms and not one

They have different lifetimes and different threat models, so merging them would
be a mistake in both directions:

**Secrets and deployment wiring** (`Settings`) come from the environment,
because that is what Docker secrets and the container runtime speak, and because
a value that must never be committed should live somewhere that cannot be
committed by accident.

**Strategy parameters** (`StrategyConfig`) come from YAML in the repository,
because they are *research artifacts*. An entry lookback of 55 is a claim about
the world that some backtest justified; it belongs in version control next to
the code it parameterises, where a diff shows who changed it and when. Putting
it in an env var makes the number unattributable — which is exactly how a live
system ends up running parameters nobody can explain.

## The `_FILE` convention

Any setting can be supplied indirectly as `QD_<NAME>_FILE=/run/secrets/x`, and
the file's contents become the value. This is the convention the codebase
already used by hand for `QD_SESSION_SECRET_FILE`,
`QD_TELEGRAM_BOT_TOKEN_FILE` and `QD_EXCHANGE_API_*_FILE`; it is generalised
here rather than reinvented. The indirection matters: a secret passed as an env
var is visible in `docker inspect`, in `/proc/<pid>/environ`, and in any crash
dump that snapshots the environment. A file is not.

Direct env always wins over the file, so an operator can override without
editing a mounted secret.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# `QD_SYMBOLS=BTC/USD,ETH/USD` is the format every deployment of this system
# already uses. Without NoDecode, pydantic-settings treats a tuple field as
# "complex" and JSON-decodes it inside the env source — *before* any validator
# runs — so a plain comma-separated value raises SettingsError at import time
# and the container never starts. NoDecode hands the raw string to the
# validator below instead. (A `mode="before"` validator alone is not enough;
# the decode happens upstream of it.)
CsvTuple = Annotated[tuple[str, ...], NoDecode]

# Where `config/*.yml` lives. The compose mount puts it at /app/config; a
# developer running from a checkout has it at <repo>/config. Resolve the
# package's parent so both work with no environment variable at all.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("QD_CONFIG_DIR") or (_REPO_ROOT / "config"))


class SecretFileSource(PydanticBaseSettingsSource):
    """Resolve `QD_<FIELD>_FILE=/path` into that file's contents.

    Runs at lower priority than the plain environment, so `QD_MODE` beats
    `QD_MODE_FILE`. A missing or unreadable file yields nothing rather than an
    empty string: an empty secret that looks *set* is worse than one that is
    obviously absent, because it silently defeats the "is this configured?"
    checks downstream (see `known_users()` in ct/web/security.py, which refuses
    every login when no hash is configured — that only works if unset stays
    unset).
    """

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        env = f"QD_{field_name.upper()}_FILE"
        path = os.environ.get(env, "").strip()
        if not path:
            return None, field_name, False
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return None, field_name, False
        return (value or None), field_name, False

    def __call__(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, field in self.settings_cls.model_fields.items():
            value, key, _ = self.get_field_value(field, name)
            if value is not None:
                out[key] = value
        return out


class Settings(BaseSettings):
    """Everything the process reads from its environment, in one typed place.

    Field names map to `QD_`-prefixed variables: `db_dsn` ← `QD_DB_DSN`.
    """

    model_config = SettingsConfigDict(
        env_prefix="QD_",
        case_sensitive=False,
        extra="ignore",          # the containers carry TZ, PATH, PG* and more
    )

    # ── deployment ───────────────────────────────────────────────────────
    env: str = "prod"
    db_dsn: str = ""
    log_level: str = "INFO"
    log_format: Literal["json", "console", "auto"] = "auto"

    # ── trading ──────────────────────────────────────────────────────────
    # `live` additionally requires an unexpired `live_arming` row written by
    # `qd arm-live`. Setting this alone does nothing — two independent
    # switches, on purpose (ARCHITECTURE.md §9).
    # Every default below is the one the existing call site already used —
    # re-derived from source, not assumed. A Settings class that quietly
    # changes a default is worse than no Settings class: it moves the kill
    # switch, or the account, without anybody editing a deployment.
    mode: Literal["paper", "live"] = "paper"
    exchange: str = "kraken"
    account_id: int = 1
    symbols: CsvTuple = ("BTC/USD", "ETH/USD")
    timeframe: str = "4h"
    # Collector backfills several timeframes; the engine trades one.
    timeframes: CsvTuple = ("1h", "4h", "1d")
    strategies: CsvTuple = ("donchian_breakout",)
    initial_equity: float = 10_000.0
    kill_switch_file: str = "/app/state/KILL"

    # ── scheduling (APScheduler, ARCHITECTURE.md Merge 5) ────────────────
    # 20 s, matching TradingEngine's own default — not a round number picked
    # here. `max_degraded_cycles = 5` is documented as "~100 s at the default
    # cadence", which is only true at 20. Changing this silently changes how
    # long the engine tolerates being blind before it halts.
    cycle_seconds: float = 20.0
    # Grace for a job whose fire time passed while the process was down. Longer
    # than one cycle would let a restart replay a stale trading cycle against
    # prices that have since moved.
    job_misfire_grace_seconds: int = 30

    # ── web ──────────────────────────────────────────────────────────────
    session_secret: str = ""
    trusted_proxies: str = ""
    sse_max_seconds: int = 600
    host_health: str = "/hosthealth/metrics.json"
    operator: str = ""

    # ── notifications ────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""

    # ── secrets (supply via QD_*_FILE in production) ──────────────────────
    exchange_api_key: str = ""
    exchange_api_secret: str = ""

    @field_validator("symbols", "strategies", "timeframes", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        """`QD_SYMBOLS=BTC/USD,ETH/USD` → a tuple.

        Without this, pydantic tries to JSON-decode a complex field and a plain
        comma-separated value — the format every deployment of this system
        already uses — fails validation at import time.
        """
        if isinstance(v, str):
            return tuple(s.strip() for s in v.split(",") if s.strip())
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority, highest first: explicit init > env > QD_*_FILE > .env.
        return (init_settings, env_settings,
                SecretFileSource(settings_cls), dotenv_settings)

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def trusted_proxy_set(self) -> frozenset[str]:
        return frozenset(p.strip() for p in self.trusted_proxies.split(",") if p.strip())

    @property
    def is_live(self) -> bool:
        return self.mode == "live"


def sqlalchemy_dsn(dsn: str | None = None) -> str:
    """Translate the runtime DSN into one SQLAlchemy can open.

    The rest of this codebase talks to Postgres through asyncpg with a bare
    `postgresql://` URL. SQLAlchemy needs an explicit driver in the scheme or it
    reaches for psycopg2, which this project does not ship (it uses psycopg 3).

    Lives here, not in `migrations/env.py`, because three callers need it —
    Alembic, `qd migrate`, and the APScheduler jobstore — and env.py cannot be
    imported by any of them: Alembic loads it with a live `context` established,
    so importing it directly raises
    `AttributeError: module 'alembic.context' has no attribute 'config'`.
    `qd migrate --status` did exactly that and failed every time it was run.
    """
    dsn = dsn if dsn is not None else settings().db_dsn
    if not dsn:
        raise ValueError(
            "QD_DB_DSN is not set. Inside the stack it comes from compose; "
            "outside it, export it or set QD_DB_DSN_FILE.")
    for prefix in ("postgresql+asyncpg://", "postgres://", "postgresql://"):
        if dsn.startswith(prefix):
            return "postgresql+psycopg://" + dsn[len(prefix):]
    return dsn


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings. Cached, because re-reading the environment
    mid-run would let two parts of the same cycle disagree about the mode."""
    return Settings()


def reset_settings_cache() -> None:
    """For tests that manipulate the environment. Not used in production."""
    settings.cache_clear()


# ─────────────────────────────────────────────────────────────────────────
# Strategy parameters — versioned YAML
# ─────────────────────────────────────────────────────────────────────────


class StrategyConfigError(Exception):
    """Malformed `config/strategies.yml`. Always fatal.

    Falling back to defaults would be the worst possible behaviour: the engine
    would trade a parameter set nobody chose, and the backtest record would
    attribute the result to a config file that was never actually read.
    """


class StrategyConfig:
    """Parsed `config/strategies.yml`.

    Deliberately does **not** merge in `Strategy.DEFAULTS`. The YAML is the
    record of what was chosen; if a parameter is absent the strategy's own
    default applies at construction time, and `params_for()` returns only what
    the file actually said. That keeps the hash a hash of *decisions*, not of
    whatever the code happened to default to on the day — otherwise editing a
    default would silently invalidate every historical backtest record.
    """

    def __init__(self, raw: dict[str, Any], *, source: Path | None = None) -> None:
        self.source = source
        if not isinstance(raw, dict):
            raise StrategyConfigError("top level of strategies.yml must be a mapping")
        self.version = raw.get("version")
        if not isinstance(self.version, int):
            raise StrategyConfigError(
                "strategies.yml needs an integer `version:` — it is the thing that "
                "makes a backtest record traceable to a specific config revision")
        strategies = raw.get("strategies") or {}
        if not isinstance(strategies, dict):
            raise StrategyConfigError("`strategies:` must be a mapping of name -> config")
        for name, body in strategies.items():
            if not isinstance(body, dict):
                raise StrategyConfigError(f"strategies.{name} must be a mapping")
            params = body.get("params", {})
            if params is not None and not isinstance(params, dict):
                raise StrategyConfigError(f"strategies.{name}.params must be a mapping")
        self.strategies: dict[str, dict[str, Any]] = strategies
        self._raw = raw

    # ── access ───────────────────────────────────────────────────────────

    def names(self) -> list[str]:
        return sorted(self.strategies)

    def enabled_names(self) -> list[str]:
        """Names with `enabled: true`.

        Note this is the *config's* opinion. The database `strategies.enabled`
        column remains authoritative at run time, because an operator pausing a
        strategy from the dashboard must not be undone by the next redeploy —
        the same reasoning as `StrategyRepository.upsert` not touching status.
        """
        return sorted(n for n, b in self.strategies.items() if b.get("enabled") is True)

    def params_for(self, name: str) -> dict[str, Any]:
        return dict((self.strategies.get(name) or {}).get("params") or {})

    def symbols_for(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        v = (self.strategies.get(name) or {}).get("symbols")
        return tuple(v) if isinstance(v, list) and v else tuple(default)

    def timeframe_for(self, name: str, default: str) -> str:
        v = (self.strategies.get(name) or {}).get("timeframe")
        return v if isinstance(v, str) and v else default

    def weight_for(self, name: str, default: float = 0.0) -> float:
        v = (self.strategies.get(name) or {}).get("weight")
        return float(v) if isinstance(v, (int, float)) else default

    # ── provenance ───────────────────────────────────────────────────────

    def config_hash(self) -> str:
        """Hash of the whole config, for "which file produced this run?".

        Canonicalised through JSON with sorted keys, so reformatting the YAML —
        reordering blocks, changing quoting or indentation — does not change the
        hash. Only a change in *content* does.
        """
        blob = json.dumps(self._raw, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def describe(self) -> str:
        where = str(self.source) if self.source else "<defaults>"
        return f"strategies.yml v{self.version} ({len(self.strategies)} strategies) " \
               f"sha256:{self.config_hash()[:12]} from {where}"


_EMPTY = {"version": 0, "strategies": {}}


def load_strategy_config(path: Path | None = None) -> StrategyConfig:
    """Load `config/strategies.yml`.

    A missing file is not an error — research on a laptop, and the test suite,
    both run without one, and every strategy has working defaults. A file that
    exists but is malformed **is** an error: the operator clearly meant to
    configure something, and guessing what would be worse than stopping.
    """
    p = path or (CONFIG_DIR / "strategies.yml")
    if not p.is_file():
        return StrategyConfig(_EMPTY, source=None)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StrategyConfigError(f"{p}: {exc}") from None
    return StrategyConfig(raw if raw is not None else _EMPTY, source=p)


@lru_cache(maxsize=1)
def strategy_config() -> StrategyConfig:
    return load_strategy_config()


def reset_strategy_config_cache() -> None:
    strategy_config.cache_clear()


__all__ = [
    "CONFIG_DIR",
    "Settings",
    "StrategyConfig",
    "StrategyConfigError",
    "load_strategy_config",
    "reset_settings_cache",
    "reset_strategy_config_cache",
    "settings",
    "strategy_config",
]
