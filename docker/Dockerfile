# Single image, five roles — the command in docker-compose.yml decides which.
# One image means one thing to build, one thing to scan, one thing to keep in
# sync. On a 4-core N100 you do not want to build five.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl is for healthchecks only; build-essential is dropped after the wheel
# build so it never ships in the runtime layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer, cached separately from source so a code edit does not
# reinstall numpy on the NAS.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        "numpy>=2.0" "ccxt>=4.4" "asyncpg>=0.30" "psycopg[binary]>=3.2" \
        "pydantic>=2.9" "pydantic-settings>=2.6" "PyYAML>=6.0" \
        "fastapi>=0.115" "uvicorn[standard]>=0.32" "jinja2>=3.1" \
        "httpx>=0.27" "apscheduler>=3.10" "alembic>=1.14" "sqlalchemy>=2.0" \
        "argon2-cffi>=23.1" "structlog>=24.4"

COPY ct/ ./ct/
COPY db/ ./db/
# Migrations must be in the image or `qd migrate` cannot run inside the
# container — which is the only place it will ever be run against the real
# database. The 0001 baseline also reads db/schema.sql at runtime, so the two
# have to travel together.
COPY alembic.ini ./
COPY migrations/ ./migrations/
# Shipped as a fallback. compose bind-mounts ./config over this read-only, but
# an image that cannot start without a mount is an image that fails confusingly
# when somebody runs it directly — and Docker silently creates a missing bind
# source as an empty directory rather than erroring.
COPY config/ ./config/
RUN pip install --no-cache-dir -e . --no-deps

# Never run the trading engine as root. The container holds the exchange
# credentials; a foothold inside it should not also be a foothold on the host
# volume mounts.
RUN useradd --create-home --uid 10001 qd \
    && mkdir -p /app/logs /app/state /app/reports /app/models \
    && chown -R qd:qd /app
USER qd

# tini reaps zombies and, more importantly here, forwards SIGTERM so the
# engine's graceful shutdown (cancel open orders, persist state) actually runs
# inside the 30s stop_grace_period.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "ct.engine"]


# ─────────────────────────────────────────────────────────────────────────
# Optional ML layer for qd-worker only. Keeping LightGBM out of the engine
# image saves ~250 MB and means an ML dependency cannot break execution.
#
# This stage is LAST, which means a build with no `target` selects it. Every
# service in docker-compose.yml therefore pins one explicitly — `base` for the
# trading containers, `ml` for qd-worker. Omitting it silently shipped the ML
# layer everywhere (measured: 745 MB), defeating the isolation this split
# exists for. There is no `TARGET` build-arg; a previous version of this
# comment claimed one.
# ─────────────────────────────────────────────────────────────────────────
FROM base AS ml
USER root
RUN pip install --no-cache-dir \
        "pandas>=2.2" "scikit-learn>=1.5" "lightgbm>=4.5" \
        "hmmlearn>=0.3" "arch>=7.0"
USER qd
