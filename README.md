# QuantDesk

**A complete algorithmic crypto trading stack that runs on a NAS in 2 GB of RAM.**
Backtester, walk-forward evaluator, risk engine, live trading loop, web dashboard
and a two-key arming ceremony — five containers, one `docker compose up`.

Built for a UGreen DXP2800 (Intel N100, 4 cores), which is the interesting
constraint: everything here is shaped by "must run all day on a box that also
serves files". It will run on anything that runs Docker.

```bash
git clone https://github.com/francescoa27122/quantdesk && cd quantdesk
cp .env.example .env                      # set POSTGRES_PASSWORD, leave QD_MODE=paper
docker compose up -d qd-db                # schema applies itself on first boot
docker compose up -d qd-collector qd-engine qd-web
```

Dashboard on `:8090`. That is a paper-trading stack collecting live Kraken data.
It cannot place a real order — not because you haven't configured one, but
because no live broker exists in this build and the engine refuses to start
pretending otherwise. Full walkthrough in [Deploy](#deploy).

**Want to try it with no database and no Docker at all?** Point a backtest at a
Kraken CSV:

```bash
pip install -e . && qd backtest --strategy ema_cross --symbol BTC/USD --csv ./XBTUSD_240.csv
```

## What makes it worth reading

- **A backtester that refuses to lie to you.** Two tests are load-bearing:
  appending future data must not change history, and a strategy's view
  physically ends at the current bar. If either fails, every performance number
  in the project is void — and the README says so rather than burying it.
- **Fee-aware from the ground up.** Kraken spot is 0.80% round-trip at the $0
  tier. A signal whose expected edge is under the fee is rejected with
  `EDGE_BELOW_FEES` before it ever becomes an order. Most "profitable" retail
  backtests die here.
- **Deflated Sharpe and purged walk-forward**, so the honest trial count is part
  of the result. `qd backtest` tells you what happened; `qd walkforward` is what
  a decision is allowed to use.
- **27 rejection codes and 3-way position sizing**, shared by the backtester and the
  live engine through one `AccountStateBuilder` — the same code prices your
  history and your next order, so they cannot silently diverge.
- **Two independent switches for live trading.** `QD_MODE=live` *and* an
  unexpired `live_arming` row written only by a confirmation phrase typed at a
  terminal. No env var, flag, stdin path or dashboard button can arm it.
- **Backups that carry a receipt.** Nightly `pg_dump` records size, sha256 and
  row counts; a Sunday drill restores the newest dump into a scratch cluster and
  compares. An unverified backup is a rumour.
- **Event-sourced.** Durable event log with replay, so "what happened at 03:14?"
  is a query rather than an archaeology project.
- **Strict everywhere.** `ruff`, `mypy --strict` across all 61 modules with zero
  exemptions, and CI runs the suite on every Python version the project claims
  to support — derived from the Dockerfile, not hardcoded.

## The dashboard

Seven pages served by `qd-web`, read-only over Tailscale. Server-rendered
Jinja with a little SSE — no build step, no JavaScript framework, and it loads
on a phone over a VPN.

> **The figures below are synthetic demo data**, generated to exercise the UI.
> They are not trading results, and no strategy in this repo has earned real
> money. That distinction is the whole point of the project, so it would be a
> poor place to start blurring it.

**Overview** — equity and drawdown, the risk budget against its caps, open
books, and the signals risk *refused*. The refusals are given equal billing on
purpose: without them you cannot tell a bad strategy from one that risk is
strangling.

![The Overview page](docs/screenshots/overview.png)

**Risk** — the whole gate stack, in the order it executes, with each gate's
limit next to its current value. Two rows read `not measured here` rather than
green, because connectivity and clock skew are checked inside the engine and
leave no row behind — so the page will not claim they are healthy.

![The Risk page](docs/screenshots/risk.png)

**Strategies** — signals, what risk let through, what actually filled, and the
promotion gate. It reports `Never evaluated` rather than a comforting green
tick, because the nightly evaluator is Phase 5 and "not measured" is not the
same as "passed".

![The Strategies page](docs/screenshots/strategies.png)

**Trades** — one row per lifecycle, traced from the signal that started it to
the P&L that ended it, including the ones that never became an order. Built on
the event log rather than a `trades` table, because nothing writes that table,
and a page over it would look like "no trades yet" instead of "nothing records
this".

![The Trades page](docs/screenshots/trades.png)

Also: [Positions](docs/screenshots/positions.png) (per-book exposure, stops and
targets) and [System](docs/screenshots/system.png) (data coverage, schema
provenance, database size, heartbeats).

**Design decisions and rationale: [ARCHITECTURE.md](ARCHITECTURE.md). Read it
before changing anything.** This file is the deployment and maintenance guide.
Practical gotchas are collected in [Tips and tricks](#tips-and-tricks).

---

## Read this before anything else

1. **Fees, not code, are the binding constraint.** Kraken spot at the $0 tier is
   0.40% taker → **0.80% round-trip**. A strategy taking 200 trades a year needs
   ~160%/yr gross just to break even. Everything here is designed to trade less.
2. **Nothing in this repo has earned real money yet.** All seven strategies are
   marked 🟡 *implemented but unproven*. The promotion gate (ARCHITECTURE.md §8.3)
   exists to stop you skipping that step, including on a day when you feel sure.
3. **Live trading needs two independent switches:** `QD_MODE=live` **and** a valid
   unexpired row in `live_arming`, written only by `qd arm-live` after you type a
   confirmation phrase. Setting the env var alone does nothing.
4. **Most retail systematic crypto strategies do not beat buy-and-hold net of
   fees.** Every report in this system compares against buy-and-hold BTC net of
   the same fees and will say so plainly when yours doesn't.

---

## Current state

| Component | Status |
|---|---|
| Domain models, position accounting | ✅ built, tested |
| Risk engine (16 gates, 3-way sizing) | ✅ built, tested |
| `AccountStateBuilder` (shared by backtest + live) | ✅ built, tested |
| Event bus + durable log + replay | ✅ built, tested, wired into the portfolio engine |
| Portfolio backtester (N symbols × M strategies) | ✅ built, tested |
| Single-symbol backtest engine + cost model | ✅ built, tested |
| Metrics incl. PSR / deflated Sharpe | ✅ built, tested |
| Purged walk-forward + Monte Carlo | ✅ built, tested |
| 7 strategies | 🟡 built, unproven |
| Strategy lifecycle FSM + allocator (`ct/strategies/lifecycle.py`, `ct/risk/allocator.py`) | ✅ built, tested |
| Paper broker | 🟡 built, needs live quotes to exercise |
| Persistence (`ct/store/`) — repos, outbox, reconciliation, PG event sink | ✅ built, tested against real TimescaleDB |
| Docker stack, DB schema | ✅ deployed — 5 containers healthy on the NAS since 2026-07-29 |
| Market data service (`ct/data/`) — feed, integrity, backfill, history import | ✅ built, tested, smoke-tested live |
| Trading engine (`ct/engine/`) — loop, control plane, execution | ✅ built, tested, run end-to-end |
| CLI (`ct/cli.py`) — every `qd …` command below | ✅ built, tested |
| Dashboard (`ct/web/`) — 8 pages + trade detail, SSE, JSON API, `/metrics` | ✅ built, tested, running against live paper data |
| Live broker + arming ceremony | 🟡 `qd arm-live` writes the arming; **no live broker exists**, and `qd-engine` refuses to start in `QD_MODE=live` |
| ML (regime HMM, meta-labeller) | ⛔ Phase 5 |

**682 tests passing** in ~48 s (`pytest -q`), plus 40 that skip without a real
Postgres — see [Database tests](#database-tests). `ruff` clean and `mypy ct`
clean with **no per-module exemptions**: the typing ratchet that once exempted
31 modules was paid off to zero on 2026-08-04. CI runs the suite on every minor
Python version between the `requires-python` floor and the version the
containers ship, both read from source rather than written into the workflow.

### The `qd` commands

`qd` is `ct.cli:main`, installed by the image. Exit codes are load-bearing:
**0** ok · **1** failure · **2** usage · **3** refused by a gate.

| Command | Does |
|---|---|
| `qd backtest` | one strategy on one symbol, or N×M as a portfolio |
| `qd walkforward` | purged walk-forward, optional Monte Carlo — **the honest evaluation** |
| `qd promotion-status` | the paper→live gate, per strategy |
| `qd init-db` | apply `db/schema.sql` to an empty database |
| `qd migrate` | apply pending Alembic migrations · `--status` · `--sql` to review first |
| `qd coverage` | what candle data actually exists, with gap counts |
| `qd backfill` | REST catch-up inside the venue's retained window |
| `qd import-history` | Kraken bulk OHLCVT CSVs (zip or directory) |
| `qd status` | equity, open books, arming, unresolved risk events |
| `qd halt` / `qd resume` | queue a control command the engine applies next cycle |
| `qd flatten` | reduce-only market closes next cycle; clears on FLATTEN_COMPLETE |
| `qd strategies` | lifecycle state of every registered strategy |
| `qd enable-strategy` | **the paper gate** — `proposed`→`paper`; interactive terminal only |
| `qd pause-strategy` / `qd resume-strategy` | one strategy on or off |
| `qd kill --engage` / `--release` | the file kill switch; needs no database |
| `qd arm-live` | **the human gate** — interactive terminal only |
| `qd disarm` / `qd arming-status` | revoke, and see whether renewal will continue |
| `qd hash-password` | argon2id hash for `QD_ADMIN_PASSWORD_HASH` (or `…_FILE`) |

Research commands read candles from Postgres, or from a Kraken CSV/zip/directory
with `--csv`, which runs with no database at all:

```bash
qd backtest --strategy ema_cross --symbol BTC/USD --csv ./XBTUSD_240.csv
```

One command the maintenance table used to name is **not built**: `qd refit
--all` (the quarterly re-fit is Phase 5). Use `qd walkforward` instead.
Backups are not a `qd` command at all — the `qd-backup` sidecar runs them
(see Maintenance).

---

## Configuration

Two mechanisms, deliberately, because they have different lifetimes and
different threat models.

**Deployment and secrets — environment.** Typed in `ct/config.py`
(`pydantic-settings`). Every field maps to a `QD_`-prefixed variable:
`db_dsn` ← `QD_DB_DSN`. Any of them can be supplied indirectly as
`QD_<NAME>_FILE=/run/secrets/x`, which is the Docker-secrets path and the one
to use in production — a secret passed as a plain env var is visible in
`docker inspect`, in `/proc/<pid>/environ`, and in any crash dump that
snapshots the environment.

**Strategy parameters — `config/strategies.yml`, in version control.** These
are research artifacts, not deployment settings. An entry lookback of 55 is a
claim about the world that some backtest justified, and it belongs where
`git log -p config/` shows who changed it and when. A parameter in an
environment variable is unattributable, which is how a live system ends up
running numbers nobody can explain.

The file carries a `version:` you bump on every change, and its hash is
recorded on every backtest row (`backtests.config_hash`), so a result six weeks
old can be traced back to the exact configuration that produced it. Anything
you omit falls back to the strategy's own defaults and is deliberately *not*
folded into the hash — otherwise editing a default in Python would silently
invalidate every historical record.

`enabled:` in that file is the config's opinion. The database
`strategies.enabled` column stays authoritative at run time, so pausing a
strategy from the dashboard survives the next redeploy. Use
`qd enable-strategy <name>` to actually turn one on.

## Schema changes

`db/schema.sql` is the Alembic baseline and is applied by Postgres on first
boot via `/docker-entrypoint-initdb.d`. **After that, every change is a
migration** — do not edit `schema.sql`, or a fresh install and an upgraded one
will disagree the moment somebody forgets one of them.

```bash
alembic revision -m "what you are changing"   # write the SQL by hand
qd migrate --sql                              # review it before it runs
qd migrate                                    # apply
qd migrate --status                           # current revision vs head
```

There is no `--autogenerate`: the schema is hand-written SQL with Timescale
hypertables, and autogenerate proposes dropping the chunk tables Timescale
manages. Write a real `downgrade()` too — "schema changes on a live trading DB
need to be reversible" is the reason Alembic is here at all.

Safe to run against an existing deployment: the baseline detects a schema that
initdb already built and adopts it rather than trying to recreate it.

## Scheduling

The engine runs an in-process `AsyncIOScheduler` (ARCHITECTURE.md Merge 5), not
a `while` loop and not a scheduler container — a separate container triggering
work in this one would need an RPC path, and that is network surface on the
order path.

Two jobs: the trading cycle on an interval (`QD_CYCLE_SECONDS`, default 20) and
promotion evaluation daily at 00:05 UTC. Both run with `max_instances=1` and
`coalesce=True`. That is not tuning — `cycle()` mutates the account and places
orders, so two at once is double-sizing against a stale snapshot, and a backlog
after a stall must run *once*, not fifteen times.

Jobs persist to Postgres when `QD_DB_DSN` is set, so the daily job's next-run
time survives a restart. If the jobstore cannot be opened the engine still
starts and trades, logging loudly that the schedule will not survive a restart.

---

## Prerequisites

- **16 GB RAM in the NAS.** With the Minecraft server running you have ~2.0 GB
  free; this stack wants ~2.2 GB always-on. One SODIMM, ~$40. If you skip it,
  keep `qd-worker` profile-gated (it already is) and expect to stop Minecraft
  while backtesting.
- Docker + docker compose on the NAS (already present).
- A Kraken account with an API key created as follows — **do this before you
  paste a key anywhere:**
  - Permissions: **Query Funds**, **Query Open/Closed Orders**, **Create & Modify Orders**
  - **Withdraw Funds: OFF.** Not "off for now" — off.
  - **IP allowlist: your NAS's egress IP.**

  With withdrawal disabled, total compromise of the NAS costs you slippage, not
  the balance. It is the single most effective control in this project.

---

## Deploy

Deploy target: `/volume1/docker/quantdesk` on the NAS.

1. **Copy the repo up.** `scp -O` is required — plain `scp` fails against UGOS.

   ```bash
   scp -O -i ~/.ssh/<your-key> -r ./quantdesk <admin-user>@<nas-host>:/volume1/docker/
   ```

2. **Create the secret files** (on the NAS, never in git):

   ```bash
   ssh -i ~/.ssh/<your-key> <admin-user>@<nas-host> 'mkdir -p /volume1/docker/quantdesk/secrets && chmod 700 /volume1/docker/quantdesk/secrets'
   ```

   Then write these four files, each containing only the value, no newline
   fuss, `chmod 600`:
   - `secrets/exchange_api_key`
   - `secrets/exchange_api_secret`
   - `secrets/telegram_bot_token`
   - `secrets/session_secret` (`openssl rand -base64 48`)

3. **Configure.**

   ```bash
   cp .env.example .env
   ```

   Set `POSTGRES_PASSWORD` (`openssl rand -base64 32`), `QD_TELEGRAM_CHAT_ID`,
   and leave `QD_MODE=paper`.

4. **Set the admin password hash.** On a deployed NAS the easy path is the
   wrapper, which does all of the below in one go (prompt → `.env` → recreate
   `qd-web`):

   ```bash
   ./reset-password.sh
   ```

   By hand instead:

   ```bash
   docker compose run --rm qd-web qd hash-password
   ```

   Paste the **`QD_ADMIN_PASSWORD_HASH=` line it prints** into `.env` verbatim.
   That line has every `$` doubled, and it must: an argon2id hash is full of
   `$`, `docker compose` interpolates `$` in `.env`, and **quoting does not
   stop it**. Measured against the real binary — the raw hash and the quoted
   hash both arrive inside the container as `=19=65536,t=3,p=4`, so every
   login fails in a way that looks like a wrong password rather than a mangled
   file.

   **Or sidestep the escaping entirely** — put the raw hash in a secret file,
   which is how every other secret here is delivered and which no interpolation
   can touch:

   ```bash
   docker compose run --rm qd-web qd hash-password --json | jq -r .hash      > secrets/admin_password_hash
   chmod 600 secrets/admin_password_hash
   ```

   `qd-web` reads `QD_ADMIN_PASSWORD_HASH_FILE` (already wired to
   `/run/secrets/admin_password_hash` in the compose file) whenever the
   environment variable is absent or mangled. Either route works; the file is
   one fewer thing to get wrong.

   Since 2026-08-04 a hash that does not start with `$argon2` is refused at
   startup with a log line naming the variable and the fix, instead of being
   passed to the verifier where it is indistinguishable from a wrong password.
   That was the whole difficulty with this failure: it looked like the operator,
   not the config.

5. **Start the database first and let it initialise the schema.**

   ```bash
   docker compose up -d qd-db
   ```

   Wait for `healthy`, then confirm the hypertables exist:

   ```bash
   docker compose exec qd-db psql -U qd -d quantdesk -c "\dt"
   ```

6. **Start the rest.**

   ```bash
   docker compose up -d qd-collector qd-engine qd-web
   ```

7. **Reach the dashboard over Tailscale** at `http://100.x.y.z:8090`.
   Do not port-forward it. Nothing about a trading control plane belongs on
   the public internet.

   Sign in with `admin` and the password you hashed in step 4. Roles come from
   env vars — `QD_ADMIN_PASSWORD_HASH`, and optionally
   `QD_OPERATOR_PASSWORD_HASH` / `QD_VIEWER_PASSWORD_HASH`. **Halt works for
   every role**, including `viewer`: stopping is never gated.

   The dashboard cannot arm live trading and has no route that could — the
   confirmation phrase is typed at a terminal or not at all.

8. **Backfill history** — ⚠ **not possible from Kraken's API.** Measured
   2026-07-28: Kraken retains only ~720 intervals per timeframe, so the API
   offers 30 days at 1h, ~4 months at 4h, ~2 years at 1d. Deep history comes
   from Kraken's bulk OHLCVT CSVs instead — a **manual Google Drive download**,
   see ARCHITECTURE.md §7B — imported with `qd import-history`. Until that lands
   the collector accumulates forward from first run, which is the supported
   fallback and needs no action.

---

## Daily operation

**Check status**

```bash
docker compose ps && docker compose logs --tail 50 qd-engine
```

**Stop trading immediately** — three independent paths that share no mechanism,
so one being broken cannot take the others with it. Use whichever you can reach
fastest. They are *not* equivalent, and the difference matters under stress:

```bash
docker compose exec qd-engine touch /app/state/KILL
```

- The dashboard's red **HALT** button (available to every role, including `viewer`).
- Telegram `/halt` to your existing bot.

| Path | Effect |
|---|---|
| `touch /app/state/KILL`, `qd kill` | **KILL** — blocks *all* orders, exits included. Immediate: the risk gate checks the file before every order, and an unreadable file counts as engaged. |
| Dashboard **HALT**, `qd halt`, Telegram `/halt` | **HALT** — stops *opening*; exits still work. Applied on the engine's next cycle. |
| `qd flatten` | **FLATTEN** — reduce-only market closes next cycle, until the book is flat. |

KILL means "I do not trust this system right now", which is why it stops exits
too. To get *out* of positions, use flatten — a third thing again, on purpose.
Stopping is never gated: every role may halt, and no drawdown state or
disarming can prevent one from taking effect.

**Resume**

```bash
docker compose exec qd-engine rm -f /app/state/KILL
```

**Run a backtest**

```bash
docker compose --profile batch run --rm qd-worker qd backtest --strategy ema_cross --symbol BTC/USD --timeframe 4h
```

**Run the honest evaluation** (this, not the above, is what a decision uses):

```bash
docker compose --profile batch run --rm qd-worker qd walkforward --strategy ema_cross --symbol BTC/USD --folds 5 --montecarlo
```

**Check the promotion gate**

```bash
docker compose --profile batch run --rm qd-worker qd promotion-status
```

---

## Tips and tricks

Things that cost someone an evening. In rough order of how likely you are to
hit them.

**Put the admin password hash in a secret file, not `.env`.** An argon2id hash
is full of `$`, `docker compose` interpolates `$` in `.env`, and **quoting does
not stop it**. The mangled hash arrives as `=19=65536,t=3,p=4` and every login
fails looking exactly like a wrong password. Sidestep it entirely:

```bash
docker compose run --rm qd-web qd hash-password --json | jq -r .hash > secrets/admin_password_hash
```

**Use `--csv` to evaluate a strategy with no infrastructure at all.** No
database, no containers, no collector. This is the fastest way to decide
whether an idea is worth deploying:

```bash
qd backtest --strategy ema_cross --symbol BTC/USD --csv ./XBTUSD_240.csv
```

**`qd backtest` is a description; `qd walkforward` is a decision.** A single
backtest over one period will flatter almost anything. The walk-forward is
purged, reports a deflated Sharpe, and takes the honest trial count — every
variant you tried, not the one you kept.

**Script against exit codes, not stdout.** `0` ok · `1` failure · `2` usage ·
`3` refused by a gate. A `3` is the system working, and a wrapper that treats it
as a crash will page you at 4 a.m. for correct behaviour.

**A quiet strategy is usually right, not broken.** Ask the database before you
debug the code:

```sql
SELECT reject_code, count(*), max(ts)
FROM signals WHERE strategy = 'ema_cross' AND NOT accepted
  AND ts > now() - interval '7 days'
GROUP BY 1 ORDER BY 2 DESC;
```

`EDGE_BELOW_FEES` at the top of that list means the fee gate is doing its job.

**Kill switch ≠ flatten, deliberately.** `KILL` blocks *all* orders including
exits, because it means "I do not trust this system right now". If you want
positions closed, that is `qd flatten`. Reaching for the wrong one under stress
is the mistake this naming exists to prevent — and **halt works for every role,
including `viewer`.** Stopping is never gated.

**Deep history does not come from the API.** Measured 2026-07-28: Kraken
retains ~720 intervals per timeframe, so the REST API gives you ~30 days at 1h
and ~2 years at 1d, no matter how you page it. Anything deeper is a manual bulk
OHLCVT download imported with `qd import-history`. If you skip this the
collector simply accumulates forward from first run, which is a supported state.

**Deliver every secret as `_FILE`.** Every `QD_*` setting accepts
`QD_<NAME>_FILE=/run/secrets/x`. A secret passed as a plain env var is visible
in `docker inspect`, in `/proc/<pid>/environ`, and in any crash dump that
snapshots the environment.

**Create the exchange key withdrawal-disabled and IP-allowlisted.** One setting,
and it caps total compromise of the box at slippage rather than your balance.
It is the highest-leverage control in the project.

**Keep the dashboard on Tailscale.** Do not port-forward it. Nothing about a
trading control plane belongs on the public internet.

**Backtests are profile-gated on purpose.** `qd-worker` sits behind
`--profile batch` so a heavy walk-forward never competes with the live engine
for the N100's four cores. On 8 GB, stop Minecraft before a big sweep.

**`scp -O` or nothing on UGOS.** Plain `scp` fails against the NAS's SFTP
implementation. Same trap bites `rsync` over the default transport.

**Recreate the test database container after touching `schema.sql`.** The suite
isolates with TRUNCATE, not DROP SCHEMA — dropping deadlocks against
TimescaleDB's background workers — so a changed schema does not take effect in
a reused container.

**Editing `schema.sql` after first boot is a trap.** It is the Alembic baseline
and is applied once by Postgres via `/docker-entrypoint-initdb.d`. Every change
after that is a migration (`qd migrate`), or a fresh install and an upgraded one
will disagree the moment somebody forgets one of them.

---

## Going live (Phase 6 — the arming half exists, the broker does not)

**Nothing can trade live today**, and the refusal is loud rather than silent:
`qd-engine` **exits with code 2 if started with `QD_MODE=live`**, because no
`CcxtBroker` exists in this build and paper-trading while you believe otherwise
is strictly worse than not starting. What *is* built is the arming ceremony —
`qd arm-live`, renewal, disarm, and the human-rooted invariant.

**You arm once. After that the system keeps itself armed and tells you when
something changed.** Full design in ARCHITECTURE.md §8.4.

1. `qd promotion-status --strategy X` must report `passed: true`. It currently
   reports that nothing has ever been evaluated, because the nightly evaluator
   is Phase 5 — so arming today requires `--force`, which is permanently
   audited and notified.
2. Set `QD_MODE=live`, recreate `qd-engine` — **only once a live broker exists.**
3. `qd arm-live --strategy X --symbol BTC/USD --max-notional 250`
   — prints the whole envelope, then requires you to type
   `ARM LIVE <account> <max-notional> USD` exactly. **It refuses without an
   interactive terminal**: there is no flag, environment variable or stdin path
   that supplies the phrase, so no script or scheduler can arm anything. Run it
   yourself:

   ```bash
   docker compose run --rm qd-engine qd arm-live --strategy X --symbol BTC/USD --max-notional 250
   ```

4. Verify with `qd arming-status`, which also recomputes the envelope hash and
   tells you whether renewal will continue or a fresh tap is coming.

From then on the engine renews the arming before it expires, indefinitely, for
as long as nothing meaningful changes. **You are not expected to keep track of
it.** What breaks the renewal and asks you again:

- You change a cap, a symbol, a strategy, risk config, or any line of strategy
  code (all folded into `envelope_hash`).
- Total drawdown breaches its limit, or the promotion gate regresses.

Transient faults — exchange unreachable, stale market data, a reconciliation
mismatch — disarm too, but the engine may re-arm itself once it has verified the
cause is resolved and the envelope is unchanged. It notifies either way.

**`qd disarm` at any time**, and the red **HALT** button always works. Stopping
is never gated, and disarming any link kills every renewal descended from it.

> **What Claude will not do:** originate the first arm, or re-arm after a
> drawdown / gate-regression / config-change disarm. Those are yours. Everything
> else about arming — renewing, disarming, halting, flattening, re-arming after
> a verified transient fault — is automatic.

---

## Maintenance

| Cadence | Task | Command |
|---|---|---|
| Daily | Automatic `pg_dump -Fc` to `/volume1/docker/quantdesk/backups` | ✅ `qd-backup` sidecar, 03:30 CT; on-demand: `docker exec qd-backup sh /backup-loop.sh backup` |
| Daily | Read the morning summary notification | 🟡 the sender is real (`ct/notify.py`); silent unless Telegram is configured — see below |
| Weekly | **Verify a backup restores** into a scratch cluster | ✅ `qd-backup` drills Sundays after the dump; on-demand: `docker exec qd-backup sh /backup-loop.sh drill` |
| Weekly | Review `risk_events` for unresolved rows | `qd status` |
| Monthly | Re-check the exchange fee tier; update `.env` if it changed | — |
| Quarterly | Walk-forward re-fit; parameters may only change at a re-fit boundary | `qd walkforward --strategy X` |
| On any code change to a live strategy | The promotion clock **resets** | enforced via `code_git_sha` |

**An unverified backup is a rumour**, so every backup carries its receipt:
`backups/latest.json` records size, sha256 and row counts at dump time, the
Sunday drill restores the newest dump into a scratch cluster *inside*
`qd-backup` (never into the RAM-capped `qd-db`) and compares structure and
counts against that receipt, and `backups/drill-latest.json` records the
verdict. A failed **or stale** backup turns `qd-backup` unhealthy, which
Mission Control surfaces. Retention: 14 daily dumps, 8 Sunday (`-weekly`)
dumps. Dumps never leave the NAS yet — off-box replication is open debt.

To restore for real: stop the stack, recreate `qd-db` with an empty
`./data/pgdata`, then `CREATE EXTENSION timescaledb; SELECT
timescaledb_pre_restore();`, `pg_restore -d quantdesk <dump>`, `SELECT
timescaledb_post_restore();` — the same sequence the drill exercises weekly.

### Notifications: real in code, silent until configured

`ct/notify.py` is the outbox's real sender — the earlier stub that logged and
claimed success is gone. It needs two settings:

- `QD_TELEGRAM_BOT_TOKEN_FILE` — path to a file holding the bot token
- `QD_TELEGRAM_CHAT_ID` — the destination chat

With either missing, `TelegramSender` logs the message and reports success on
purpose, so the outbox drains rather than growing a retry backlog it can never
clear. That degradation is deliberate (see the module docstring), and the cost
is that *every* loud notification — halts, quarantines, drawdown breaches,
arming changes — ends up only in `docker compose logs qd-engine`.

`qd status` reports which of the two states you are in, under
`telegram_configured`; it asks `TelegramSender` rather than assuming, so it
cannot drift from the sender it describes. After configuring, confirm with:

```bash
docker logs qd-engine | grep "telegram notifications are LIVE"
```

That line is logged lazily, on the first notification after a restart — its
absence right after recreating the container means "nothing sent yet", not
"broken".

**Stopping the engine from your phone does not depend on any of this.** That
path is inbound and works regardless — see *Stop trading immediately* above.

### Log locations

- Container logs: `docker compose logs <service>` (capped 10 MB × 3).
- Structured app logs: `/volume1/docker/quantdesk/logs/`.
- The authoritative record of *what happened and why* is not a log file — it is
  the `signals`, `risk_events`, and `audit_log` tables. Query those first.

### When a strategy goes quiet

That is usually correct behaviour, not a bug. Ask the database:

```sql
SELECT reject_code, count(*), max(ts)
FROM signals WHERE strategy = 'ema_cross' AND NOT accepted
  AND ts > now() - interval '7 days'
GROUP BY 1 ORDER BY 2 DESC;
```

`EDGE_BELOW_FEES` in that list is the system working as designed.

---

## Development

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev,ml]"
```

```bash
python -m pytest tests/ -q
```

### Database tests

`tests/test_store.py` needs a real Postgres and **skips without one**. There is
no in-memory substitute: what it tests is `ON CONFLICT` semantics,
`FOR UPDATE SKIP LOCKED`, CHECK constraints, a recursive CTE and a plpgsql
function. A fake database would only test the fake.

Set `QD_TEST_DSN` to run them:

```bash
QD_TEST_DSN=postgresql://qd:qdtest@127.0.0.1:5432/quantdesk python -m pytest tests/test_store.py -q
```

The NAS blocks both inbound Postgres ports and SSH forwarding, so on that host
the tests run in a container beside the database — which also matches how the
stack actually runs:

```bash
docker run -d --name qdtest -p 55433:5432 -e POSTGRES_USER=qd -e POSTGRES_PASSWORD=qdtest -e POSTGRES_DB=quantdesk timescale/timescaledb:2.17.2-pg16
```

The interpreter is read from `docker/Dockerfile` rather than written out here,
so "matches how the stack actually runs" stays true after a base-image bump —
this line used to say `python:3.12-slim` while the containers had moved to 3.14.

```bash
PY="$(grep -oP '^FROM python:\K[0-9]+\.[0-9]+' /tmp/qdsrc/docker/Dockerfile)" && docker run --rm --network host -v /tmp/qdsrc:/src -w /src -e QD_TEST_DSN=postgresql://qd:qdtest@127.0.0.1:55433/quantdesk "python:$PY-slim" sh -c "pip install -q asyncpg pytest pytest-asyncio numpy && python -m pytest tests/test_store.py -q -p no:cacheprovider"
```

**Recreate the container to pick up a changed `schema.sql`.** The tests isolate
with TRUNCATE rather than DROP SCHEMA, because dropping deadlocks against
TimescaleDB's background workers. `schema_version` records the hash the database
was built from and `bootstrap()` warns on drift.

Two tests are load-bearing. If either fails, **stop trusting every performance
number in the project until it passes again**:

- `test_indicators_are_causal` — appending future data must not change history.
- `test_no_lookahead_in_the_engine` — a strategy's view physically ends at the
  current bar.

### Adding a strategy

1. Subclass `Strategy` in `ct/strategies/`, set `name`, `family`, `DEFAULTS`,
   `warmup`, implement `on_bar`. Emit `Signal`s — never orders, never sizes.
2. Register it in `ct/strategies/__init__.py`.
3. Walk-forward it. Report the **deflated** Sharpe with the **honest** trial
   count — every variant you evaluated, not just the one you kept.
4. It starts `enabled: false`. Paper for 90 days. Then the gate decides.

### Known limitations

- ~~**The backtester is O(n²) in strategy indicators.**~~ **Fixed 2026-07-28**
  (ARCHITECTURE.md §7.1). Indicators are computed once per run and each bar gets
  a view truncated at that bar — legal because every indicator is causal, which
  is now asserted for all 26 indicator outputs. The 6,000-bar × 18-parameter ×
  5-fold walk-forward went from *not finishing in ten minutes* to **6.7 s**.
  All seven strategies emit byte-identical signals before and after.
- **Protective exits ignore the participation cap** — a stop always fills in
  full at the stop price regardless of the bar's volume. Optimistic, and it
  matters most for large positions in thin bars.
- **Exit-reason labelling is imprecise:** an exit is reported `trail` whenever
  `trail_atr_mult` is set, even if the trail never ratcheted above the original
  stop. Don't read the `stop` vs `trail` split as meaningful yet.
- **Warm-up bars are still counted** in `n_periods`, `exposure` and the return
  series, which biases Sharpe and CAGR (S2-3).
- Cross-exchange and funding-rate arbitrage are **monitor-only**: data is
  recorded, no executor exists, and both are likely closed to US persons anyway.
- `PaperBroker`'s queue model is an approximation. It is deliberately
  pessimistic (you are last in the queue), but it is not a matching engine, and
  it does **not** reserve funds at placement the way a real venue does.
