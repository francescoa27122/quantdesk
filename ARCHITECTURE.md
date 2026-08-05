# QuantDesk — Architecture & Decision Record

Self-hosted autonomous crypto trading platform for the UGreen DXP2800.
Source of truth for design decisions. Read this before changing anything under `ct/`.

**Status legend used throughout:** ✅ production-ready · 🟡 implemented but unproven · 🧪 experimental · ⛔ deliberately not built

---

## 0. The three facts that shaped every decision

### 0.1 Fees are the binding constraint

Kraken spot fee schedule at the `$0–$10k` 30-day volume tier:

| Order type | Fee | Round trip |
|---|---|---|
| Taker (market) | 0.40% | **0.80%** |
| Maker (resting limit) | 0.25% | **0.50%** |

Break-even gross edge required, by trade frequency:

| Trades / year | Taker-only | Maker-only |
|---|---|---|
| 50 | 40% /yr | 25% /yr |
| 200 | 160% /yr | 100% /yr |
| 1,000 | 800% /yr | 500% /yr |
| 5,000 | 4,000% /yr | 2,500% /yr |

**Consequence:** every intraday / high-frequency idea in the original brief (Stochastic RSI on 5m, VWAP reversion, tick-data scalping) is arithmetically dead at this fee tier. The platform is therefore designed around:

- **Primary timeframes: 4h and 1d.** 1h available for regime features, not for entries.
- **Maker-first execution.** Post-only limit at the passive side of the book, with a taker fallback only after `MAKER_TIMEOUT_S` and only if the signal is still valid. This is the single largest realizable alpha in the whole system: it is a guaranteed 30 bps/round-trip, which is larger than most published retail edges.
- **Fee model is a first-class input to the backtester**, not a post-hoc haircut. A backtest that ignores fees here is not optimistic — it is inverted.
- **Volume-tier awareness.** `fees.tier` is read from the exchange at startup and cached; the risk engine refuses to size a trade whose expected edge is below `MIN_EDGE_BPS` given the *current* tier.

> If you only take one thing from this document: **frequency is the enemy.** Every design pressure in QuantDesk pushes trade count down.

### 0.2 Hardware envelope (measured 2026-07-27, live)

```
CPU:   Intel N100, 4 cores, no SMT, no usable GPU (iGPU has no CUDA/ROCm)
RAM:   7,691 MB total · 4,602 MB available with `minecraft` stopped
       → ~1,900–2,100 MB available with `minecraft` running (3 GB cap)
Disk:  15 TB free on /volume1
Load:  12 containers already running (node-1, dashboard, health-collector,
       umami×2, sitehealth×4, workspace-mcp, tailscale, openclaw-gateway)
```

**Disk is free. RAM is the scarce resource. CPU is scarce during backtests only.**

RAM budget as designed:

| Container | `mem_limit` | Notes |
|---|---|---|
| `qd-db` | 1024m | TimescaleDB, `shared_buffers=256MB` |
| `qd-engine` | 512m | strategy + risk + execution, single process |
| `qd-collector` | 320m | websocket ingest + REST backfill |
| `qd-web` | 320m | FastAPI + HTMX dashboard & JSON API |
| `qd-worker` | 1024m | backtests / ML training, `cpus: 1.5`, lowest priority |
| `qd-proxy` | 64m | Caddy (optional — Tailscale-only is the default) |
| **Total** | **~3.2 GB** | |

**This does not fit alongside Minecraft on 8 GB.** Three options, in order of preference:

1. **Upgrade to 16 GB DDR5 SODIMM (~$40).** The DXP2800 has one SODIMM slot; 16 GB is supported. Everything fits with headroom. *Strongly recommended.*
2. Run `qd-worker` on-demand only (`docker compose run --rm worker backtest ...`) and stop it when idle. Saves 1 GB. Backtests then contend with Minecraft for CPU.
3. Run QuantDesk and Minecraft as mutually exclusive stacks.

The design assumes option 1 or 2. `qd-worker` is already `profiles: [batch]` in compose so it does **not** start by default.

### 0.3 Jurisdiction limits what can be built

You are US-based (`TZ=America/Chicago`). This is **register (c) — inference from your timezone, not verified**, and it materially gates three features from the brief:

| Feature | Status | Reason |
|---|---|---|
| Spot trading (Kraken, Coinbase) | ✅ available | |
| Perpetual futures / funding-rate arb | ⛔ not built | Binance.com, Bybit, OKX perps are closed to US persons. Kraken's US derivatives offering is limited by state and product. **Verify your own eligibility before asking for this.** |
| Futures basis trade | ⛔ not built | Same. |
| Cross-exchange arbitrage | 🧪 monitor-only | Requires pre-funded inventory on ≥2 venues, and net-of-fee/withdrawal spreads on US-legal venues are almost always negative. The collector *records* cross-venue spreads so you can decide from data, not vibes. No executor is wired. |

**Decision: build spot-only, single-venue first.** The exchange layer is an interface (`ct/execution/broker.py`) with a `CcxtBroker` implementation, so adding Coinbase Advanced is a config entry plus a fee-schedule row — not a refactor.

---

## 1. Container architecture — 6, not 13

The brief asked for 13 services. I'm delivering 6, and the merges are deliberate reliability decisions, not budget cuts.

```
                          ┌──────────────┐
   Kraken WS/REST ───────▶│ qd-collector │──┐
                          └──────────────┘  │
                                            ▼
                          ┌──────────────────────────┐
                          │        qd-db             │
                          │  TimescaleDB (Postgres)  │
                          └──────────────────────────┘
                             ▲          ▲          ▲
                             │          │          │
              ┌──────────────┘          │          └──────────────┐
              │                         │                         │
       ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
       │  qd-engine   │         │   qd-web     │         │  qd-worker   │
       │ strategy +   │         │ FastAPI +    │         │ backtest /   │
       │ risk + exec  │         │ HTMX + API   │         │ ML / reports │
       └──────┬───────┘         └──────────────┘         └──────────────┘
              │                         ▲
              ▼                         │  (Tailscale or Caddy)
       Kraken private API          your browser
```

### Merge 1 — trading engine + strategy engine + risk engine → `qd-engine`

**Rejected:** three containers with a message bus between them.

**Why:** these three components share one piece of mutable state — the authoritative position and open-order book — and every order must pass all three atomically. Splitting them puts a network hop *inside the critical order path*. That hop can fail after risk approval but before execution, or duplicate a message on retry, producing a double-sized position. Every distributed-systems mitigation for that (idempotency keys, two-phase commit, saga compensation) is strictly more failure modes than one process holding a lock.

**The rule this encodes: `qd-engine` is the single writer of order and position state.** Nothing else in the system may write to `orders`, `fills`, or `positions`. `qd-web` writes *intent* rows (`control_commands`) that the engine polls and applies. This is the same file-only-contract discipline as your agent platform, expressed in SQL.

**Cost accepted:** a strategy bug can crash the executor. Mitigated by running `Strategy.on_bar()` inside a try/except that quarantines the offending strategy (`strategy_state.status='quarantined'`) rather than killing the loop, and by `restart: unless-stopped` + a startup reconciliation pass (§6.3).

### Merge 2 — API gateway + dashboard → `qd-web`

One FastAPI app serves both the JSON API (`/api/v1/*`) and the server-rendered HTMX UI. A separate gateway in front of one backend adds a hop and an auth surface for zero benefit at one-operator scale. Rate limiting and auth live in FastAPI middleware.

### Merge 3 — notification service → transactional outbox in `qd-engine`

**Rejected:** a notification container consuming a queue.

**Why:** the failure that matters is "a trade was executed but you were never told." A separate service adds a second thing that can be down during exactly that event. Instead, the engine writes the notification into the **same database transaction as the fill** (`notifications` table, `status='pending'`), and a drain task retries with exponential backoff. If the drain is down, the message is not lost — it is sitting in a row you can see. If the engine is down, no trade happened.

Channels: **Telegram** ✅ (reuse your existing bot; secrets on `qd-engine` only, never `qd-web`), **Discord webhook** ✅, **SMTP email** 🟡, **Web Push** ⛔ (deferred — requires VAPID + service worker; the Telegram path already reaches your phone).

### Merge 4 — Prometheus + Grafana → deferred, with a hook

**Rejected for Phase 1 on RAM.** Prometheus (~250 MB) + Grafana (~200 MB) is a fifth of your remaining budget to display data that already lives in TimescaleDB and is already rendered by `qd-web`.

**What is built instead:** `qd-web` exposes `/metrics` in Prometheus text format from day one. If you upgrade RAM, adding a Prometheus + Grafana stack is a compose file, not a code change. Container/host health is already solved on your NAS by `health-collector` → `/workspace/health/metrics.json`; `qd-web` reads that file read-only rather than re-implementing it.

### Merge 5 — scheduler → `APScheduler` inside `qd-engine`

A dedicated scheduler container that triggers work in another container needs an RPC path. In-process `AsyncIOScheduler` with a Postgres jobstore survives restarts and has no network surface.

### Kept separate — `qd-collector`

Market data ingest has a genuinely different failure mode and restart cadence than trading (websocket reconnects, exchange maintenance windows, backfill jobs that run for hours). It must be able to die and restart without touching the order path, and the engine must be able to run off stale-but-known-stale data. This is the one split that buys real isolation.

### Kept separate — `qd-worker`

Backtests and ML training are CPU-hungry and would starve the engine's event loop on 4 cores. Isolated with `cpus: 1.5` and `profiles: [batch]` so it never runs unless invoked.

---

## 2. Technology decisions

| Layer | Choice | Alternatives rejected | Rationale |
|---|---|---|---|
| Language | **Python 3.12+** (the containers ship 3.14) | Rust, Go | The bottleneck is exchange round-trip latency (50–300 ms), not language speed. Python owns the quant ecosystem (pandas, LightGBM, ccxt, statsmodels). A one-person system that you can actually debug at 2 a.m. beats a faster one you can't. The floor lives in `requires-python`, the shipped version in `docker/Dockerfile`, and CI derives its test matrix from both rather than repeating either — see `.github/workflows/ci.yml`. |
| Exchange lib | **ccxt ≥4.x** (unified REST + `ccxt.pro` websockets, MIT) | Hand-rolled Kraken client | ccxt Pro was merged into the main MIT package in v4, so websockets are free. Unified API is the whole reason adding an exchange is cheap. *Verify the license/merge state at install — this is register (b) knowledge.* |
| Async | **asyncio**, one event loop per container | threads | Natural fit for websockets + concurrent REST. No GIL contention for I/O. |
| Database | **TimescaleDB** (Postgres 16 + extension) | InfluxDB, Postgres+Influx, SQLite, ClickHouse | One database. Full SQL joins between candles and trades — essential for feature engineering and attribution. Native columnar compression gives ~10–20× on OHLCV, so years of 1m candles fit in a few GB. InfluxDB means a second query language and no joins. Two databases do not fit in 2 GB of RAM. ClickHouse is excellent but its RAM floor is too high here. SQLite is used for exactly one thing: the local backtest result cache in `qd-worker`. |
| Web | **FastAPI + Jinja2 + HTMX + Chart.js** | React/Next SPA | Same stack and same reasoning as your existing dashboard: no node build step, tiny RAM, server-rendered. Real-time via **SSE** (`text/event-stream`), not websockets — one-way push is all a dashboard needs, and SSE reconnects itself. |
| Migrations | **Alembic** | hand-run SQL | Schema changes on a live trading DB need to be reversible. |
| Config | **pydantic-settings**, env + `config/*.yml` | bare env vars | Secrets from env / Docker secrets; strategy params from versioned YAML that is hashed into every backtest record for reproducibility. |
| Reverse proxy | **Tailscale only** (default) · Caddy optional | public HTTPS | You already gate Minecraft this way. A trading system's control plane should not be on the public internet, period. Caddy profile exists if you ever need LAN TLS. |
| Testing | pytest + hypothesis | | Property-based tests on the risk engine — see §7. |

---

## 3. Data model

Full DDL in [`db/schema.sql`](db/schema.sql). Design notes:

**Hypertables** (Timescale, time-partitioned + compressed after 7 days):
- `candles(exchange, symbol, timeframe, ts, o, h, l, c, v, trades, vwap)` — PK `(exchange, symbol, timeframe, ts)`. Compression `segmentby=(exchange,symbol,timeframe)`, `orderby=ts DESC`.
- `book_snapshots` — top-of-book L1 every 5 s, used for realistic slippage calibration. 30-day retention policy.
- `funding_rates` — recorded even though we don't trade perps, so the basis question can be answered from data later.

**Relational** (small, high-integrity):
- `orders` — every intent, with `client_order_id` UNIQUE. Idempotency key for retries.
- `fills` — partial fills are first-class rows; `orders.filled_qty` is derived, never authoritative.
- `positions` — current state, one row per `(account, symbol)`, with `avg_entry`, `qty`, `realized_pnl`.
- `equity_curve` — mark-to-market snapshot every minute, per account. Feeds every metric.
- `signals` — every signal a strategy emitted, *including the ones risk rejected*, with the rejection reason. Without this you cannot tell "the strategy is bad" from "risk is strangling it."
- `risk_events` — every breach, pause, and circuit-breaker trip.
- `control_commands` — the only thing `qd-web` may write. `{halt, resume, flatten_symbol, arm_live, disarm}` with `requested_by`, `applied_at`.
- `live_arming` — append-only. `armed_at`, `expires_at`, `confirmation_hash`, `max_notional_usd`, `disarmed_at`.
- `backtests` / `backtest_trades` — every run keyed by `(strategy, params_hash, data_hash, code_git_sha)`. Reproducibility is enforced, not hoped for.
- `notifications` — the transactional outbox.
- `audit_log` — append-only, every state transition with actor.

**The invariant that matters:** `positions` is derivable by replaying `fills`. A reconciliation job (§6.3) recomputes it from fills + the exchange's own balance report on every engine start and every hour, and raises a `risk_event` on any mismatch beyond dust tolerance. Disagreeing with the exchange about what you own is the most expensive bug class in this domain.

---

## 4. The single most important design decision: one code path, three brokers

```python
class Strategy(Protocol):
    def on_bar(self, ctx: Context) -> list[Signal]: ...
```

A strategy sees **only** `ctx` — a read-only view of history up to and including the current closed bar, plus current positions. It cannot see the future because the object does not contain it. It emits `Signal`s (direction, strength, stop/target intent) — **never orders**.

The same `Strategy` object runs identically under three brokers:

| Broker | Used by | Fills how |
|---|---|---|
| `BacktestBroker` | `qd-worker` | Simulated against next bar's OHLC + fee/slippage model |
| `PaperBroker` | `qd-engine` (`MODE=paper`) | Simulated against *live* book snapshots — real spreads, real latency |
| `CcxtBroker` | `qd-engine` (`MODE=live`) | Real orders |

Between the strategy and the broker sits the **risk engine**, which is the only component that turns a `Signal` into an `Order` (sizing it) and the only component that can veto.

```
Strategy.on_bar() → Signal → RiskEngine.evaluate() → Order | Rejection → Broker.submit()
```

**Why this matters more than any strategy:** the #1 cause of "it backtested at 3 Sharpe and lost money live" is that the backtest and the live system are different programs. Here they are the same program with a different `Broker` and a different clock. A discrepancy between paper and backtest results over the same period is therefore a *bug report*, and `qd-worker` runs that comparison nightly (§8.2).

### Look-ahead prevention, enforced structurally

- Bars are only delivered to `on_bar` **after close**. `ctx.bars[-1]` is the last *closed* bar.
- `Context` exposes numpy arrays that are slices, not references to the full series.
- Indicator functions in `ct/strategies/indicators.py` are all causal; each has a `warmup` property, and the engine refuses to emit signals before `warmup` bars have accumulated.
- The backtest engine fills at bar `t+1`, never at bar `t`'s close.
- Property test: for a random strategy, injecting arbitrary future data must not change any emitted signal (`tests/test_backtest.py::test_no_lookahead`).

---

## 5. Strategies

All are 🟡 **implemented but unproven** — none has earned live capital. That is the correct status for a strategy that has not completed the promotion gate (§8.3).

Implemented in `ct/strategies/`:

| Family | Strategy | File | Core logic |
|---|---|---|---|
| Trend | `EmaCrossStrategy` | `trend.py` | Fast/slow EMA cross, ADX ≥ threshold filter, ATR trailing stop |
| Trend | `SupertrendStrategy` | `trend.py` | ATR-band Supertrend flip, long-only above 200-EMA |
| Trend | `DonchianTrendStrategy` | `breakout.py` | Classic Turtle-style N-day channel break with 2N ATR stop |
| Reversion | `BollingerReversionStrategy` | `reversion.py` | Close < lower band **and** regime = ranging (ADX < 20). Exits at mid-band. |
| Reversion | `ZScoreReversionStrategy` | `reversion.py` | Rolling z-score of log price vs. its own mean; entry \|z\| > 2, exit \|z\| < 0.5 |
| Breakout | `AtrSqueezeBreakoutStrategy` | `breakout.py` | Bollinger-inside-Keltner squeeze release, direction by close vs. mid |
| Momentum | `MacdMomentumStrategy` | `trend.py` | MACD histogram sign change confirmed by ROC > 0 |

**Deliberately NOT implemented, with reasons:**

- **VWAP reversion** ⛔ — intraday by nature; dead at 0.8% round-trip (§0.1).
- **Stochastic RSI** ⛔ — the standard parameterization fires several times per day. Same arithmetic.
- **Ichimoku** ⛔ for now — it is five overlapping trend filters; it adds little beyond EMA+ADX while adding five parameters to overfit. Reconsider only if the walk-forward shows the existing trend book is capacity-constrained.
- **Kelly-optimal sizing at full Kelly** ⛔ — see §6.1.

**Strategy admission process** — a strategy is not "done" when it's written:
1. In-sample fit on 2018–2022 → params proposed.
2. Purged walk-forward (§8.1) on 2018–2026 → out-of-sample equity curve.
3. **Deflated Sharpe Ratio** computed against the *number of variants tried*. A raw Sharpe of 1.5 after 200 backtests is noise; DSR says so numerically.
4. Monte Carlo block-bootstrap → 5th-percentile max drawdown.
5. Paper trading, ≥90 days (§8.3).

---

## 6. Risk management — the core of the system

Implemented in [`ct/risk/engine.py`](ct/risk/engine.py) and [`ct/risk/sizing.py`](ct/risk/sizing.py). **Pure functions, no I/O, fully unit-tested.** The risk engine can be reasoned about without a database or an exchange, which is the point.

### 6.1 Position sizing — three caps, take the minimum

```
size_units = min(
    risk_budget_size,      # (equity × risk_per_trade) / (atr_mult × ATR)
    vol_target_size,       # (target_ann_vol / realized_ann_vol) × equity × weight / price
    notional_cap_size,     # max_position_pct × equity / price
)
```

- **Risk-budget sizing** ties position size to the *stop distance*, so a wide-stop trade in a volatile market is automatically smaller. This is the primary sizer.
- **Volatility targeting** normalizes portfolio risk across regimes. Realized vol is EWMA of log returns, annualized with **365** (crypto trades 24/7 — using 252 here overstates vol by ~19% and is a classic silent bug).
- **Notional cap** is the backstop.

**Kelly:** implemented as `fractional_kelly()` but **defaults to `kelly_fraction=0.0` (off)**. Full Kelly is provably growth-optimal *only* when you know the true edge distribution. You don't — you have a small, non-stationary sample. Half-Kelly on a mis-estimated edge routinely produces >60% drawdowns. If enabled, it is hard-capped at **quarter-Kelly** and can only *reduce* the size chosen above, never increase it.

### 6.2 The gate stack

Every signal passes all checks; **any** failure rejects the order and writes a row to `signals` with the reason.

| # | Gate | Default | Behavior on breach |
|---|---|---|---|
| 1 | Kill switch (file or DB) | — | Reject all; flatten if `KILL_FLATTEN=true` |
| 2 | Live arming valid & unexpired | — | Reject all live orders |
| 3 | Market data freshness | 3 bars | Reject; `risk_event` |
| 4 | Exchange connectivity / clock skew | 2 s | Reject |
| 5 | Per-order notional cap | 2% equity | Clamp, or reject if below exchange minimum |
| 6 | Per-symbol exposure cap | 15% equity | Clamp |
| 7 | Gross exposure cap | 60% equity | Clamp |
| 8 | Max concurrent positions | 5 | Reject new |
| 9 | Correlation cap | ρ>0.8 cluster ≤ 25% | Clamp |
| 10 | Leverage | **1.0 (spot only)** | Hard reject |
| 11 | Daily drawdown | 3% | **Halt new entries** for the UTC day |
| 12 | Weekly drawdown | 7% | Halt for the week |
| 13 | Max total drawdown from HWM | 20% | **Flatten all, disarm live, page** |
| 14 | Consecutive-loss breaker | 5 | Pause the *offending strategy* 24 h |
| 15 | Min expected edge vs. fees | 60 bps | Reject — the §0.1 gate, in code |
| 16 | Order-rate limiter | 30/hr | Reject; `risk_event` (runaway-loop backstop) |

Stops, trailing stops, and take-profits are attached at order construction and additionally **enforced server-side by the engine** on every bar — never relying solely on a resting exchange stop, because exchange stop orders can be cancelled by maintenance, rejected on thin books, and are visible to the venue.

### 6.3 Failure and recovery

- **Startup reconciliation.** On every engine boot: fetch exchange balances + open orders, rebuild positions from `fills`, compare all three. Any mismatch > dust → refuse to trade, raise `risk_event`, notify. **The engine starts in a halted state and must reconcile clean before it will place an order.**
- **Power loss.** Postgres is ACID; `orders` carry `client_order_id`, so on restart every non-terminal order is re-queried by that ID before any new order is placed. No blind resubmission, ever.
- **Exchange down.** Circuit breaker on the broker: 5 consecutive API failures → 5-minute backoff, notify, no order attempts. Existing stops are re-armed as soon as connectivity returns.
- **Emergency liquidation.** `flatten_all()` — market-sells every position, cancels every order, sets `halted`. Reachable from CLI, Telegram `/halt`, and the dashboard's red button. Deliberately taker-only: when you want out, you pay the spread.

---

## 7. Backtesting engine

[`ct/backtest/engine.py`](ct/backtest/engine.py) — event-driven, bar-close, one bar at a time.

**Rejected: vectorized backtesting** (shift-and-multiply on a DataFrame). It is 100× faster and structurally invites look-ahead: a single off-by-one on a `.shift()` silently produces a 5-Sharpe curve. The N100 backtests 8 years of 4h bars across 7 strategies in well under a minute event-driven. Speed is not the constraint; correctness is.

### 7.1 The indicator cache (2026-07-28) — how the loop got fast without going vectorized

Bar-close iteration is not what made this slow; **recomputation** was. Each
`on_bar` was handed a growing prefix and called `ema(ctx.close, 20)` over all of
it, so every indicator cost O(n²) across a run and the audit's B2 benchmark —
6,000 bars × 18 parameters × 5 folds — *did not finish in ten minutes* and was
killed.

`ct/strategies/features.py` computes each indicator **once over the whole
series** and hands each bar a view truncated at that bar. This is legal for
exactly one reason: **every indicator is causal**, so
`f(full)[i] == f(full[:i+1])[i]`. The backtest engine already leaned on this for
its ATR hoist; the cache generalises it, and `tests/test_features.py` now
asserts causality for **all 26 indicator outputs** rather than the 10 the
original test sampled.

The strategy keeps receiving an array whose last element is the current bar, so
the structural look-ahead guarantee is unchanged — there is still nothing ahead
of `index` to reach.

Measured, same machine, same code path:

| | Before | After |
|---|---|---|
| 1,500 bars, `ema_cross`, per-bar loop | 2,721 ms | **17 ms** (157×) |
| 6,000 bars × 18 params × 5 folds | did not finish in 10 min | **6.7 s** |

**The test that decided whether it shipped** is not the benchmark: it is
`test_the_cache_does_not_change_what_a_strategy_trades`, which asserts all
seven strategies emit byte-identical signals with the cache and without it. A
faster backtest that produces different signals is not an optimisation, it is a
new strategy wearing the old one's name.

**Cost model (all on by default):**
- Fees by maker/taker, from the live tier.
- **Slippage:** `spread/2 + impact`, where `impact = k · (order_notional / bar_dollar_volume)^0.5` (square-root market impact). `k` is calibrated per symbol from `book_snapshots` once live data exists; until then it defaults conservatively.
- Fills capped at a fraction of the bar's volume (`MAX_BAR_VOLUME_PCT`, default 1%) — you cannot backtest yourself into liquidity that wasn't there.
- Limit orders fill only if the bar's range actually traded through the price, and are treated as maker.

**Metrics** ([`ct/backtest/metrics.py`](ct/backtest/metrics.py)) — annualized with **365**, not 252:
Sharpe · Sortino · Calmar · Max Drawdown (+ duration) · CAGR · Profit Factor · Win Rate · Expectancy · Payoff ratio · Exposure · Turnover · **total fees paid** (shown next to net profit, because that ratio is the honest story) · trade-return distribution · **Probabilistic Sharpe Ratio** · **Deflated Sharpe Ratio**.

**Validation** ([`ct/backtest/walkforward.py`](ct/backtest/walkforward.py)):
- **Purged, embargoed walk-forward.** Standard k-fold leaks: label windows overlap fold boundaries. Purging drops overlapping samples; the embargo drops a further `embargo_pct` after each test fold. Non-negotiable for anything with a lookahead label.
- **Monte Carlo:** stationary block bootstrap of *trade returns* (preserves autocorrelation, unlike i.i.d. shuffling) → distribution of terminal equity and max DD. The number you act on is the **5th-percentile max drawdown**, not the mean.
- **Benchmark:** every report compares against buy-and-hold BTC and a 50/50 BTC/ETH rebalanced portfolio, net of the same fees. **A strategy that does not beat buy-and-hold on a risk-adjusted basis should not be run.** Most don't. The report says so plainly.

---

## 8. AI / ML — where it earns its place, and where it doesn't

**Firm rule, inherited from your operating manual: no LLM is ever in the order path.** Not for sizing, not for entry, not for exit. LLM output is non-deterministic, unauditable, and prompt-injectable via the news it reads.

| Application | Verdict | Implementation |
|---|---|---|
| **Regime classification** | ✅ build | Not an LLM. A 3-state Gaussian HMM on (realized vol, trend strength, volume z-score) → `{trending, ranging, crisis}`. Strategies subscribe to regime; reversion is disabled in `trending`, trend is disabled in `ranging`, everything is disabled in `crisis`. This is the highest-value ML component and it is small enough to be verifiable. |
| **Signal filtering (meta-labeling)** | ✅ build | LightGBM binary classifier: given that strategy X fired, will this trade hit its profit target before its stop? Trained on triple-barrier labels with purged CV. Used **only to veto or size down** — it can never create a trade. This asymmetry is what makes it safe. |
| **Trade confidence scoring** | ✅ | The meta-labeler's predicted probability, mapped to a size multiplier in `[0.0, 1.0]`. Never `>1.0`. |
| **Risk prediction** | 🟡 | GARCH(1,1) / EWMA vol forecast feeding the vol-targeting sizer. Classical, well-understood, tiny. |
| **Portfolio optimization** | 🟡 | Hierarchical Risk Parity (López de Prado) over strategy-level return streams. Rejected mean-variance: it is famously unstable to estimation error in expected returns, which is exactly what we can't estimate. HRP needs only the correlation matrix — **and that is the problem: the correlation matrix needs per-strategy equity curves, which do not exist** (§7A). Shipped 2026-07-28 instead: inverse-volatility weights from per-trade returns, labelled `inverse_vol` / `mixed` / `equal` so a report can never imply a risk model that did not run. HRP lands when the return streams do; that is a data dependency, not a scheduling one. |
| **Adaptive parameter tuning** | 🧪 | Walk-forward re-fit on a fixed quarterly cadence, with a hard rule: parameters may only change at a re-fit boundary, and the new set must beat the old one out-of-sample. Continuous online adaptation is a way to fit noise in real time. |
| **News sentiment** | 🧪 gated | An LLM *may* summarize headlines into a numeric score written to a table. That score is only ever readable by the risk engine as a **veto input** (e.g. "halt new entries on an exchange-insolvency headline"). It cannot open a position. Treated as untrusted input per your instruction-source boundary — headlines are data, never commands. |
| LSTM / Transformer / RL | ⛔ | No GPU on an N100. More importantly: the evidence for them beating gradient boosting on engineered features at daily/4h crypto frequency is weak, and each is dramatically easier to overfit and harder to debug. RL specifically requires a simulator whose fidelity we cannot establish. **Revisit only if LightGBM meta-labeling demonstrably plateaus.** |

**Every model must pass the same bar as a strategy:** purged walk-forward, deflated Sharpe, and a documented improvement over the no-model baseline. A model that doesn't beat "always trade the raw signal" gets deleted. Novelty is not a reason to ship.

### 8.1–8.3 Promotion gate (paper → live)

Live trading is refused by code until **all** hold, computed by `qd-worker` nightly and stored in `promotion_status`:

1. ≥ **90 calendar days** of continuous paper trading with no code changes to the strategy or risk config (enforced by `code_git_sha` equality).
2. ≥ **30 closed trades** (below that, no metric means anything).
3. Paper Sharpe ≥ **1.0**, net of modeled fees.
4. Paper max drawdown ≤ **1.5×** the backtest's 95th-percentile Monte Carlo drawdown.
5. Paper-vs-backtest return correlation ≥ **0.6** over the overlap period — i.e. the simulation is telling the truth.
6. Beats buy-and-hold BTC on Calmar over the same window.

You can override with `--force`, which writes a loud, permanent `audit_log` row and sends a notification. The gate exists to make going live a *decision*, not a drift.

### 8.4 Arming lifecycle — arm once, not weekly

**Design goal (user directive, 2026-07-27): the operator should not have to keep track of arming.**
An expiry that forces a human to re-arm every 7 days is a reliability problem, not a safety
feature: the predictable outcome is either a system that silently stops trading on day 8, or a
human who reflexively re-taps without reading. Both are worse than the alternative below.

**The model: a human originates; the engine sustains; anything meaningful re-asks.**

```
  ┌── human tap ────────────────────────────────────────────────┐
  │  qd arm-live  (typed phrase)  OR  Telegram "Arm" button      │
  │  → live_arming(origin='human', confirmation_hash=…,          │
  │                envelope_hash=H, renews_id=NULL)              │
  └──────────────────────────────┬───────────────────────────────┘
                                 │  engine renews before expiry,
                                 │  indefinitely, while hash == H
                                 ▼
        live_arming(origin='auto_renew', renews_id=parent, envelope_hash=H)
                                 │
        ┌────────────────────────┴─────────────────────────┐
        │ envelope_hash changes, or a hard event fires      │
        ▼                                                   ▼
   renewal REFUSED                                    engine DISARMS
   → notify + one-tap re-arm                          → notify + one-tap re-arm
```

**`envelope_hash`** = sha256 over (account, allowed symbols, allowed strategies,
`max_notional_usd`, `max_daily_notional_usd`, active `risk_config.id`, `code_git_sha`).
A renewal extends *the same decision*; it can never extend a different one. Add a symbol,
raise a cap, edit risk config, or change a line of strategy code, and the hash breaks —
renewal stops and a human tap is required. That is what makes indefinite renewal safe.

**Claude may** auto-renew, disarm, halt, flatten, and re-arm after its *own* disarm when the
cause was transient (exchange outage, stale feed, reconciliation mismatch since verified clean)
and the hash is unchanged. **Claude may not** originate the first arm, or re-arm after a disarm
caused by drawdown, promotion-gate regression, or any code/config change.

**Auto-disarm triggers** (each notifies, each needs a human tap to resume):

| Trigger | Claude may re-arm? |
|---|---|
| `max_total_drawdown` breached | **No** |
| Promotion gate regresses (`promotion_status.passed` → false) | **No** |
| `envelope_hash` changed (code, config, caps, symbols) | **No** |
| Reconciliation mismatch beyond dust | Yes, once verified clean |
| Exchange unreachable > 30 min | Yes, once reconnected + reconciled |
| Market data stale beyond `max_data_age_bars` | Yes, once fresh + reconciled |

#### The invariant, and how it is enforced

**Every arming chain must terminate in a human tap.** This is enforced in three independent
places, because the code-level check is the one most likely to be refactored away:

1. **Schema `CHECK` constraints** — `human_arming_is_rooted` (a human row has no parent and
   must carry a typed confirmation) and `renewal_must_have_parent` (a renewal without a parent
   cannot be inserted at all).
2. **`arming_is_valid(id)`** — walks the lineage to its root, returning false if any ancestor
   is disarmed, if the root is not human-confirmed, or if the chain is broken or cyclic.
3. **`active_arming` view** — the single query the engine runs before any live order.

**Verified against a real TimescaleDB instance on 2026-07-27** (throwaway container, since
removed). Nine cases; the results below are actual output, not intent:

| Test | Result |
|---|---|
| Human arming accepted, is valid | pass |
| 5-deep renewal chain stays valid | pass |
| Orphan renewal rejected **by the schema** | pass — `renewal_must_have_parent` |
| Human arming without typed confirmation rejected | pass — `human_arming_is_rooted` |
| **Disarming the root kills the whole chain** | **FAILED first, then fixed** |
| Exactly one governing row per account after renewals | pass — `DISTINCT ON` |
| Disarming a middle link kills descendants, spares ancestors | pass |
| Expired leaf inactive; older live row governs | pass |
| Cycle cannot be created (and would not hang) | pass — rejected by CHECK |

> **The failure is the important row.** The first version checked lineage but not ancestor
> disarm state. A renewal is its own row with its own `disarmed_at`, so disarming the root left
> all five children active — **a drawdown halt would have set `disarmed_at` on the root while
> the system kept trading.** The schema looked correct and was not. `arming_is_valid` now
> refuses if *any* ancestor is disarmed: disarm poisons the entire lineage. Expiry is still
> checked only on the leaf, since extending past a parent's expiry is the whole point.

---

## 7A. Portfolio backtesting — N symbols × M strategies, one account

Built 2026-07-27 (`ct/backtest/portfolio.py`). Status: ✅ **production-ready.**

`BacktestEngine` runs one strategy on one symbol, which meant four risk gates —
gross exposure, per-symbol exposure, correlation clusters, max open positions —
were written, unit-tested in isolation, and **never once executed against price
data.** A single-symbol account cannot breach a portfolio cap.

### Books, not positions

Every `(strategy, symbol)` pair gets its own book, keyed `"<strategy>|<symbol>"`.
Two strategies both long BTC hold two independent books, so a close from one
cannot flatten the other. Risk still reasons in *symbol* terms — exposure and
correlation caps aggregate every book on a symbol, because that is what the
exchange holds. This is the data-structure-level answer to "never let strategies
fight over capital": it is not a policy that can be forgotten, it is a key.

### One merged clock

Symbols do not share a bar timeline. The engine walks the **union** of bar
timestamps; a symbol with no bar at time *T* is not consulted at *T* and is
**never forward-filled**. A forward-filled bar is a price that did not trade, and
every gate downstream would treat it as real.

### The bug this found: caps were checked against a stale view

**Severity: would have lost real money.** The risk engine is stateless and
evaluates one signal against one snapshot. When N signals arrive in the same
engine pass — normal for a multi-symbol portfolio, and equally normal live —
each was checked against the *same* snapshot showing the exposure before any of
them. Each passed the cap independently. Together they breached it N-fold.

Measured on first run of the portfolio tests:

| Cap | Configured | Actually reached |
|---|---|---|
| Gross exposure | 30% | **50%** |
| Correlation cluster | 12% | **50%** |
| Max open positions | 1 | **2** |

**This was never a backtest artifact.** The live engine evaluates a batch of
signals per pass too, so the same breach would have happened with real capital
the first time two symbols signalled together.

**Fix:** `AccountState.pending` — approved-but-unfilled orders consume exposure
budget from the moment of approval, not of fill. Every cap now checks
*committed* exposure (filled + in flight), reservations are released as fills
arrive or orders expire, and the engine re-snapshots between signals in a pass.
After the fix, committed exposure peaks at **0.11999999** against a 0.12 cap.

### Caps bind at entry, not continuously

`peak_gross_exposure` (mark-to-market) can legitimately exceed the cap;
`peak_committed_exposure` (at approval) cannot. A held position that appreciates
carries exposure up with it, and the system deliberately does **not**
force-deleverage: selling a winner to satisfy a bookkeeping limit is a worse
policy than carrying the breach. Both numbers are reported so the distinction is
visible rather than argued about later.

### What a real multi-strategy run showed

3 strategies × 4 symbols × 2,200 4h bars, default `RiskLimits`:

```
return +14.6%   maxDD 4.5%   Sharpe 1.79   DSR 0.96
trades 163      fees 754     benchmark +721.9%   beats: False
peak committed exposure 40.0% (cap 60%)   peak open books 5 (cap 5)
rejections: MAX_POSITIONSx34, CORRELATION_CAPx6
```

Two things worth keeping:

1. **The binding constraint was `max_open_positions`, not exposure.** 34
   rejections came from the position-count cap while gross exposure never
   reached two-thirds of its limit. Tuning exposure limits would have changed
   nothing.
2. **Summing per-symbol backtests overstates deployable P&L by roughly the
   number of books.** The same three strategies run in isolation across the same
   four symbols summed to **+95,933**; deployed as one portfolio they made
   **+1,459**. Each isolated run assumes it owns the entire account — twelve of
   them assume twelve accounts. Any strategy book evaluated by adding up
   single-symbol backtests is inflated by about that factor.

---

## 7B. History is not available from the trading venue

**Measured 2026-07-28, not assumed.**

```
1h : 721 candles, oldest 2026-06-28   (30 days)
4h : 721 candles, oldest 2026-03-30   (4 months)
1d : 721 candles, oldest 2024-08-07   (2 years)

since = 10 days ago  ->  240 candles     <- since IS honoured
since = 60 days ago  ->  721 candles
since =  5 years ago ->  721 candles     <- same window; cannot page back
```

`since` works correctly as a forward cursor — the 720 figure is a **retention
limit**, not a pagination bug. Two consequences, and they point opposite ways:

* **Bad:** deep history cannot be fetched from the API at all. No amount of
  paging reaches 2018.
* **Good:** REST gap-fill after a websocket disconnect *does* work, because
  `since` addresses any point inside the retained window. That is exactly what
  the collector needs, and it is why `ct/data/backfill.py` is cursor-based.

So `qd backfill --since 2018-01-01` — written into the deploy guide and assumed by the
whole validation design — **cannot work against the live venue.** Purged walk-forward over
2018–2026, the promotion gate's regime coverage, and the deflated Sharpe's sample size all
rested on history the exchange does not serve.

Three ways out, none free:

1. **Kraken's bulk OHLCVT download** — **CHOSEN 2026-07-28.** Full history as CSVs in
   per-pair, per-interval files (1/5/15/30/60/240/720/1440 minutes), published as ZIPs on
   two **Google Drive links** in [Kraken's support article](https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data):
   one complete archive, plus quarterly incrementals. Same venue, same prices, so backtest
   and live still agree.

   ⚠ **Two caveats, both stated rather than smoothed over.** (a) It is a *manual browser
   download* — Google Drive, no direct URL, so this is the one step Claude cannot do.
   (b) **Kraken's own developer docs contradict the support article**, saying "Kraken does
   not provide a bulk historical data dump." The support article describes the mechanism in
   detail (single ZIP + quarterly updates); the dev docs deny it exists. Unresolved without
   a browser. `ct/data/history.py` imports the documented CSV format either way, so if the
   links are live the data lands; if they are not, we fall back to (3) having lost nothing.
2. **A different source for history, Kraken for live** — Binance's public data dumps go back
   years and are free. But backtesting on one venue's prices and trading on another's means
   the fee model, the spread and the fills all differ from what was simulated. That is the
   same class of error as a backtest/live divergence, imported deliberately.
3. **Collect forward** — the collector stores everything from first run. Honest and exact,
   but 4h bars need ~90 days before the promotion gate has anything to judge.

**Recommendation: (1), falling back to (3).** Option 2 buys history at the cost of the one
property this project has spent the most effort protecting — that the simulation and the
live system see the same world.

Until one is chosen, **treat every multi-year backtest number in this repo as
hypothetical.** The engine is correct; the data to feed it is not yet obtainable.

---

## 8A. Events — the observability spine

Built 2026-07-27 (`ct/events/`, 1,401 lines + 31 tests). Status: ✅ **production-ready.**

### The rule that shapes it

**Events are an output. They are never the order-path control flow.**
`Signal → Risk → Order → Broker → Fill → Position` stays a synchronous in-process
call chain; each transition *emits* an event afterwards. Nothing in the trading path
waits on, reads from, or is scheduled by the bus.

This is enforced structurally, not by convention:

- `Event` is frozen and carries no callbacks, futures or handles. An event cannot be
  used to *ask* for anything — only to report that something already happened. Every
  `EventType` is named in the past tense because that is all one can be.
- There is no acknowledgement, no retry, no delivery guarantee and no reply channel.
  None of the machinery that would let the bus creep into the order path exists.

### The one contract

**`publish()` never raises and never blocks.** Everything else follows. All of these
are swallowed, counted and reported rather than propagated: a subscriber that raises,
a subscriber that is slow, a full or dead disk, a closed asyncio loop. A bus that can
throw or stall would make observability a net liability — the whole point is that
losing telemetry must never cost a stop-loss.

The corollary: **the log is the authority, not a subscriber queue.** Queues are
bounded and drop under pressure by design; `SUBSCRIBER_LAGGED` makes every drop
visible, so a gap is never silent.

| Delivery mode | Use | Failure behaviour |
|---|---|---|
| `subscribe(handler)` | fast in-process handlers: counters, projections | 3 raises → **quarantined permanently**. Degrade to no observability, never to a broken engine. |
| `stream()` | dashboard SSE, anything possibly slow | bounded, **drop-oldest**, lag counted and announced |

A meta-event is never delivered to the subscriber it is *about*. Pushing "your buffer
overflowed" into the buffer that just overflowed evicts a real event to report an
eviction — it doubled every drop count — and delivering "your handler raised" to the
handler that just raised makes it raise again inside the failure path.

### Durability, tiered

| Class | Policy | Rationale |
|---|---|---|
| Money-relevant (fills, orders, arming, risk breaches, reconciliation) | **fsync immediately** | must survive a power cut |
| Everything else | flush on 32 events **or 1 second**, whichever first | fsyncing a heartbeat on a 4-core NAS is pure cost |

The *time* bound is not decoration. Count-based flushing alone bounds lag by event
volume, which on a quiet system is no bound at all — a written event could sit in
`qd-engine`'s buffer indefinitely, invisible to `qd-web` reading the same file from
another container. Found by testing across two log instances rather than reading back
through the writer's own handle.

A truncated final line — the ordinary result of `docker kill` — is skipped and
counted, never raised. A log that refuses to be read after a crash is not a log.

### Sinks

`JsonlEventLog` (date-rotated files, readable with `jq` on the NAS without this
codebase) and `InMemoryEventLog` (bounded, for tests and the backtester) are both
complete implementations of one protocol. A Postgres sink lands with the persistence
layer against the `events` table already in the schema. `NullEventBus` is the same
*type* as the real bus deliberately — `if bus is not None` scattered through the
engine is how backtest and live drift apart again.

### Why `events` is the one table that is NOT a hypertable

TimescaleDB refuses a unique index that excludes the partitioning column:

```
cannot create a unique index without the column "ts" (used in partitioning)
```

(verified, timescaledb 2.17.2-pg16). The best a hypertable allows is
`UNIQUE (seq, ts)` — which permits the same `seq` at two timestamps, and that is
exactly the corruption worth preventing. Two `qd-engine` containers running at once,
or a bus that failed to recover its high-water mark, would silently produce duplicate
sequences and quietly break every replay spanning them.

The trade is disk, and **disk is not the scarce resource here**: ~3k events/day is
~1.1M rows and well under a GB per year against 15 TB free. A plain table with a BRIN
index on `ts` (a few KB, ideal for append-only time-ordered data) gets the strong
invariant for free. RAM is what is scarce, and a plain table uses less of it.

### Replay

`correlation_id` groups a whole lifecycle (signal → risk → order → fill → close);
`causation_id` points at the event that caused this one, so replay builds a causal
tree rather than a flat timeline. From that:

- **Trade replay** and the dashboard's **execution timeline**.
- **Execution quality** — decision latency, fill latency, signed slippage (positive
  always means worse for us), and **maker ratio**, the direct measure of whether the
  15 bps/side maker-first edge (§0.1) is actually being realised.
- **`rebuild_positions()`** — replays fills from the log alone, sharing no code with
  the live position bookkeeping. Agreement is therefore real evidence, not the same
  bug twice: a genuine third opinion for the reconciliation invariant in §6.3.

> **A bug worth recording.** The SQL `trade_lifecycles` view first computed
> `maker_ratio` without coalescing the numerator, so an all-taker lifecycle yielded
> `NULL` instead of `0` — indistinguishable from "no fills". A dashboard averaging
> that column would have silently dropped exactly the trades with the *worst*
> execution. On the four-lifecycle fixture the average read **0.625** where the truth
> is **0.4167**: a 50% overstatement of the single metric that validates the largest
> claimed edge in the project. Caught only by running the view against real Postgres.

---

## 9. Security

- **Secrets:** Docker secrets (file-mounted, `*_FILE` env convention), never in `docker-compose.yml`, never in the image, never in a log. `qd-web` and `qd-worker` **have no exchange credentials at all** — only `qd-engine` does. A dashboard compromise cannot trade.
- **Exchange API keys:** created with **trading enabled, withdrawal DISABLED, IP-allowlisted** to your NAS's egress IP. This is the single most effective control in this document — with withdrawal disabled, total compromise of the NAS costs you slippage, not the balance. Set it up this way before the first key is ever pasted.
- **Encryption at rest:** API keys additionally sealed with an age/`sops` key that lives outside the container; the engine decrypts at boot into memory only.
- **Network:** Tailscale-only by default. `qd-db` is on an internal Docker network with **no published port**. Nothing binds `0.0.0.0` except the proxy.
- **Auth:** session cookie + argon2id password + optional TOTP on `qd-web`. Roles: `viewer` (read), `operator` (halt/flatten/resume), `admin` (arm live, change risk config). Halt is available to *every* role — stopping is never gated.
- **Rate limiting:** per-IP on auth endpoints, per-session on control endpoints.
- **Audit log:** append-only, every control action, every arming, every config change, with actor and before/after.
- **Backups:** `pg_dump` nightly + WAL archiving to `/volume1`, plus the existing PC mirror pull. **Verified by a weekly automated restore into a scratch container** — an unrestored backup is a rumor.
- **Fail-safe shutdown:** `SIGTERM` → cancel all open orders → persist state → exit. Positions are *not* auto-flattened on shutdown (that would turn a container restart into a taxable market sell); the resting stops remain the protection, and the engine re-arms them on boot.

---

## 10. Dashboard

Dark mode, SSE-driven, HTMX partial swaps, Chart.js. Wireframes:

```
┌─ QuantDesk ──────────────────────── ● LIVE ARMED (expires 3d) ─ [HALT] ─┐
│                                                                         │
│  Equity  $10,482.13    Today +0.82%   Week +2.1%   MaxDD -4.3%          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │        equity curve (vs BTC buy&hold, drawdown shaded)            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  OPEN POSITIONS                          RISK BUDGET                    │
│  ┌──────────────────────────────────┐    ┌──────────────────────────┐   │
│  │ BTC/USD  0.041  +2.3%  stop 61.2k│    │ Gross    38% / 60%  ▓▓▓░ │   │
│  │ ETH/USD  0.88   -0.4%  stop 3.1k │    │ Daily DD  0.9% / 3%  ▓░░ │   │
│  └──────────────────────────────────┘    │ Positions   2 / 5        │   │
│                                          └──────────────────────────┘   │
│  STRATEGY PERFORMANCE          │  LIVE SIGNALS        │  SYSTEM         │
│  ema_cross    +4.2%  SR 1.3 ✅ │  ema_cross BTC LONG  │  Kraken   ●     │
│  bb_revert    -1.1%  SR 0.2 🟡 │  bb_revert ETH  —    │  Data 4s ●      │
│  donchian     +0.9%  SR 0.6 🟡 │  (rejected: gross    │  CPU 34% RAM 61%│
│                                │   exposure cap)      │  Disk 1% ●      │
└─────────────────────────────────────────────────────────────────────────┘
```

Pages: **Overview** (above) · **Positions & Orders** · **Trades** · **Strategies** · **Risk** · **System** · **Activity**.

Every number on the dashboard is read from the DB. `qd-web` computes nothing that the engine also computes — two implementations of P&L that disagree is a support nightmare.

### 10.1 Built 2026-07-28 (`ct/web/`, 4,813 lines + 56 tests). Status: ✅ **production-ready**, undeployed.

**Three deviations from the plan above, stated rather than smuggled.**

1. **No Chart.js. Charts are server-rendered SVG** (`ct/web/charts.py`). A
   control plane must not need the internet to draw its own equity curve, and
   vendoring 200 KB of minified third-party JavaScript into a repo whose whole
   security posture is "small and auditable" is worse than the dependency it
   replaces. The cost is real and accepted: no pan, no zoom, a redraw is a
   request.
2. **No HTMX either.** The parts of it this dashboard uses are a `fetch` and an
   `innerHTML` swap — about forty lines. Same reasoning.
3. **The two together buy one property worth more than either:** the
   Content-Security-Policy is `script-src 'self'; style-src 'self'` with **no
   exceptions** — no `unsafe-inline`, no `unsafe-eval`, no third-party origin.
   Nothing on the page comes from anywhere but the NAS.

> That CSP immediately caught a bug of its own making. The rejection histogram
> drew its bars with `style="width:41%"`, and `style-src 'self'` blocks inline
> style *attributes* — so five bars with five different counts all rendered
> full length. A chart that is confidently wrong is worse than no chart. Bars
> are now SVG `<rect>` elements, whose `width` is a presentation attribute that
> nothing strips. **Found by screenshotting the page, not by reading it.**

**Two pages were built against different tables than planned**, because the
planned ones have no writer anywhere in the codebase:

- **Trades** reads the `trade_lifecycles` view over `events`, not the `trades`
  table. Nothing writes `trades`, so that page would have sat permanently empty
  and read as "no trades yet" rather than "nothing records this". The view also
  carries the rejection reason, the maker ratio and the causal chain.
- **Strategies** is derived from `signals` + `fills` + `positions`, not the
  `strategies` table, which nothing populates. The engine's roster comes from
  `QD_STRATEGIES` at boot, so the truthful answer to "what is running" is "what
  has produced rows".

**`risk_events` had no writer at all** — `RiskEventRepository` was called from
nowhere. Both `qd status` and the Risk page would have reported *0 unresolved
risk events* whether or not anything had breached. The engine now writes one on
a drawdown halt, a reconciliation mismatch and a strategy quarantine. A green
light that cannot turn red is worse than no light.

**What the dashboard cannot do**, structurally rather than by policy: place or
cancel an order (no such code path), hold an exchange credential (none is
mounted), or arm live trading (no route exists, and the phrase is typed at a
terminal or not at all). Its only writes are `control_commands` — *intent* the
engine polls — and `audit_log`.

**Deliberately not built: TOTP.** §9 lists it as optional. A second factor
without an enrolment flow, recovery codes and a re-key path is theatre that
locks you out at the worst moment, and the surface is already Tailscale-only.
Stated as a gap rather than half-built.

---

## 11. Roadmap

| Phase | Deliverable | Gate to exit |
|---|---|---|
| **1 — Foundation** | Compose stack, TimescaleDB + schema, config, exchange adapter (read-only), dashboard shell, Tailscale access | Stack survives a NAS reboot; `qd-web` shows live BTC price |
| **2 — Data** | Websocket ingest, backfill from a **bulk history source** (see below), compression + retention, gap detection | As much history as the chosen source actually provides, zero gaps, < 5 GB |
| **3 — Backtesting** | Backtest engine, cost model, metrics, purged walk-forward, Monte Carlo, HTML reports | The 7 strategies produce reproducible reports; the no-lookahead property test passes |
| **4 — Risk & paper** | Risk engine wired live, PaperBroker on real book data, reconciliation, notifications, kill switch | 30 days of paper trading with zero reconciliation mismatches |
| **5 — ML** | Regime HMM, LightGBM meta-labeler, HRP allocation | Each model beats its no-model baseline out-of-sample, or is deleted |
| **6 — Live** | `CcxtBroker`, arming ceremony, promotion gate, maker-first execution | Promotion gate passes on ≥1 strategy; first live trade at minimum size |
| **7 — Hardening** | Prometheus/Grafana (post-RAM-upgrade), restore drills, second exchange adapter, capacity analysis | Weekly automated restore drill green for a month |

**Future enhancements** (explicitly out of scope now): perp/funding arb pending jurisdiction; options; on-chain data features; multi-account; tax-lot accounting and Form 8949 export; strategy marketplace/hot-reload.

---

## 12. What is honestly production-ready

✅ **Ready:** container architecture, database schema, risk engine, backtest engine + cost model, metrics incl. DSR, walk-forward, paper broker, secrets posture.

🟡 **Implemented, unproven:** all 7 strategies (no strategy is proven until §8.3 passes), notification delivery, dashboard.

🧪 **Experimental:** meta-labeling, regime HMM, adaptive re-fit, news sentiment.

⛔ **Not built:** live executor arming ceremony (Phase 6), derivatives, cross-exchange arb execution, Web Push, Grafana.

**The most likely outcome, stated plainly:** most retail systematic crypto strategies do not beat buy-and-hold net of 0.5–0.8% round-trip fees. This platform is built to *find that out cheaply and honestly* rather than to discover it with money. The benchmark comparison and the deflated Sharpe are in the reports specifically so the system can tell you "don't trade this." A platform that can say no is worth more than one that always finds a signal.
