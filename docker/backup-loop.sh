#!/bin/sh
# qd-backup — nightly pg_dump of the QuantDesk database, plus a weekly restore
# drill into a scratch cluster INSIDE this container (never into qd-db, whose
# 1 GB mem_limit is budgeted for trading, not for restore verification).
#
# Runs in the stock timescale/timescaledb image so pg_dump/pg_restore exactly
# match the server version — a dump taken with a mismatched client is the kind
# of backup that fails only on the day it is needed.
#
# Usage:
#   backup-loop.sh          # the container entrypoint: schedule loop
#   backup-loop.sh backup   # one backup now (docker exec qd-backup ...)
#   backup-loop.sh drill    # one restore drill now, against the newest dump
#
# Connection comes from PGHOST/PGUSER/PGDATABASE/PGPASSWORD in the compose
# environment. Status lands in $BACKUP_DIR/latest.json and drill-latest.json;
# the compose healthcheck greps those, so a failed or stale backup turns the
# container unhealthy and shows up on the Mission Control NOC page.
set -u

BACKUP_DIR=${QD_BACKUP_DIR:-/backups}
DUMP_TIME=${QD_BACKUP_TIME:-0330}     # HHMM, container TZ (America/Chicago)
DRILL_DOW=${QD_DRILL_DOW:-7}          # ISO weekday for the drill; 7 = Sunday
KEEP_DAILY_DAYS=14
KEEP_WEEKLY_DAYS=56
HOST_OWNER=1000:10001                 # <admin-user> : image group

DRILL_DIR=/tmp/qd-drill
DRILL_SOCK=/tmp
DRILL_PORT=5544
DRILL_DB=quantdesk_drill

log()  { echo "$(date '+%F %T') qd-backup: $*"; }
fail_json() {
    # $1 = target json, $2 = message. Written fresh so the healthcheck's
    # ok-grep fails while the staleness check still sees a recent attempt —
    # unhealthy on the NOC page, no retry-thrash until the next window.
    printf '{\n  "ok": false,\n  "error": "%s",\n  "finished": "%s"\n}\n' \
        "$2" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$1"
    chown "$HOST_OWNER" "$1" 2>/dev/null
    log "FAILED: $2"
}

live_psql() { psql -X -A -t -v ON_ERROR_STOP=1 -c "$1"; }

wait_for_db() {
    # Wait for the database to actually answer a query, up to $1 seconds.
    #
    # `depends_on: service_healthy` is not enough on a FIRST deploy. That gate
    # is pg_isready, and Postgres reports ready, then the entrypoint runs
    # /docker-entrypoint-initdb.d and restarts the server around it. This
    # container therefore wins the race, the catch-up backup fails with
    # "pre-dump count query failed", and — by the deliberate no-retry rule
    # below — it then sits unhealthy until the next nightly window. Roughly
    # 24 hours of a red tile on Mission Control for a boot race.
    #
    # Observed exactly that on a clean `docker compose up`: latest.json said
    # ok:false while `psql -h qd-db -c "select 1"` from this same container
    # answered fine a minute later.
    #
    # This is a dependency wait, not a retry: it runs BEFORE the first attempt,
    # so the "a failed attempt does not retrigger" rule is untouched.
    waited=0
    limit="${1:-120}"
    while [ "$waited" -lt "$limit" ]; do
        if live_psql "select 1" >/dev/null 2>&1; then
            [ "$waited" -gt 0 ] && log "database answered after ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "database still unreachable after ${limit}s; proceeding anyway so the failure is recorded"
    return 1
}

do_backup() {
    day=$(date +%F)
    suffix=""
    [ "$(date +%u)" = "$DRILL_DOW" ] && suffix="-weekly"
    out="$BACKUP_DIR/quantdesk-$day$suffix.dump"
    tmp="$BACKUP_DIR/.inprogress.dump"
    rm -f "$tmp"
    started=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    t0=$(date +%s)

    # Counts captured immediately before the dump. The dump snapshot begins
    # after this capture, so a restored count may legitimately exceed these —
    # the drill therefore checks >= for rows and == for structure.
    tables=$(live_psql "select count(*) from information_schema.tables where table_schema='public' and table_type='BASE TABLE'") || { fail_json "$BACKUP_DIR/latest.json" "pre-dump count query failed - is qd-db reachable?"; return 1; }
    hypertables=$(live_psql "select count(*) from timescaledb_information.hypertables")
    candles=$(live_psql "select count(*) from candles")
    events=$(live_psql "select count(*) from events")
    orders=$(live_psql "select count(*) from orders")
    fills=$(live_psql "select count(*) from fills")

    if ! pg_dump -Fc --no-owner -f "$tmp" 2>/tmp/pg_dump.err; then
        fail_json "$BACKUP_DIR/latest.json" "pg_dump failed: $(head -c 300 /tmp/pg_dump.err | tr '\n\"' ' .')"
        rm -f "$tmp"
        return 1
    fi

    # Integrity: the archive must be listable and must actually contain the
    # schema. A zero-length or truncated file passes an existence check and
    # nothing else.
    toc_entries=$(pg_restore --list "$tmp" 2>/dev/null | wc -l)
    if [ "$toc_entries" -lt 20 ]; then
        fail_json "$BACKUP_DIR/latest.json" "archive integrity check failed: pg_restore --list returned $toc_entries entries"
        rm -f "$tmp"
        return 1
    fi

    sha=$(sha256sum "$tmp" | cut -d' ' -f1)
    bytes=$(wc -c < "$tmp")
    mv "$tmp" "$out"

    cat > "$out.meta" <<EOF
sha256=$sha
bytes=$bytes
tables=$tables
hypertables=$hypertables
candles=$candles
events=$events
orders=$orders
fills=$fills
EOF

    cat > "$BACKUP_DIR/latest.json" <<EOF
{
  "ok": true,
  "file": "$(basename "$out")",
  "bytes": $bytes,
  "sha256": "$sha",
  "tables": $tables,
  "hypertables": $hypertables,
  "candles": $candles,
  "events": $events,
  "orders": $orders,
  "fills": $fills,
  "toc_entries": $toc_entries,
  "started": "$started",
  "finished": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "duration_s": $(( $(date +%s) - t0 ))
}
EOF
    chown "$HOST_OWNER" "$out" "$out.meta" "$BACKUP_DIR/latest.json" 2>/dev/null
    chmod 640 "$out" "$out.meta" "$BACKUP_DIR/latest.json" 2>/dev/null
    log "backup OK: $(basename "$out") ($bytes bytes, $candles candles, $events events, ${toc_entries} TOC entries)"

    prune
    return 0
}

prune() {
    # Two tiers: daily dumps live $KEEP_DAILY_DAYS days, Sunday dumps (named
    # *-weekly) live $KEEP_WEEKLY_DAYS. rm loop rather than find -delete —
    # busybox find does not reliably have -delete, and each dump drags its
    # .meta along.
    find "$BACKUP_DIR" -name 'quantdesk-*.dump' ! -name '*-weekly.dump' -mtime +"$KEEP_DAILY_DAYS" 2>/dev/null | while read -r f; do
        log "prune: $(basename "$f")"; rm -f "$f" "$f.meta"
    done
    find "$BACKUP_DIR" -name 'quantdesk-*-weekly.dump' -mtime +"$KEEP_WEEKLY_DAYS" 2>/dev/null | while read -r f; do
        log "prune: $(basename "$f")"; rm -f "$f" "$f.meta"
    done
}

mirror_host_health() {
    # The System page's Host panel reads the agent platform's metrics.json.
    # qd-web (uid 10001) cannot read the source directly: UGOS layers its own
    # ACLs over the POSIX bits (mode shows 777, reads are denied, and POSIX
    # setfacl grants are not honoured — measured 2026-07-30). This container
    # runs as root, so it mirrors the one file into ./state/hosthealth with
    # ownership qd-web can read. Copy-then-rename so qd-web never sees a
    # half-written JSON.
    [ -r /hosthealth-src/metrics.json ] && [ -d /hosthealth-out ] || return 0
    cp /hosthealth-src/metrics.json /hosthealth-out/.metrics.tmp 2>/dev/null || return 0
    chown "$HOST_OWNER" /hosthealth-out/.metrics.tmp 2>/dev/null
    chmod 640 /hosthealth-out/.metrics.tmp 2>/dev/null
    mv /hosthealth-out/.metrics.tmp /hosthealth-out/metrics.json
}

drill_cleanup() {
    if [ -d "$DRILL_DIR" ]; then
        su-exec postgres pg_ctl -D "$DRILL_DIR" -m immediate stop >/dev/null 2>&1
        rm -rf "$DRILL_DIR"
    fi
}

drill_psql() { psql -h "$DRILL_SOCK" -p "$DRILL_PORT" -U postgres -X -A -t -v ON_ERROR_STOP=1 "$@"; }

do_drill() {
    # Restore the newest dump into a throwaway cluster in this container's
    # /tmp, then compare structure and row counts against the .meta captured
    # at dump time. An unrestored backup is a rumour; this is the receipt.
    dump=$(ls -t "$BACKUP_DIR"/quantdesk-*.dump 2>/dev/null | head -1)
    if [ -z "$dump" ]; then
        fail_json "$BACKUP_DIR/drill-latest.json" "no dump found to drill"
        return 1
    fi
    meta="$dump.meta"
    if [ ! -f "$meta" ]; then
        fail_json "$BACKUP_DIR/drill-latest.json" "no .meta beside $(basename "$dump")"
        return 1
    fi
    command -v su-exec >/dev/null 2>&1 || { fail_json "$BACKUP_DIR/drill-latest.json" "su-exec missing from image"; return 1; }

    exp_tables=$(sed -n 's/^tables=//p' "$meta")
    exp_hypertables=$(sed -n 's/^hypertables=//p' "$meta")
    exp_candles=$(sed -n 's/^candles=//p' "$meta")
    exp_events=$(sed -n 's/^events=//p' "$meta")
    exp_sha=$(sed -n 's/^sha256=//p' "$meta")
    t0=$(date +%s)

    sha_now=$(sha256sum "$dump" | cut -d' ' -f1)
    if [ "$sha_now" != "$exp_sha" ]; then
        fail_json "$BACKUP_DIR/drill-latest.json" "dump sha256 mismatch - file corrupted since backup"
        return 1
    fi

    drill_cleanup
    mkdir -p "$DRILL_DIR" && chown postgres "$DRILL_DIR"
    if ! su-exec postgres initdb -D "$DRILL_DIR" -A trust --no-sync >/tmp/drill-initdb.log 2>&1; then
        fail_json "$BACKUP_DIR/drill-latest.json" "initdb failed: $(tail -c 200 /tmp/drill-initdb.log | tr '\n\"' ' .')"
        drill_cleanup; return 1
    fi
    if ! su-exec postgres pg_ctl -D "$DRILL_DIR" -w -t 60 -l /tmp/drill-pg.log start \
        -o "-c listen_addresses='' -c unix_socket_directories=$DRILL_SOCK -c port=$DRILL_PORT -c shared_preload_libraries=timescaledb -c shared_buffers=32MB -c timescaledb.telemetry_level=off"; then
        fail_json "$BACKUP_DIR/drill-latest.json" "scratch postgres failed to start: $(tail -c 200 /tmp/drill-pg.log | tr '\n\"' ' .')"
        drill_cleanup; return 1
    fi

    ok=1
    reason=""
    # Timescale's documented restore procedure: create the extension at the
    # same version, enter pre-restore mode, pg_restore, leave it. pg_restore
    # re-runs the dump's own CREATE EXTENSION and reports "already exists" —
    # that error is expected and harmless, which is why there is no
    # --exit-on-error: verification is the count comparison below, not
    # pg_restore's exit code.
    drill_psql -d postgres -c "CREATE DATABASE $DRILL_DB" \
      && drill_psql -d "$DRILL_DB" -c "CREATE EXTENSION timescaledb" \
      && drill_psql -d "$DRILL_DB" -c "SELECT timescaledb_pre_restore()" \
      || { ok=0; reason="scratch db/extension setup failed"; }

    if [ "$ok" = 1 ]; then
        pg_restore -h "$DRILL_SOCK" -p "$DRILL_PORT" -U postgres -d "$DRILL_DB" --no-owner "$dump" 2>/tmp/drill-restore.err
        drill_psql -d "$DRILL_DB" -c "SELECT timescaledb_post_restore()" >/dev/null 2>&1
        got_tables=$(drill_psql -d "$DRILL_DB" -c "select count(*) from information_schema.tables where table_schema='public' and table_type='BASE TABLE'" 2>/dev/null || echo -1)
        got_hypertables=$(drill_psql -d "$DRILL_DB" -c "select count(*) from timescaledb_information.hypertables" 2>/dev/null || echo -1)
        got_candles=$(drill_psql -d "$DRILL_DB" -c "select count(*) from candles" 2>/dev/null || echo -1)
        got_events=$(drill_psql -d "$DRILL_DB" -c "select count(*) from events" 2>/dev/null || echo -1)
        # Structure must match exactly; rows may only have grown between the
        # meta capture and the dump snapshot, never shrunk.
        [ "$got_tables" = "$exp_tables" ]            || { ok=0; reason="tables: restored $got_tables, expected $exp_tables"; }
        [ "$got_hypertables" = "$exp_hypertables" ]  || { ok=0; reason="$reason hypertables: restored $got_hypertables, expected $exp_hypertables"; }
        [ "$got_candles" -ge "$exp_candles" ] 2>/dev/null || { ok=0; reason="$reason candles: restored $got_candles, expected >= $exp_candles"; }
        [ "$got_events" -ge "$exp_events" ] 2>/dev/null   || { ok=0; reason="$reason events: restored $got_events, expected >= $exp_events"; }
    fi
    drill_cleanup

    if [ "$ok" = 1 ]; then
        cat > "$BACKUP_DIR/drill-latest.json" <<EOF
{
  "ok": true,
  "dump": "$(basename "$dump")",
  "restored_tables": $got_tables,
  "restored_hypertables": $got_hypertables,
  "restored_candles": $got_candles,
  "restored_events": $got_events,
  "finished": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "duration_s": $(( $(date +%s) - t0 ))
}
EOF
        chown "$HOST_OWNER" "$BACKUP_DIR/drill-latest.json" 2>/dev/null
        log "restore drill OK: $(basename "$dump") -> $got_tables tables, $got_hypertables hypertables, $got_candles candles, $got_events events ($(( $(date +%s) - t0 ))s)"
        return 0
    fi
    fail_json "$BACKUP_DIR/drill-latest.json" "$(echo "$reason" | tr '\"' "'")"
    return 1
}

loop() {
    log "scheduler up: nightly backup at $DUMP_TIME, drill on weekday $DRILL_DOW, dir $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    # Before the first catch-up, not inside it — see wait_for_db.
    wait_for_db 120 || true
    last_day_file=/tmp/qd-backup-last-day
    while :; do
        today=$(date +%F)
        mirror_host_health
        # Catch-up: no backup yet, or the newest attempt is >26h old (covers
        # first boot and restarts that slept through the window). A *failed*
        # fresh attempt does not retrigger — it stays visibly unhealthy until
        # the next window rather than hammering the database.
        if { [ ! -f "$BACKUP_DIR/latest.json" ] || [ -z "$(find "$BACKUP_DIR/latest.json" -mmin -1560 2>/dev/null)" ]; } \
                && [ "$(cat "$last_day_file" 2>/dev/null)" != "$today" ]; then
            # The marker guard caps this branch at once per day even if the
            # staleness test itself ever misbehaves (e.g. a find without
            # -mmin): degraded to midnight-ish scheduling, never a retry storm.
            log "no fresh backup found - running catch-up backup"
            echo "$today" > "$last_day_file"
            do_backup
            [ ! -f "$BACKUP_DIR/drill-latest.json" ] && do_drill
        elif [ "$(date +%H%M)" = "$DUMP_TIME" ] && [ "$(cat "$last_day_file" 2>/dev/null)" != "$today" ]; then
            echo "$today" > "$last_day_file"
            do_backup && [ "$(date +%u)" = "$DRILL_DOW" ] && do_drill
        fi
        sleep 30
    done
}

case "${1:-loop}" in
    backup) do_backup ;;
    drill)  do_drill ;;
    loop)   loop ;;
    *) echo "usage: backup-loop.sh [backup|drill|loop]" >&2; exit 2 ;;
esac
