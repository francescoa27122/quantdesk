#!/bin/sh
# Reset the QuantDesk dashboard admin password (login user: admin).
#
# Run it from the deploy directory, or by its own path — it locates itself:
#
#     ./reset-password.sh
#
# Prompts twice for the new password (minimum 12 characters; a short phrase
# like "purple fish dinner" passes and is easy to type), rewrites
# QD_ADMIN_PASSWORD_HASH in .env, backs the old .env up, and recreates qd-web.
#
# Why this wrapper exists at all: an argon2id hash is full of `$`, and docker
# compose interpolates `$` in .env — quoting does not stop it. `qd
# hash-password` prints a ready-to-paste line with every `$` doubled, and this
# script writes that line for you so the escaping is never done by hand. Get it
# wrong and every login fails 401 with correct credentials, which reads like a
# forgotten password rather than a mangled file.
#
# The alternative, if you prefer it, is the secret-file route: write the raw
# hash to secrets/admin_password_hash and let QD_ADMIN_PASSWORD_HASH_FILE pick
# it up. No escaping involved. See README "Set the admin password hash".
#
# Why the temp-dir dance: the qd-web image runs as uid 10001, which some NAS
# ACL schemes keep out of the project root — so the container writes its JSON
# into a throwaway mode-777 subdir that this script creates and removes.
set -eu

# Locate the deploy directory: the one holding this script. Beats a hardcoded
# path, and means the script works wherever the stack is checked out.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

[ -f docker-compose.yml ] || {
    echo "no docker-compose.yml beside this script - run it from the deploy directory" >&2
    exit 1
}
[ -f .env ] || {
    echo "no .env beside this script - copy .env.example first" >&2
    exit 1
}

TMPDIR=.pwtmp
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"
chmod 777 "$TMPDIR"

echo "Choose a new QuantDesk admin password - minimum 12 characters."
echo "A short phrase works well and is easy to type."
echo
if ! docker compose run --rm --no-deps \
      -v "$PWD/$TMPDIR:/hostout" qd-web \
      sh -c 'qd hash-password --json > /hostout/hash.json'; then
    rm -rf "$TMPDIR"
    echo >&2
    echo "Password refused (too short, or the two entries did not match)." >&2
    echo "Nothing was changed - run me again." >&2
    exit 1
fi

LINE=$(tr -d '\r\n' < "$TMPDIR/hash.json" | sed 's/.*"env_line": *"\([^"]*\)".*/\1/')
rm -rf "$TMPDIR"
case "$LINE" in
    QD_ADMIN_PASSWORD_HASH=\$\$argon2*) ;;
    *) echo "Could not extract the hash line - nothing was changed." >&2; exit 1 ;;
esac

cp .env ".env.bak-$(date +%Y%m%d-%H%M%S)"
grep -v '^QD_ADMIN_PASSWORD_HASH=' .env > .env.new
printf '%s\n' "$LINE" >> .env.new
chmod 600 .env.new
mv .env.new .env

# `docker restart` would NOT reload .env; the container must be recreated.
docker compose up -d qd-web

PORT=$(sed -n 's/.*"\([0-9]\{4,5\}\):8090".*/\1/p' docker-compose.yml | head -1)
echo
echo "Done. Log in as 'admin' with your new password on port ${PORT:-8090}."
echo "The previous .env is kept as .env.bak-*"
