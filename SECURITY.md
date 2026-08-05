# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Use GitHub's private reporting: **Security → Report a vulnerability** on this
repository. That opens a channel visible only to the maintainers.

Please include:

- what an attacker can do, not just what is wrong
- the smallest reproduction you have — a request, a payload, a config
- the commit you tested against

You will get an acknowledgement within 7 days. This is a personal project
maintained by one person, so please assume good faith on timelines rather than
an SLA.

## Scope

This repository trades real money when armed. Findings in these areas are the
most valuable, in roughly this order:

1. **Anything that lets code place, modify, or cancel an order without a
   human-originated arming.** The arming chain is the core invariant — see
   ARCHITECTURE.md §9. Every renewal must walk back to a human tap.
2. **Anything that exposes exchange credentials**, including through logs,
   error responses, the dashboard, or a container that should not hold them.
   `qd-web` and `qd-collector` must never be able to reach an exchange with
   trading credentials.
3. **Authentication and session handling** in `qd-web` (`ct/web/security.py`).
4. **Injection of any kind** into the database, the templates, or a shell.
5. **Risk-gate bypasses** — anything that lets a position exceed its configured
   envelope.

## Known and accepted limitations

These are deliberate, documented, and not vulnerabilities in this threat model.
Reporting them is fine; you will just get this section back.

- **No TOTP / second factor.** Reasoned through in `ct/web/security.py`. A
  second factor without enrolment, recovery codes and a re-key path locks you
  out at the worst possible moment. The intended deployment is Tailscale-only.
- **`halt` is available to every authenticated role, including `viewer`.**
  Actions that reduce risk are never gated. This is load-bearing, not an
  oversight.
- **Sessions are signed cookies, not a server-side store.** One operator, one
  container; a session table would add a write path to a read-mostly service.
  The consequence — you cannot revoke one session without rotating the session
  secret — is accepted.
- **The login rate limiter is in-process.** A restart forgives outstanding
  lockouts. An attacker cannot trigger a restart.

## Deployment expectations

The security posture assumes all of the following. If you deploy differently,
you own the difference:

- `qd-web` is reachable only over a private network (Tailscale or equivalent).
  It is **not** hardened for the open internet.
- If you put a reverse proxy in front of it, set `QD_TRUSTED_PROXIES` to that
  proxy's address. Otherwise `X-Forwarded-For` is ignored — which is the safe
  default, but means every request shares one rate-limit bucket.
- Exchange API keys are created **withdrawal-disabled and IP-allowlisted**.
  That single setting caps a total host compromise at slippage rather than
  balance. Do not skip it.
- Secrets are supplied as Docker secret *files*, never as environment
  variables in a compose file you might commit.

## Supported versions

The `main` branch only. This project has not cut a release.
