## What this changes

<!-- One or two sentences. What behaviour is different after this? -->

## Why

<!-- The defect or need. If it fixes a bug, what was the failure mode? -->

## What I ran, and what it printed

<!-- Paste the real output, including failures. A green summary over a red
     test is worse than no summary. -->

```
$ pytest -q

```

## Checklist

- [ ] Tests cover the new behaviour, and I ran them
- [ ] Failure paths handled (empty input, dead network, restart mid-operation, duplicate event)
- [ ] Logs would let someone diagnose this at 3 a.m. without the code in front of them
- [ ] Docs and comments match the code
- [ ] `ruff check .` and `mypy ct` pass
- [ ] This does **not** widen the money boundary (no new path to an arming without a human tap)

## Risk

<!-- The strongest objection to merging this, and what would change your mind. -->
