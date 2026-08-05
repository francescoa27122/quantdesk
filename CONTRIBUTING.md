# Contributing

Thanks for looking. A few things up front so you can decide whether this is
worth your time.

This is a personal system that trades real money. It is published because the
engineering may be useful to read, not because it is looking for feature
contributions. **Bug reports, security findings, and corrections to the
documentation are very welcome. Large unsolicited features probably are not** —
open an issue first and let's talk about it before you write code.

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest -q
```

The test suite runs entirely offline — no database, no exchange, no network.
If a test you wrote needs any of those, it belongs behind a marker and a skip.

```bash
ruff check .                  # lint
mypy ct                       # types (strict on ct/risk and ct/models)
pytest -q                     # 594 tests, ~35s
```

## The bar for a change

This project's own operating rules are stricter than most. In short:

1. **Tests exist for the new behaviour, and you ran them and report the real
   result.** A green summary over a red test is the worst thing you can submit.
2. **Failure paths are handled** — empty input, dead network, restart
   mid-operation, duplicate event. "It works" is not the same as "it does not
   corrupt state when it doesn't work."
3. **It logs enough that a 3 a.m. failure is diagnosable from the logs alone.**
4. **Comments explain why, not what.** The convention in this codebase is that
   a comment records the defect that motivated the code. Read
   `ct/engine/execution.py` or `ct/store/arming.py` for the house style — that
   is what a good comment looks like here.
5. **Docs match the code.** A doc that disagrees with the code is a defect.

## Things that will be declined

- **Anything that widens the money boundary.** Adding a code path that can
  create an arming whose ancestry does not terminate in a human tap. This is
  not negotiable and is the single invariant the whole design exists to
  protect. See ARCHITECTURE.md §9.
- **New dependencies**, unless the alternative is genuinely worse. The
  dependency list is short on purpose — every entry is a thing that can break
  the trading engine at 3 a.m., and the target hardware is a 4-core N100 with
  about 2 GB to spend. `pyproject.toml` records why `scipy`, `TA-Lib` and
  `pandas`-in-the-hot-path are absent.
- **Strategy contributions with backtest results attached but no walk-forward.**
  In-sample results are not evidence. The promotion gate (ARCHITECTURE.md §8.3)
  is the standard.
- **Moving ML dependencies into `qd-engine`.** They live in `qd-worker`
  precisely so an ML dependency cannot break execution.

## Pull requests

- One logical change per PR. Ties break toward the smaller change.
- Say what you ran and what it printed, including failures.
- If you found a real problem with the approach, say so in a sentence — that is
  inside the contract, not outside it.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
