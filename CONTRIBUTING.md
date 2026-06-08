# Contributing to BirdVision

Thanks for your interest. This guide covers the practical bits — environment
setup, the test/lint expectations, and how PRs flow.

## Setup

```bash
git clone https://github.com/evanwtf/birdvision.git
cd birdvision
uv sync
cp config.yaml.example config.yaml   # edit as needed
```

`uv` manages the virtualenv and dependencies; `uv.lock` is committed and
authoritative. Run anything via `uv run` (e.g. `uv run pytest`, `uv run
scripts/serve.py --debug`). Don't install into a global Python.

Python 3.12 minimum. No GPU is required for the test suite — heavy
dependencies (`ultralytics`, `open-clip-torch`, `transformers`, `hailort`)
are monkeypatched in `tests/conftest.py`.

## Tests

```bash
uv run pytest            # full suite, ~3s on a modern laptop
uv run pytest tests/test_tracker.py::TestName::test_name   # single test
```

The coverage gate is enforced by `pytest-cov` — runs fail if line coverage
under `src/` drops below the current floor (see `[tool.pytest.ini_options]`
in `pyproject.toml`). Add tests for new modules; don't lower the floor to
land a feature.

## Lint and format

```bash
uv run ruff check        # lint
uv run ruff format       # apply formatting
uv run ruff format --check   # check without writing
```

CI runs both. Two rule categories (`E501`, `B023`) are ignored project-wide
pending follow-up cleanup; everything else is enforced.

## Pi-only modules

The Raspberry Pi pipeline lives in `src/hailo_*.py`, `stream_capture.py`,
`ws_frame_source.py`, `file_frame_source.py`, `display_overlay.py`, and
`realtime_pipeline.py`. Those modules may import from the desktop side,
but **the webapp / desktop pipeline must not import them** — `hailort`
is aarch64-only and would break CI on x86_64.

`hailort` is intentionally NOT in `[project.dependencies]`. Don't add it.
See `pi/README.md` for the manual install path on a real Pi.

## Branches and PRs

- Branch off `main`, name with a short kebab-case descriptor
  (`fix/some-bug`, `feat/new-thing`, `chore/whatever`).
- One logical change per PR. Format-only commits go in their own commit
  (and ideally their own PR) so the substantive diff stays readable.
- PRs are required for `main`; tests + lint must pass.
- Use the existing labels (`area: *`, `priority: *`) when known.

## What kinds of changes are welcome

- Bug fixes with a regression test.
- New detector/classifier backends following the existing module pattern.
- Better priors logic, eBird coverage for additional regions, or smarter
  GPS gating beyond Long Island.
- Pi pipeline improvements (latency, accuracy, additional capture
  sources).
- Docs improvements.

## What's out of scope

- Cloud inference. Everything runs locally on purpose.
- Adding a database for state. JSON sidecars + the asset index are
  intentional.
- Rewriting the asset store, OAuth, or tracker without a concrete reason
  — they're stable and tested.

## Reporting issues

Bugs and feature requests go on the GitHub issue tracker. Security issues
should go to the address listed in `SECURITY.md` (once added).
