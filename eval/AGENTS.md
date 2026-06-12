# BirdVision — Model-Comparison Eval (`eval/`)

Standalone container that runs the existing asset library through multiple
classifier backends and renders a self-contained HTML report for visual
comparison. Cross-cutting rules live in the root `AGENTS.md`.

`eval_runner.py` writes one "sidecar" JSON per (clip, model);
`report_generator.py` renders them to `report/report.html`, copying crops
into `report/crops/` with relative paths so the report directory is
portable. BioCLIP scores are not recomputed — they are read from existing
results JSONs, so a BioCLIP-only pass needs no GPU or model downloads.

## Commands

```bash
# Local (no Docker; repo-relative paths)
uv run eval/eval_runner.py --config eval/config-local.yaml [--max-clips 20]
uv run eval/report_generator.py --config eval/config-local.yaml
# → open eval/report/report.html

# Docker (run in order; container paths via eval/config.yaml)
docker compose -f eval/docker-compose.yml run --rm prefetch          # models
docker compose -f eval/docker-compose.yml up eval                    # GPU
docker compose -f eval/docker-compose.yml --profile report up report # no GPU
```

## Conventions

- Two configs with the same schema: `eval/config.yaml` (container paths,
  `/data/...`) and `eval/config-local.yaml` (repo-relative). Keep in sync.
- Sidecars land in `<report_dir>/eval/<asset_sha>_<model_id>.json` with
  `model_id`, `model_label`, `asset_sha`, `source_results_file`, and
  `tracks[]` (`track_id`, `crop_file`, `top_species`).
- Adding a model: add/enable an entry under `models:` in both configs, plus
  a backend in `eval_runner.py` if it is a new backend type.
- `implausible_at_feeder:` lists species the report flags when ranked #1 at
  a feeder location.
- The report regenerates from sidecars alone — re-run the `report` service
  (no GPU) without re-running inference.

## Guardrails

- `eval/report/` is generated output and gitignored — never commit it.
- The container runs as uid 65532 (nonroot). If `eval/report/` was first
  created by a local run, `chmod 777 eval/report` before the Docker `eval`
  step, or it cannot write.
- `eval/Dockerfile` builds with the repo root as context and installs the
  root `pyproject.toml`/`uv.lock` — root dependency changes affect this
  image.
- `prefetch` deliberately skips BioCLIP and YOLO (the current flow reuses
  existing results JSONs) — see its `command:` in `eval/docker-compose.yml`.

## Agent Notes

Symlinked as `CLAUDE.md` and `GEMINI.md`; keep instructions tool-neutral.
