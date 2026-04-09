# ── builder ──────────────────────────────────────────────────────────────────
FROM cgr.dev/chainguard/python:latest-dev AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# ── runtime ───────────────────────────────────────────────────────────────────
FROM cgr.dev/chainguard/python:latest-dev AS runtime

USER root

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apk add --no-cache ffmpeg mesa-gl exiftool

WORKDIR /app

COPY --from=builder --chown=nonroot:nonroot /app/.venv /app/.venv

COPY --chown=nonroot:nonroot src/ ./src/
COPY --chown=nonroot:nonroot scripts/ ./scripts/
COPY --chown=nonroot:nonroot eval/ ./eval/
COPY --chown=nonroot:nonroot data/ ./data/
COPY --chown=nonroot:nonroot ebird_data/ ./ebird_data/

USER nonroot
RUN /usr/local/bin/uv run scripts/import_ebird_barchart.py ebird_data/ --db data/ebird_priors.db
USER root

RUN mkdir -p /data/results /data/report && \
    chown -R nonroot:nonroot /data

VOLUME ["/data/videos/assets", "/data/results", "/data/report", "/data/models"]

USER nonroot

ENV HF_HOME=/data/models \
    TORCH_HOME=/data/models \
    ULTRALYTICS_DIR=/data/models \
    PYTHONUNBUFFERED=1

ENTRYPOINT []

# Intended run order (see eval/docker-compose.yml):
#   1. prefetch service  — download/verify all configured models
#   2. eval service      — run inference, write sidecars  (this CMD)
#   3. report service    — generate report.html from sidecars
CMD ["/usr/local/bin/uv", "run", "eval/eval_runner.py", "--config", "eval/config.yaml"]
