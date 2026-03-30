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

# ffmpeg is needed by opencv for video decoding
RUN apk add --no-cache ffmpeg mesa-gl exiftool

WORKDIR /app

# Copy venv separately so it stays cached across code changes
COPY --from=builder --chown=nonroot:nonroot /app/.venv /app/.venv

# Source files — changes here only re-export this small layer
COPY --chown=nonroot:nonroot src/ ./src/
COPY --chown=nonroot:nonroot scripts/ ./scripts/
COPY --chown=nonroot:nonroot templates/ ./templates/
COPY --chown=nonroot:nonroot data/ ./data/

# Writable directories for input videos, results, and model cache
RUN mkdir -p /data/videos /data/results /data/models && \
    chown -R nonroot:nonroot /data

VOLUME ["/data/videos", "/data/results", "/data/models"]

USER nonroot

ENV HF_HOME=/data/models \
    TORCH_HOME=/data/models \
    ULTRALYTICS_DIR=/data/models \
    PYTHONUNBUFFERED=1

ENTRYPOINT []
CMD ["/usr/local/bin/uv", "run", "scripts/serve.py"]
