# ── builder ──────────────────────────────────────────────────────────────────
FROM cgr.dev/chainguard/python:latest-dev AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies into a prefix uv can copy wholesale
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy source and install the project itself
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY templates/ ./templates/
COPY data/ ./data/
RUN uv sync --frozen

# ── runtime ───────────────────────────────────────────────────────────────────
FROM cgr.dev/chainguard/python:latest-dev AS runtime

USER root

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ffmpeg is needed by opencv for video decoding
RUN apk add --no-cache ffmpeg mesa-gl

WORKDIR /app

COPY --from=builder --chown=nonroot:nonroot /app /app

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
