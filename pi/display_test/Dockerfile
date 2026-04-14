# Display pipeline test — no Hailo, no ML.
# Captures from /dev/video0, writes scaled+captioned frames to /dev/fb0.
# Architecture: linux/arm64

# ── builder ───────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.13

WORKDIR /app

COPY requirements.txt ./
RUN uv venv --python 3.13 .venv && \
    VIRTUAL_ENV=/app/.venv uv pip install --no-cache -r requirements.txt

# ── runtime ───────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /app

COPY --from=builder /opt/uv-python /opt/uv-python
COPY --from=builder /app/.venv /app/.venv

COPY test_display.py ./

RUN mkdir -p /data && \
    useradd -m -u 65532 -s /sbin/nologin nonroot && \
    chown -R nonroot:nonroot /app /data

USER nonroot

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

CMD ["python", "test_display.py"]
