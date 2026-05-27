# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:python3.12-bookworm-slim /usr/local/bin/uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    UV_NO_CACHE=1 \
    # Keep the managed Python inside /app so one COPY brings everything to runtime
    UV_PYTHON_INSTALL_DIR=/app/.python \
    UV_PYTHON_DOWNLOADS=automatic \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./

# README.md stub: required by build backend but content is irrelevant
RUN echo "" > README.md && uv sync --frozen --no-dev --no-editable --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu20.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Both the managed Python interpreter and the venv live under /app
COPY --from=builder /app/.python /app/.python
COPY --from=builder /app/.venv   /app/.venv
COPY --from=builder /app/src     /app/src
COPY onnx_model/ /app/onnx_model/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Defaults — override via env or .env file
ENV GENDER_HOST=0.0.0.0
ENV GENDER_PORT=8000
ENV GENDER_LOG_DIR=/app/logs
ENV GENDER_ONNX_DIR=/app/onnx_model
ENV GENDER_WORKERS=2
ENV GENDER_PRECISION=fp16

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["gender-prediction"]
