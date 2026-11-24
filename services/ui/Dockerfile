# Image du front — construite depuis la racine du dépôt : docker build -f services/ui/Dockerfile .

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/
COPY services/api/pyproject.toml services/api/
COPY services/ui/pyproject.toml services/ui/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package docrouter-ui --no-install-workspace

COPY packages/core packages/core
COPY services/ui services/ui
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package docrouter-ui --no-editable


FROM python:3.11-slim
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/output && chown -R app:app /data
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DOCROUTER_OUTPUT_DIR=/data/output \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_RUN_ON_SAVE=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER app
WORKDIR /app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["docrouter-ui"]
