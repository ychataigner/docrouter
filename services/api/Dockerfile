# Image de l'API — construite depuis la racine du dépôt : docker build -f services/api/Dockerfile .
# Par défaut, image légère sans IA (tri simulé). Avec le modèle :
#   --build-arg ML_EXTRA=onnx  production : onnxruntime, sans PyTorch
#   --build-arg ML_EXTRA=ml    PyTorch (modèle lourd de référence, comparaisons)

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
ARG ML_EXTRA=

# 1) Dépendances seules (couche mise en cache tant que uv.lock ne change pas)
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/
COPY services/api/pyproject.toml services/api/
COPY services/ui/pyproject.toml services/ui/
COPY training/pyproject.toml training/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package docrouter-api --no-install-workspace \
    $([ -n "$ML_EXTRA" ] && echo "--extra $ML_EXTRA")

# 2) Code du projet
COPY packages/core packages/core
COPY services/api services/api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package docrouter-api --no-editable \
    $([ -n "$ML_EXTRA" ] && echo "--extra $ML_EXTRA")


FROM python:3.11-slim
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models && chown app:app /models
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models
USER app
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "docrouter_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
