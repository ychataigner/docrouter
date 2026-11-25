import asyncio
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from docrouter_core import (
    ClassificationResult,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
    destination_for,
)
from fastapi import FastAPI, File, HTTPException, Request, Security, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from PIL import Image

from docrouter_api import __version__
from docrouter_api.classifier import DocumentClassifier, build_classifier
from docrouter_api.preprocessing import UnsupportedDocumentError, load_document
from docrouter_api.security import (
    API_KEY_HEADER,
    MULTIPART_OVERHEAD,
    ApiKeys,
    keep_uploads_in_memory,
)
from docrouter_api.settings import Settings, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("docrouter.api")

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Format non supporté ou fichier illisible"},
    401: {"model": ErrorResponse, "description": "Clé d'API manquante ou invalide"},
    413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
    503: {
        "model": ErrorResponse,
        "description": "Modèle non chargé ou authentification non configurée",
    },
}

# Déclare la clé dans OpenAPI : bouton « Authorize » de Swagger. Le contrôle lui-même est fait
# par le middleware `guard`, avant la lecture du document.
api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER, auto_error=False, description="Clé d'API du client"
)


def create_app(
    settings: Settings | None = None, classifier: DocumentClassifier | None = None
) -> FastAPI:
    settings = settings or get_settings()
    api_keys = ApiKeys.parse(settings.api_keys.get_secret_value())
    if not api_keys:
        logger.warning("Aucune clé d'API (DOCROUTER_API_KEYS) : les routes /v1/* sont fermées.")
    keep_uploads_in_memory(settings.max_upload_bytes)
    inference_slots = asyncio.Semaphore(settings.max_concurrent_inferences)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier
        if app.state.classifier is None:
            try:
                app.state.classifier = build_classifier(
                    settings.classifier,
                    settings.model_id,
                    settings.probe_head,
                    settings.onnx_model,
                    settings.onnx_threads,
                )
                _warm_up(app.state.classifier, settings.pdf_render_dpi)
            except Exception:
                # L'API démarre quand même : /health signale le problème (pas de crash en boucle).
                logger.exception("Impossible de charger le modèle (%s)", settings.classifier)
        yield

    app = FastAPI(
        title="DocRouter API",
        version=__version__,
        description="Classification de documents administratifs : facture, note de frais, "
        "bon de livraison, autre. Aucun fichier n'est conservé par l'API.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Clé d'API et taille vérifiées avant de lire le document (itération 6)."""
        if request.url.path.startswith("/v1/"):
            if not api_keys:
                return _error(503, "Authentification non configurée (DOCROUTER_API_KEYS).")
            client = api_keys.identify(request.headers.get(API_KEY_HEADER))
            if client is None:
                return _error(
                    401, "Clé d'API manquante ou invalide.", {"WWW-Authenticate": "ApiKey"}
                )
            length = request.headers.get("content-length")
            if request.method == "POST" and length is None:
                return _error(411, "En-tête Content-Length requis.")
            if length is not None and int(length) > settings.max_upload_bytes + MULTIPART_OVERHEAD:
                return _error(413, f"Fichier trop volumineux (max {settings.max_upload_mb} Mo).")
            request.state.client = client
        return await call_next(request)

    def get_classifier(request: Request) -> DocumentClassifier:
        clf = request.app.state.classifier
        if clf is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé.")
        return clf

    @app.get("/health", response_model=HealthResponse, tags=["système"])
    def health(request: Request) -> HealthResponse:
        clf = request.app.state.classifier
        return HealthResponse(
            status="ok" if clf else "degraded",
            version=__version__,
            model=clf.name if clf else settings.expected_model,
            model_ready=clf is not None,
            confidence_threshold=settings.confidence_threshold,
            max_upload_mb=settings.max_upload_mb,
        )

    @app.post(
        "/v1/classify",
        response_model=ClassifyResponse,
        responses=ERROR_RESPONSES,
        tags=["classification"],
        summary="Classer un document (PDF, PNG, JPEG)",
    )
    async def classify(
        request: Request,
        file: Annotated[UploadFile, File()],
        _key: Annotated[str | None, Security(api_key_scheme)] = None,
    ) -> ClassifyResponse:
        clf = get_classifier(request)
        request_id = uuid.uuid4().hex[:8]
        timings: dict[str, float] = {}

        data = await file.read(settings.max_upload_bytes + 1)
        await file.close()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"Fichier trop volumineux (max {settings.max_upload_mb} Mo).",
            )

        start = time.perf_counter()
        try:
            document = await run_in_threadpool(load_document, data, settings.pdf_render_dpi)
        except UnsupportedDocumentError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        finally:
            del data
        timings["preprocess"] = _elapsed_ms(start)

        start = time.perf_counter()
        async with inference_slots:  # file d'attente : le débit reste stable sous la charge
            timings["queue"] = _elapsed_ms(start)
            prediction = await run_in_threadpool(clf.predict, document.image)
        timings["classify"] = _elapsed_ms(start) - timings["queue"]

        doc_type, confidence = prediction.best
        destination = destination_for(doc_type, confidence, settings.confidence_threshold)
        # On ne journalise que des métadonnées techniques, jamais le contenu (RGPD).
        logger.info(
            "request_id=%s client=%s type=%s confidence=%.3f pages=%d timings=%s",
            request_id,
            request.state.client,
            doc_type,
            confidence,
            document.page_count,
            timings,
        )
        return ClassifyResponse(
            request_id=request_id,
            filename=file.filename or "document",
            page_count=document.page_count,
            classification=ClassificationResult(
                type=doc_type,
                confidence=round(confidence, 4),
                scores={t: round(s, 4) for t, s in prediction.scores.items()},
                destination=destination,
                needs_review=confidence < settings.confidence_threshold,
                raw_label=prediction.raw_label,
            ),
            model=clf.name,
            timings_ms=timings,
        )

    return app


def _warm_up(classifier: DocumentClassifier, pdf_dpi: int) -> None:
    """La 1re inférence et le 1er décodage JPEG / PDF sont lents (chargement des bibliothèques,
    jusqu'à 10 s) : on les paie au démarrage plutôt que sur le premier document envoyé."""
    for fmt in ("JPEG", "PDF"):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(buffer, format=fmt)
        classifier.predict(load_document(buffer.getvalue(), pdf_dpi).image)


def _error(code: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=code, headers=headers)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


app = create_app()
