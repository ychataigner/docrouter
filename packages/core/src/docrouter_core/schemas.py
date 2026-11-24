"""Contrat d'échange entre l'API et ses clients (spec §8)."""

from pydantic import BaseModel, Field

from docrouter_core.documents import DocumentType


class ClassificationResult(BaseModel):
    type: DocumentType
    confidence: float = Field(ge=0, le=1)
    scores: dict[DocumentType, float]
    destination: str
    needs_review: bool
    raw_label: str | None = Field(
        default=None, description="Label brut du modèle avant regroupement en 4 classes"
    )


class ClassifyResponse(BaseModel):
    request_id: str
    filename: str
    page_count: int
    classification: ClassificationResult
    model: str
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    model_ready: bool
    confidence_threshold: float
    max_upload_mb: int


class ErrorResponse(BaseModel):
    detail: str
