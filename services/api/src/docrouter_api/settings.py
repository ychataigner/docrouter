from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration de l'API, surchargeable par variables d'environnement `DOCROUTER_*`."""

    model_config = SettingsConfigDict(env_prefix="DOCROUTER_", env_file=".env", extra="ignore")

    # "dummy" = tri simulé sans IA (développement du front).
    # "probe" (sonde CLIP, recommandé) et "dit" requièrent l'extra `ml`.
    classifier: Literal["probe", "dit", "dummy"] = "dummy"
    model_id: str = "microsoft/dit-base-finetuned-rvlcdip"  # utilisé par "dit"
    # Tête de la sonde : fichier local ou hf://<compte>/<dépôt>/<fichier>.npz
    probe_head: str = "models/probes/probe-clip.npz"
    confidence_threshold: float = 0.60
    max_upload_mb: int = 10
    pdf_render_dpi: int = 150

    @property
    def expected_model(self) -> str:
        """Modèle attendu (affiché par /health même s'il n'a pas pu être chargé)."""
        return {"probe": self.probe_head, "dit": self.model_id}.get(self.classifier, "dummy")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
