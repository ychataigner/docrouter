from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration de l'API, surchargeable par variables d'environnement `DOCROUTER_*`."""

    model_config = SettingsConfigDict(env_prefix="DOCROUTER_", env_file=".env", extra="ignore")

    # "dummy" = tri simulé sans IA (développement du front) ; "dit" requiert l'extra `ml`.
    classifier: Literal["dit", "dummy"] = "dummy"
    model_id: str = "microsoft/dit-base-finetuned-rvlcdip"
    confidence_threshold: float = 0.60
    max_upload_mb: int = 10
    pdf_render_dpi: int = 150

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
