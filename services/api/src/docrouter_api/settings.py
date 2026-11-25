from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
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
    # Réseau de la sonde exporté en ONNX (onnxruntime, sans PyTorch). Vide → PyTorch.
    onnx_model: str = "models/onnx/clip/model_int8.onnx"
    # Threads d'onnxruntime par inférence (0 = auto). À aligner sur les CPU du conteneur.
    onnx_threads: int = 0
    # Inférences exécutées en même temps (itération 5). onnxruntime utilise déjà tous les CPU
    # du conteneur pour une inférence : au-delà de 1, elles se disputent les cœurs et le débit
    # baisse sous forte charge. Les requêtes suivantes attendent leur tour.
    max_concurrent_inferences: int = 1
    confidence_threshold: float = 0.60
    # Clés d'API autorisées, une par client : "front:<clé>,compta:<clé>". Vide → /v1/* refusé.
    api_keys: SecretStr = SecretStr("")
    max_upload_mb: int = 10
    pdf_render_dpi: int = 150

    @property
    def expected_model(self) -> str:
        """Modèle attendu (affiché par /health même s'il n'a pas pu être chargé)."""
        probe = f"{self.probe_head} + {self.onnx_model or 'pytorch'}"
        return {"probe": probe, "dit": self.model_id}.get(self.classifier, "dummy")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
