from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class UISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCROUTER_", env_file=".env", extra="ignore")

    api_url: str = "http://localhost:8000"
    api_timeout_s: float = 60.0
    output_dir: Path = Path("output")


@lru_cache
def get_settings() -> UISettings:
    return UISettings()
