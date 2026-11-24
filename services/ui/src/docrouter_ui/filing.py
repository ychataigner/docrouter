"""Rangement des documents dans les dossiers de destination (côté client, spec §5)."""

from datetime import date
from pathlib import Path


def normalized_name(original: str, folder: str, request_id: str, day: date | None = None) -> str:
    """Nom normalisé `AAAAMMJJ_<dossier>_<id>.<ext>` (spec §6.1)."""
    day = day or date.today()
    ext = Path(original).suffix.lower() or ".bin"
    return f"{day:%Y%m%d}_{folder}_{request_id}{ext}"


def file_document(
    output_dir: Path, folder: str, original_name: str, request_id: str, data: bytes
) -> Path:
    target_dir = output_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / normalized_name(original_name, folder, request_id)
    target.write_bytes(data)
    return target
