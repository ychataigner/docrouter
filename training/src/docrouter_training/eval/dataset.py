"""Lecture du jeu généré (`manifest.csv`) et chargement des images comme le fait l'API."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from docrouter_api.preprocessing import load_document
from docrouter_core import DocumentType
from PIL import Image

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    path: Path
    split: str
    label: DocumentType
    layout: int
    profile: str

    @property
    def fmt(self) -> str:
        return "pdf" if self.path.suffix.lower() == ".pdf" else "image"


def read_manifest(root: Path, split: str | None = None) -> list[Sample]:
    """Échantillons du manifest, dans l'ordre du fichier (éventuellement filtrés par split)."""
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return [
        Sample(
            path=root / row["fichier"],
            split=row["split"],
            label=DocumentType(row["classe"]),
            layout=int(row["gabarit"]),
            profile=row["profil"],
        )
        for row in rows
        if split is None or row["split"] == split
    ]


def dataset_fingerprint(root: Path) -> dict[str, str]:
    """Identité du jeu de données, enregistrée avec chaque run MLflow."""
    info_path = root / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return {
            "dataset_sha256": info["manifest_sha256"],
            "dataset_seed": str(info["seed"]),
            "generator_version": info["generator_version"],
        }
    digest = hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest()
    return {"dataset_sha256": digest}


def load_image(path: Path) -> Image.Image:
    """Même chargement que l'API : PDF → 1re page, orientation EXIF, niveaux de gris en RVB."""
    return load_document(path.read_bytes()).image
