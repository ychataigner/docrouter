"""Registre des documents rangés : garde la trace de leur tri (côté client, l'API ne stocke rien).

Une ligne JSON par document dans `<output_dir>/registre.jsonl`. Les fichiers présents dans les
dossiers sans entrée au registre (copiés à la main, par exemple) sont aussi listés.
"""

from datetime import datetime
from pathlib import Path

from docrouter_core import DocumentType
from pydantic import BaseModel

from docrouter_ui.filing import normalized_name

REGISTRY_FILENAME = "registre.jsonl"


class FiledDocument(BaseModel):
    id: str
    original_name: str
    path: str  # relatif au dossier de sortie, séparateurs "/"
    folder: str
    filed_at: datetime
    predicted_type: DocumentType | None = None
    predicted_folder: str | None = None
    confidence: float | None = None
    needs_review: bool = False
    model: str | None = None
    page_count: int | None = None

    @property
    def corrected(self) -> bool:
        """Rangé ailleurs que là où le modèle l'avait proposé."""
        return self.predicted_folder is not None and self.folder != self.predicted_folder


class Registry:
    def __init__(self, output_dir: Path, folders: list[str]) -> None:
        self.output_dir = output_dir
        self.folders = folders
        self._file = output_dir / REGISTRY_FILENAME

    def add(self, doc: FiledDocument) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self._file.open("a", encoding="utf-8") as f:
            f.write(doc.model_dump_json() + "\n")

    def all(self) -> list[FiledDocument]:
        """Documents réellement présents sur disque, du plus récent au plus ancien."""
        known = {doc.path: doc for doc in self._read() if (self.output_dir / doc.path).is_file()}
        for folder in self.folders:
            folder_dir = self.output_dir / folder
            if not folder_dir.is_dir():
                continue
            for file in folder_dir.iterdir():
                rel = f"{folder}/{file.name}"
                if file.is_file() and rel not in known:
                    known[rel] = FiledDocument(
                        id=file.stem,
                        original_name=file.name,
                        path=rel,
                        folder=folder,
                        filed_at=datetime.fromtimestamp(file.stat().st_mtime),
                    )
        return sorted(known.values(), key=lambda d: d.filed_at, reverse=True)

    def move(self, doc: FiledDocument, folder: str) -> FiledDocument:
        """Reclasse un document : déplace le fichier et met à jour le registre."""
        if folder == doc.folder:
            return doc
        new_name = normalized_name(doc.original_name, folder, doc.id, doc.filed_at.date())
        target = self.output_dir / folder / new_name
        target.parent.mkdir(parents=True, exist_ok=True)
        (self.output_dir / doc.path).replace(target)

        moved = doc.model_copy(update={"folder": folder, "path": f"{folder}/{new_name}"})
        entries = [e for e in self._read() if e.path != doc.path] + [moved]
        self._write(entries)
        return moved

    def _read(self) -> list[FiledDocument]:
        if not self._file.is_file():
            return []
        lines = self._file.read_text(encoding="utf-8").splitlines()
        return [FiledDocument.model_validate_json(line) for line in lines if line.strip()]

    def _write(self, entries: list[FiledDocument]) -> None:
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text("".join(e.model_dump_json() + "\n" for e in entries), encoding="utf-8")
        tmp.replace(self._file)
