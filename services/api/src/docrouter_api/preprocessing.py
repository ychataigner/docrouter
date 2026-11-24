"""Chargement des fichiers reçus en image exploitable par le modèle, sans rien écrire sur disque."""

import io
from dataclasses import dataclass

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

PDF_MAGIC = b"%PDF"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


class UnsupportedDocumentError(ValueError):
    """Fichier dans un format non pris en charge ou illisible."""


@dataclass
class LoadedDocument:
    image: Image.Image
    page_count: int


def detect_format(data: bytes) -> str:
    """Détecte le format à partir du contenu (on ne fait pas confiance à l'extension)."""
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    raise UnsupportedDocumentError(
        "Format non supporté : seuls les fichiers PDF, PNG et JPEG sont acceptés."
    )


def load_document(data: bytes, pdf_dpi: int = 150) -> LoadedDocument:
    """Renvoie la 1re page du document en RVB, prête pour la classification."""
    if not data:
        raise UnsupportedDocumentError("Fichier vide.")

    if detect_format(data) == "pdf":
        image, page_count = _render_pdf_first_page(data, pdf_dpi)
    else:
        image, page_count = _open_image(data), 1

    return LoadedDocument(image=_normalize(image), page_count=page_count)


def _render_pdf_first_page(data: bytes, dpi: int) -> tuple[Image.Image, int]:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise UnsupportedDocumentError("PDF corrompu ou illisible.") from exc
    try:
        if len(pdf) == 0:
            raise UnsupportedDocumentError("Le PDF ne contient aucune page.")
        image = pdf[0].render(scale=dpi / 72).to_pil()
        return image, len(pdf)
    finally:
        pdf.close()


def _open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedDocumentError("Image corrompue ou illisible.") from exc
    # Les photos de smartphone stockent souvent leur orientation dans l'EXIF.
    return ImageOps.exif_transpose(image)


def _normalize(image: Image.Image) -> Image.Image:
    # RVL-CDIP (données d'entraînement de DiT) est en niveaux de gris : on s'en rapproche.
    return image.convert("L").convert("RGB")
