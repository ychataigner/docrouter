import io

import pypdfium2 as pdfium
from PIL import Image, ImageOps

THUMBNAIL_SIZE = (360, 480)


def make_thumbnail(data: bytes) -> Image.Image | None:
    """Miniature de la 1re page (PDF ou image). `None` si le fichier est illisible."""
    try:
        if data.startswith(b"%PDF"):
            pdf = pdfium.PdfDocument(data)
            try:
                image = pdf[0].render(scale=60 / 72).to_pil()
            finally:
                pdf.close()
        else:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        image.thumbnail(THUMBNAIL_SIZE)
        return image.convert("RGB")
    except Exception:
        return None
