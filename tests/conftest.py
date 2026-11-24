import io

import pytest
from PIL import Image


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 560), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 560), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def pdf_bytes() -> bytes:
    """PDF de 2 pages."""
    pages = [Image.new("RGB", (400, 560), "white") for _ in range(2)]
    buffer = io.BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()
