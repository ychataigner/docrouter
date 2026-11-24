from datetime import date

import httpx
import pytest
from docrouter_ui.client import ApiClient, ApiError
from docrouter_ui.filing import file_document, normalized_name
from docrouter_ui.preview import make_thumbnail

CLASSIFY_BODY = {
    "request_id": "abc12345",
    "filename": "scan.png",
    "page_count": 1,
    "classification": {
        "type": "facture",
        "confidence": 0.9,
        "scores": {"facture": 0.9, "note_de_frais": 0.0, "bon_de_livraison": 0.0, "autre": 0.1},
        "destination": "factures",
        "needs_review": False,
        "raw_label": "invoice",
    },
    "model": "dummy",
    "timings_ms": {"preprocess": 1.0, "classify": 2.0},
}


def client_with(handler) -> ApiClient:
    return ApiClient("http://api", transport=httpx.MockTransport(handler))


def test_classify_parses_response():
    client = client_with(lambda request: httpx.Response(200, json=CLASSIFY_BODY))
    result = client.classify("scan.png", b"data", "image/png")
    assert result.classification.destination == "factures"


def test_api_error_detail_is_surfaced():
    client = client_with(
        lambda request: httpx.Response(400, json={"detail": "Format non supporté"})
    )
    with pytest.raises(ApiError, match="Format non supporté"):
        client.classify("x.txt", b"data", "text/plain")


def test_unreachable_api_raises_api_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="injoignable"):
        client_with(handler).health()


def test_normalized_name():
    name = normalized_name("Ma Facture.PDF", "factures", "abc12345", date(2026, 11, 24))
    assert name == "20261124_factures_abc12345.pdf"


def test_file_document_writes_into_folder(tmp_path):
    path = file_document(tmp_path, "a_verifier", "photo.jpg", "id1", b"content")
    assert path.parent == tmp_path / "a_verifier"
    assert path.read_bytes() == b"content"


def test_thumbnails(png_bytes, pdf_bytes):
    assert make_thumbnail(png_bytes) is not None
    assert make_thumbnail(pdf_bytes) is not None
    assert make_thumbnail(b"not an image") is None


def _filed(output_dir, folder="factures", predicted="factures", doc_id="id1"):
    from datetime import datetime

    from docrouter_ui.registry import FiledDocument

    path = file_document(output_dir, folder, "scan.png", doc_id, b"img")
    return FiledDocument(
        id=doc_id,
        original_name="scan.png",
        path=path.relative_to(output_dir).as_posix(),
        folder=folder,
        filed_at=datetime(2026, 11, 24, 10, 0),
        predicted_type="facture",
        predicted_folder=predicted,
        confidence=0.9,
    )


FOLDERS_ALL = ["factures", "notes_de_frais", "bons_de_livraison", "autres", "a_verifier"]


def test_registry_lists_filed_documents(tmp_path):
    from docrouter_ui.registry import Registry

    registry = Registry(tmp_path, FOLDERS_ALL)
    registry.add(_filed(tmp_path))
    [doc] = registry.all()
    assert doc.folder == "factures"
    assert doc.confidence == 0.9
    assert not doc.corrected


def test_registry_detects_manual_correction(tmp_path):
    from docrouter_ui.registry import Registry

    registry = Registry(tmp_path, FOLDERS_ALL)
    registry.add(_filed(tmp_path, folder="notes_de_frais", predicted="factures"))
    assert registry.all()[0].corrected


def test_registry_includes_untracked_files(tmp_path):
    from docrouter_ui.registry import Registry

    (tmp_path / "autres").mkdir()
    (tmp_path / "autres" / "copie_manuelle.pdf").write_bytes(b"%PDF")
    [doc] = Registry(tmp_path, FOLDERS_ALL).all()
    assert doc.folder == "autres"
    assert doc.predicted_type is None


def test_registry_move_relocates_file_and_updates_entry(tmp_path):
    from docrouter_ui.registry import Registry

    registry = Registry(tmp_path, FOLDERS_ALL)
    doc = _filed(tmp_path, folder="a_verifier", predicted="a_verifier")
    registry.add(doc)
    moved = registry.move(doc, "notes_de_frais")

    assert not (tmp_path / doc.path).exists()
    assert (tmp_path / moved.path).read_bytes() == b"img"
    assert moved.path.startswith("notes_de_frais/")
    [listed] = registry.all()
    assert listed.folder == "notes_de_frais"
    assert listed.corrected
