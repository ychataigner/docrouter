import pytest
from docrouter_api.classifier import DummyClassifier, aggregate_scores
from docrouter_api.main import create_app
from docrouter_api.settings import Settings
from docrouter_core import DocumentType
from fastapi.testclient import TestClient

CONFIDENT_INVOICE = {
    DocumentType.FACTURE: 0.85,
    DocumentType.NOTE_DE_FRAIS: 0.05,
    DocumentType.BON_DE_LIVRAISON: 0.05,
    DocumentType.AUTRE: 0.05,
}


def make_client(classifier=None, **settings) -> TestClient:
    classifier = classifier or DummyClassifier(CONFIDENT_INVOICE)
    return TestClient(create_app(Settings(classifier="dummy", **settings), classifier))


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["confidence_threshold"] == 0.6


@pytest.mark.parametrize("fixture, name", [("png_bytes", "scan.png"), ("jpeg_bytes", "photo.jpg")])
def test_classify_image(client, request, fixture, name):
    data = request.getfixturevalue(fixture)
    response = client.post("/v1/classify", files={"file": (name, data, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == name
    assert body["page_count"] == 1
    assert body["classification"]["type"] == "facture"
    assert body["classification"]["destination"] == "factures"
    assert body["classification"]["needs_review"] is False
    assert set(body["classification"]["scores"]) == {t.value for t in DocumentType}
    assert {"preprocess", "classify"} <= set(body["timings_ms"])


def test_classify_pdf_counts_pages(client, pdf_bytes):
    response = client.post(
        "/v1/classify", files={"file": ("doc.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    assert response.json()["page_count"] == 2


def test_low_confidence_goes_to_review():
    uncertain = DummyClassifier({t: 0.25 for t in DocumentType})
    with make_client(uncertain) as client:
        body = client.post("/v1/classify", files={"file": ("x.png", _png(), "image/png")}).json()
    assert body["classification"]["needs_review"] is True
    assert body["classification"]["destination"] == "a_verifier"


def test_unsupported_format_is_rejected(client):
    response = client.post("/v1/classify", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_corrupted_pdf_is_rejected(client):
    response = client.post(
        "/v1/classify", files={"file": ("x.pdf", b"%PDF-1.7 garbage", "application/pdf")}
    )
    assert response.status_code == 400


def test_too_large_file_is_rejected(pdf_bytes):
    with make_client(max_upload_mb=0) as client:
        response = client.post(
            "/v1/classify", files={"file": ("doc.pdf", pdf_bytes, "application/pdf")}
        )
    assert response.status_code == 413


def test_missing_file_is_rejected(client):
    assert client.post("/v1/classify").status_code == 422


def test_model_not_loaded_returns_503(monkeypatch):
    def failing_build(*_):
        raise OSError("modèle introuvable")

    monkeypatch.setattr("docrouter_api.main.build_classifier", failing_build)
    with TestClient(create_app(Settings())) as client:
        assert client.get("/health").json()["model_ready"] is False
        response = client.post("/v1/classify", files={"file": ("x.png", _png(), "image/png")})
    assert response.status_code == 503


def test_aggregate_scores_maps_rvlcdip_labels():
    scores = aggregate_scores(
        {"invoice": 0.7, "letter": 0.2, "memo": 0.1}, {"invoice": DocumentType.FACTURE}
    )
    assert scores[DocumentType.FACTURE] == pytest.approx(0.7)
    assert scores[DocumentType.AUTRE] == pytest.approx(0.3)
    assert scores[DocumentType.NOTE_DE_FRAIS] == 0.0


def _png() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_simulated_classifier_is_deterministic_and_varied():
    from PIL import Image

    clf = DummyClassifier()
    images = [Image.new("RGB", (32, 32), (i, 255 - i, i // 2)) for i in range(0, 250, 5)]
    predictions = [clf.predict(img).best for img in images]
    assert clf.predict(images[0]).best == predictions[0]
    assert {t for t, _ in predictions} == set(DocumentType)
    assert any(conf < 0.6 for _, conf in predictions)
    assert all(0.4 <= conf <= 0.95 for _, conf in predictions)


def test_dit_without_ml_extra_explains_how_to_install(monkeypatch):
    import builtins

    from docrouter_api.classifier import build_classifier

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(RuntimeError, match="extra `ml`"):
        build_classifier("dit", "any/model")
