"""Itération 6 : authentification par clé d'API et aucune image conservée sur le serveur."""

import io
import logging
import tempfile

import httpx
import pytest
from docrouter_api.classifier import DummyClassifier
from docrouter_api.main import create_app
from docrouter_api.security import ApiKeys
from docrouter_api.settings import Settings
from docrouter_core import DocumentType
from docrouter_ui.client import ApiClient
from fastapi.testclient import TestClient
from PIL import Image

FRONT = "front-0123456789abcdefghijklmn"
COMPTA = "compta-0123456789abcdefghijklm"
SCORES = {t: 0.9 if t == DocumentType.FACTURE else 0.1 / 3 for t in DocumentType}


def make_app(**settings):
    settings.setdefault("api_keys", f"front:{FRONT},compta:{COMPTA}")
    return create_app(Settings(classifier="dummy", **settings), DummyClassifier(SCORES))


def big_jpeg(min_bytes: int = 2_500_000) -> bytes:
    """JPEG de plus de 2 Mo (bruit) : au-delà du seuil où Starlette écrivait sur disque."""
    import numpy as np

    noise = np.random.default_rng(0).integers(0, 255, (1800, 1300, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noise).save(buffer, format="JPEG", quality=95)
    data = buffer.getvalue()
    assert len(data) > min_bytes
    return data


def post(client, data, headers=None, name="doc.jpg"):
    return client.post("/v1/classify", files={"file": (name, data, "image/jpeg")}, headers=headers)


# --- Clés d'API ---------------------------------------------------------------------------


def test_parse_named_and_anonymous_keys():
    keys = ApiKeys.parse(f" front:{FRONT} , {COMPTA} ")
    assert keys.identify(FRONT) == "front"
    assert keys.identify(COMPTA) == "client-2"
    assert keys.identify("x" * 30) is None
    assert keys.identify("") is None and keys.identify(None) is None
    assert not ApiKeys.parse("")


def test_short_keys_are_refused_at_startup():
    with pytest.raises(ValueError, match="trop courte"):
        ApiKeys.parse("front:1234")


def test_missing_or_wrong_key_is_401(png_bytes):
    with TestClient(make_app()) as client:
        assert post(client, png_bytes).status_code == 401
        response = post(client, png_bytes, {"X-API-Key": "mauvaise-cle-0123456789abc"})
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_each_client_has_its_own_key(png_bytes, caplog):
    with TestClient(make_app()) as client, caplog.at_level(logging.INFO, "docrouter.api"):
        assert post(client, png_bytes, {"X-API-Key": FRONT}).status_code == 200
        assert post(client, png_bytes, {"X-API-Key": COMPTA}).status_code == 200
    clients = [r.getMessage() for r in caplog.records if "request_id" in r.getMessage()]
    assert any("client=front" in m for m in clients) and any("client=compta" in m for m in clients)


def test_revoked_client_is_refused(png_bytes):
    """Révoquer = retirer la clé de DOCROUTER_API_KEYS : les autres clients ne sont pas touchés."""
    with TestClient(make_app(api_keys=f"front:{FRONT}")) as client:
        assert post(client, png_bytes, {"X-API-Key": COMPTA}).status_code == 401
        assert post(client, png_bytes, {"X-API-Key": FRONT}).status_code == 200


def test_without_configured_keys_the_api_is_closed(png_bytes):
    with TestClient(make_app(api_keys="")) as client:
        response = post(client, png_bytes, {"X-API-Key": FRONT})
    assert response.status_code == 503
    assert "DOCROUTER_API_KEYS" in response.json()["detail"]


def test_public_routes_stay_open():
    with TestClient(make_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200


def test_swagger_declares_the_api_key():
    with TestClient(make_app()) as client:
        schema = client.get("/openapi.json").json()
    scheme = next(iter(schema["components"]["securitySchemes"].values()))
    assert scheme == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Clé d'API du client",
    }


def test_too_large_request_is_refused_before_reading(png_bytes):
    with TestClient(make_app(max_upload_mb=1)) as client:
        response = post(client, big_jpeg(), {"X-API-Key": FRONT})
    assert response.status_code == 413


def test_front_client_sends_its_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("X-API-Key")
        return httpx.Response(200, json={"status": "ok", "version": "t", "model": "m",
                                         "model_ready": True, "confidence_threshold": 0.6,
                                         "max_upload_mb": 10})  # fmt: skip

    ApiClient("http://api", transport=httpx.MockTransport(handler), api_key=FRONT).health()
    assert seen["key"] == FRONT
    ApiClient("http://api", transport=httpx.MockTransport(handler)).health()
    assert seen["key"] is None


# --- Aucune image conservée ---------------------------------------------------------------


def test_large_upload_never_touches_the_disk(monkeypatch):
    """Sans correctif, Starlette écrit sur disque tout fichier de plus de 1 Mo reçu."""
    written = []
    real_rollover = tempfile.SpooledTemporaryFile.rollover

    def spy_rollover(self):
        written.append(True)
        return real_rollover(self)

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", spy_rollover)
    with TestClient(make_app()) as client:
        response = post(client, big_jpeg(), {"X-API-Key": FRONT})
    assert response.status_code == 200
    assert written == []


def test_no_file_left_in_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with TestClient(make_app()) as client:
        assert post(client, big_jpeg(), {"X-API-Key": FRONT}).status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_logs_never_contain_the_document(caplog):
    data = big_jpeg()
    with TestClient(make_app()) as client, caplog.at_level(logging.DEBUG):
        post(client, data, {"X-API-Key": FRONT}, name="facture-dupont.jpg")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "facture-dupont" not in text  # pas même le nom du fichier
    assert FRONT not in text  # ni la clé
