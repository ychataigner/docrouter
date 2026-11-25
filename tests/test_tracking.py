"""Itération 8 : historique des requêtes envoyé à MLflow (faux serveur MLflow en mémoire)."""

import csv
import io
import json
import urllib.error

from docrouter_api.classifier import DummyClassifier
from docrouter_api.main import create_app
from docrouter_api.settings import Settings
from docrouter_api.tracking import RequestEvent, RequestTracker
from docrouter_core import DocumentType
from fastapi.testclient import TestClient

KEY = "cle-de-test-tracking-0123456789"


class FakeMlflow:
    """Imite l'API REST de MLflow et garde tout ce qu'on lui envoie."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.fail = fail

    def __call__(self, method, path, body, content_type):
        if self.fail:
            raise urllib.error.URLError("connexion refusée")
        self.calls.append((method, path, body))
        if "get-by-name" in path:
            raise urllib.error.HTTPError(path, 404, "absent", {}, None)
        if path.endswith("experiments/create"):
            return {"experiment_id": "7"}
        if path.endswith("runs/create"):
            return {
                "run": {
                    "info": {"run_id": "r1", "artifact_uri": "mlflow-artifacts:/7/r1/artifacts"}
                }
            }
        return {}

    def batches(self) -> list[dict]:
        return [json.loads(b) for m, p, b in self.calls if p.endswith("log-batch")]

    def metrics(self) -> list[dict]:
        return [m for batch in self.batches() for m in batch.get("metrics", [])]

    def csv_rows(self) -> list[dict]:
        files = [b for m, p, b in self.calls if m == "PUT"]
        return [row for f in files for row in csv.DictReader(io.StringIO(f.decode()))]


def event(doc_type: str = "facture", review: bool = False, client: str = "front") -> RequestEvent:
    return RequestEvent("abcd1234", client, doc_type, 0.42 if review else 0.97,
                        "a_verifier" if review else "factures", review, 1,
                        120.0, 30.0, 5.0, 60.0)  # fmt: skip


def test_flush_creates_the_run_then_logs_metrics_counts_and_csv():
    mlflow = FakeMlflow()
    tracker = RequestTracker(mlflow, run_name="test", params={"model": "m"})
    tracker.record(event())
    tracker.record(event("autre", review=True, client="compta"))
    assert tracker.flush() == 2

    paths = [p for _, p, _ in mlflow.calls]
    assert any("experiments/create" in p for p in paths)  # expérience absente → créée
    assert any(p.endswith("runs/create") for p in paths)
    assert {"key": "model", "value": "m"} in mlflow.batches()[0]["params"]

    by_key = {}
    for m in mlflow.metrics():
        by_key.setdefault(m["key"], []).append(m)
    assert [m["step"] for m in by_key["latency_total_ms"]] == [1, 2]
    assert by_key["confidence"][1]["value"] == 0.42
    last = {k: v[-1]["value"] for k, v in by_key.items()}
    assert last["requests_total"] == 2
    assert last["count_facture"] == 1 and last["count_autre"] == 1
    assert last["client_front"] == 1 and last["client_compta"] == 1
    assert last["review_rate"] == 0.5

    rows = mlflow.csv_rows()
    assert [r["type"] for r in rows] == ["facture", "autre"]
    put_path = next(p for m, p, _ in mlflow.calls if m == "PUT")
    assert put_path.startswith("/api/2.0/mlflow-artifacts/artifacts/7/r1/artifacts/history/")


def test_steps_continue_across_flushes():
    mlflow = FakeMlflow()
    tracker = RequestTracker(mlflow, run_name="t", params={})
    tracker.record(event())
    tracker.flush()
    tracker.record(event())
    tracker.flush()
    steps = [m["step"] for m in mlflow.metrics() if m["key"] == "latency_total_ms"]
    assert steps == [1, 2]
    assert sum(p.endswith("runs/create") for _, p, _ in mlflow.calls) == 1  # un seul run


def test_mlflow_down_never_breaks_anything():
    tracker = RequestTracker(FakeMlflow(fail=True), run_name="t", params={})
    tracker.record(event())
    assert tracker.flush() == 0
    assert tracker.dropped == 1
    tracker.stop()  # ne lève rien


def test_full_queue_drops_instead_of_blocking():
    tracker = RequestTracker(FakeMlflow(), run_name="t", params={}, max_queue=1)
    tracker.record(event())
    tracker.record(event())
    assert tracker.dropped == 1


def test_api_records_one_event_per_request_without_document_content(png_bytes):
    mlflow = FakeMlflow()
    tracker = RequestTracker(mlflow, run_name="api", params={}, flush_every_s=3600)
    scores = {t: 0.9 if t == DocumentType.FACTURE else 0.1 / 3 for t in DocumentType}
    app = create_app(
        Settings(classifier="dummy", api_keys=f"front:{KEY}"), DummyClassifier(scores), tracker
    )
    with TestClient(app, headers={"X-API-Key": KEY}) as client:
        for _ in range(3):
            response = client.post(
                "/v1/classify", files={"file": ("facture-dupont.png", png_bytes, "image/png")}
            )
            assert response.status_code == 200
    # À l'arrêt de l'API, le tracker vide sa file et clôt le run.
    rows = mlflow.csv_rows()
    assert len(rows) == 3 and {r["client"] for r in rows} == {"front"}
    sent = b"".join(b or b"" for _, _, b in mlflow.calls)
    assert b"facture-dupont" not in sent  # ni nom de fichier …
    assert png_bytes[:16] not in sent  # … ni contenu
    assert any(p.endswith("runs/update") for _, p, _ in mlflow.calls)


def test_tracking_disabled_by_default(png_bytes):
    app = create_app(Settings(classifier="dummy", api_keys=f"front:{KEY}"), DummyClassifier())
    with TestClient(app, headers={"X-API-Key": KEY}) as client:
        assert client.app.state.tracker is None
        assert client.post("/v1/classify", files={"file": ("x.png", png_bytes)}).status_code == 200


def test_history_summary_and_table():
    from docrouter_training.api_history import summarize, table

    rows = [
        {"step": str(i), "total_ms": str(100 + i), "inference_ms": "50", "queue_ms": "0",
         "needs_review": "True" if i == 0 else "False"}
        for i in range(10)
    ]  # fmt: skip
    s = summarize(rows)
    assert s["requests"] == 10 and s["inference_p50_ms"] == 50
    assert s["review_rate"] == 0.1
    md = table([{"name": "onnx", "model": "m", "rows": rows}, {"name": "vide", "rows": []}])
    assert "| onnx | 10 |" in md and "vide" not in md
