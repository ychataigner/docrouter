"""Itération 8 — historique des requêtes de l'API dans MLflow.

Chaque démarrage de l'API ouvre un run MLflow (expérience `docrouter-api`) ; chaque requête y
ajoute un point : durées (totale, prétraitement, attente, inférence), confiance, passage en
« à vérifier », et des compteurs cumulés par type et par client. Toutes les 5 s, un fichier CSV
de l'historique est déposé dans les artefacts du run.

- **Aucun contenu du document** n'est envoyé : ni image, ni nom de fichier, ni texte.
- **Sans impact sur la réponse** : la route dépose l'événement dans une file en mémoire ; un
  thread l'envoie par lots. Si MLflow est indisponible, les événements sont abandonnés (avec un
  avertissement) et l'API continue de répondre.
- **Bibliothèque standard uniquement** (API REST de MLflow) : le paquet `mlflow` n'est pas
  installé dans l'image de production.
"""

import csv
import io
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

logger = logging.getLogger("docrouter.tracking")

EXPERIMENT = "docrouter-api"
CSV_FIELDS = [
    "step",
    "time",
    "request_id",
    "client",
    "type",
    "confidence",
    "destination",
    "needs_review",
    "pages",
    "total_ms",
    "preprocess_ms",
    "queue_ms",
    "inference_ms",
]


@dataclass
class RequestEvent:
    """Métadonnées d'une requête : rien qui permette de retrouver le document."""

    request_id: str
    client: str
    type: str
    confidence: float
    destination: str
    needs_review: bool
    pages: int
    total_ms: float
    preprocess_ms: float
    queue_ms: float
    inference_ms: float
    time: float = 0.0  # horodatage Unix, renseigné à l'enregistrement


# (méthode, chemin, corps JSON ou octets, type de contenu) → réponse JSON
Transport = Callable[[str, str, bytes | None, str], dict]


def http_transport(base_url: str, timeout: float = 5.0) -> Transport:
    base = base_url.rstrip("/")

    def send(method: str, path: str, body: bytes | None, content_type: str) -> dict:
        request = urllib.request.Request(
            base + path, data=body, method=method, headers={"Content-Type": content_type}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        return json.loads(payload) if payload else {}

    return send


class RequestTracker:
    def __init__(
        self,
        transport: Transport,
        run_name: str,
        params: dict[str, str],
        flush_every_s: float = 5.0,
        max_queue: int = 10_000,
    ) -> None:
        self._send = transport
        self._run_name, self._params = run_name, params
        self._flush_every_s = flush_every_s
        self._queue: queue.Queue[RequestEvent] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_id: str | None = None
        self._artifact_path: str | None = None
        self._step = 0
        self._counts: Counter[str] = Counter()
        self.dropped = 0

    # --- côté API (doit rester instantané) -------------------------------------------------

    def record(self, event: RequestEvent) -> None:
        event.time = time.time()
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    # --- cycle de vie -------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="mlflow-tracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.flush()
        if self._run_id:
            self._call(
                "POST",
                "/api/2.0/mlflow/runs/update",
                {"run_id": self._run_id, "status": "FINISHED", "end_time": _now_ms()},
            )

    def _loop(self) -> None:
        while not self._stop.wait(self._flush_every_s):
            self.flush()

    # --- envoi vers MLflow ---------------------------------------------------------------

    def flush(self) -> int:
        events: list[RequestEvent] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not events:
            return 0
        if not self._ensure_run():
            self.dropped += len(events)
            return 0
        first_step = self._step + 1
        metrics, rows = [], []
        for event in events:
            self._step += 1
            self._counts[f"count_{event.type}"] += 1
            self._counts[f"client_{event.client}"] += 1
            self._counts["count_needs_review"] += int(event.needs_review)
            ts = int(event.time * 1000)
            values = {
                "latency_total_ms": event.total_ms,
                "latency_preprocess_ms": event.preprocess_ms,
                "latency_queue_ms": event.queue_ms,
                "latency_inference_ms": event.inference_ms,
                "confidence": event.confidence,
                "needs_review": float(event.needs_review),
            }
            metrics += [_metric(k, v, ts, self._step) for k, v in values.items()]
            readable = datetime.fromtimestamp(event.time).isoformat(timespec="milliseconds")
            rows.append({**asdict(event), "step": self._step, "time": readable})
        ts = int(events[-1].time * 1000)
        total = self._step
        metrics += [_metric(k, v, ts, total) for k, v in self._counts.items()]
        metrics.append(_metric("requests_total", total, ts, total))
        metrics.append(
            _metric("review_rate", self._counts["count_needs_review"] / total, ts, total)
        )
        for start in range(0, len(metrics), 900):  # limite MLflow : 1 000 valeurs par lot
            self._call(
                "POST",
                "/api/2.0/mlflow/runs/log-batch",
                {"run_id": self._run_id, "metrics": metrics[start : start + 900]},
            )
        self._upload_csv(f"history/requests_{first_step:07d}-{total:07d}.csv", rows)
        return len(events)

    def _ensure_run(self) -> bool:
        if self._run_id:
            return True
        found = self._call(
            "GET",
            "/api/2.0/mlflow/experiments/get-by-name?"
            + urllib.parse.urlencode({"experiment_name": EXPERIMENT}),
            None,
        )
        if found and "experiment" in found:
            experiment_id = found["experiment"]["experiment_id"]
        else:
            created = self._call("POST", "/api/2.0/mlflow/experiments/create", {"name": EXPERIMENT})
            if not created:
                return False
            experiment_id = created["experiment_id"]
        run = self._call(
            "POST",
            "/api/2.0/mlflow/runs/create",
            {
                "experiment_id": experiment_id,
                "run_name": self._run_name,
                "start_time": _now_ms(),
                "tags": [{"key": "source", "value": "docrouter-api"}],
            },
        )
        if not run:
            return False
        info = run["run"]["info"]
        self._run_id = info["run_id"]
        self._artifact_path = info["artifact_uri"].removeprefix("mlflow-artifacts:/")
        self._call(
            "POST",
            "/api/2.0/mlflow/runs/log-batch",
            {
                "run_id": self._run_id,
                "params": [{"key": k, "value": str(v)[:500]} for k, v in self._params.items()],
            },
        )
        logger.info("Suivi MLflow actif : run %s (%s)", self._run_id, self._run_name)
        return True

    def _upload_csv(self, name: str, rows: list[dict]) -> None:
        if not self._artifact_path:
            return
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        path = f"/api/2.0/mlflow-artifacts/artifacts/{self._artifact_path}/{name}"
        self._call("PUT", path, buffer.getvalue().encode(), content_type="text/csv")

    def _call(
        self, method: str, path: str, body, content_type: str = "application/json"
    ) -> dict | None:
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        try:
            return self._send(method, path, data, content_type)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and method == "GET":  # ex. expérience pas encore créée
                return {}
            logger.warning("MLflow a refusé %s %s : %s", method, path.split("?")[0], exc)
            return None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("MLflow indisponible (%s %s) : %s", method, path.split("?")[0], exc)
            return None


def _metric(key: str, value: float, timestamp: int, step: int) -> dict:
    return {"key": key, "value": float(value), "timestamp": timestamp, "step": step}


def _now_ms() -> int:
    return int(time.time() * 1000)
