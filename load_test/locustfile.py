"""Itération 5 — afflux continu de documents vers l'API (Locust).

Chaque utilisateur virtuel envoie en boucle, sans pause, un document tiré au hasard dans le jeu
de test (factures, tickets photographiés, BL, PDF…) : c'est un flux continu. Le nombre
d'utilisateurs monte par paliers (`LOAD_STEPS`), pour trouver le débit maximal et le point où
les temps de réponse se dégradent.

    $env:DOCROUTER_API_KEY = "<clé du front>"
    uv run locust -f load_test/locustfile.py --headless --host http://localhost:8000

Variables : LOAD_STEPS (défaut « 1,2,4,8,16,32 »), LOAD_STEP_SECONDS (défaut 30),
LOAD_DATA (défaut data/synth), LOAD_OUT (défaut reports/load/requests.csv).
Chaque requête est enregistrée (instant, palier, durée, succès, classe attendue / prédite) ;
`docrouter-load-report` en tire le tableau et les graphiques.
"""

import csv
import os
import random
import time
from pathlib import Path

from locust import FastHttpUser, LoadTestShape, events, task

DATA = Path(os.environ.get("LOAD_DATA", "data/synth"))
OUT = Path(os.environ.get("LOAD_OUT", "reports/load/requests.csv"))
STEPS = [int(n) for n in os.environ.get("LOAD_STEPS", "1,2,4,8,16,32").split(",")]
STEP_SECONDS = int(os.environ.get("LOAD_STEP_SECONDS", "30"))
API_KEY = os.environ.get("DOCROUTER_API_KEY", "")


def _load_documents() -> list[tuple[str, str, bytes, str]]:
    """(nom, classe attendue, contenu, type MIME) : tout en mémoire, le disque ne freine pas."""
    with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter=";") if r["split"] == "test"]
    docs = []
    for row in rows:
        path = DATA / row["fichier"]
        mime = "application/pdf" if path.suffix == ".pdf" else "image/jpeg"
        docs.append((path.name, row["classe"], path.read_bytes(), mime))
    return docs


DOCUMENTS = _load_documents()
RECORDS: list[dict] = []
_current_step = {"users": 0}


class DocumentSender(FastHttpUser):
    """Un client qui envoie un document, attend la réponse, et recommence aussitôt."""

    connection_timeout = 30
    network_timeout = 60

    @task
    def classify(self) -> None:
        name, expected, data, mime = random.choice(DOCUMENTS)
        start = time.perf_counter()
        with self.client.post(
            "/v1/classify",
            files={"file": (name, data, mime)},
            headers={"X-API-Key": API_KEY},
            name="/v1/classify",
            catch_response=True,
        ) as response:
            elapsed_ms = (time.perf_counter() - start) * 1000
            predicted = ""
            if response.status_code == 200:
                predicted = response.json()["classification"]["type"]
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
            RECORDS.append(
                {
                    "t": time.time(),
                    "users": _current_step["users"],
                    "ms": round(elapsed_ms, 1),
                    "ok": int(response.status_code == 200),
                    "status": response.status_code,
                    "expected": expected,
                    "predicted": predicted,
                    "bytes": len(data),
                }
            )


class StepLoad(LoadTestShape):
    """Paliers d'utilisateurs simultanés, chacun maintenu STEP_SECONDS secondes."""

    def tick(self):
        step = int(self.get_run_time() // STEP_SECONDS)
        if step >= len(STEPS):
            return None
        users = STEPS[step]
        _current_step["users"] = users
        return users, users  # (utilisateurs, nouveaux utilisateurs par seconde)


@events.test_start.add_listener
def _check(environment, **_):
    if not API_KEY:
        raise SystemExit("Définir DOCROUTER_API_KEY (clé du front, voir .env).")
    print(f"{len(DOCUMENTS)} documents ; paliers {STEPS} × {STEP_SECONDS} s")


@events.test_stop.add_listener
def _save(environment, **_):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RECORDS[0]) if RECORDS else ["t"])
        writer.writeheader()
        writer.writerows(RECORDS)
    print(f"{len(RECORDS)} requêtes enregistrées dans {OUT}")
