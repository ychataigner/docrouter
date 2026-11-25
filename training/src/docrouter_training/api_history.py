"""Itération 8 — synthèse de l'historique des requêtes enregistré par l'API dans MLflow.

    uv run docrouter-api-history                  # les 2 derniers runs (expérience docrouter-api)
    uv run docrouter-api-history --runs 3 --out docs/images/api_history.png

Le tableau de bord vivant est l'interface MLflow (http://localhost:5000, expérience
`docrouter-api`) : courbes de latence et de confiance requête par requête, compteurs par type et
par client. Cette commande en tire une version imprimable : tableau comparatif des runs (modèle
lourd / optimisé) et graphique.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

EXPERIMENT = "docrouter-api"


def load_runs(limit: int) -> list[dict]:
    import mlflow

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise SystemExit(f"Expérience « {EXPERIMENT} » introuvable : l'API a-t-elle tourné ?")
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=limit
    )
    loaded = []
    for run in reversed(runs):  # du plus ancien au plus récent
        rows = []
        for artifact in client.list_artifacts(run.info.run_id, "history"):
            path = client.download_artifacts(run.info.run_id, artifact.path)
            with open(path, encoding="utf-8", newline="") as f:
                rows += list(csv.DictReader(f))
        rows.sort(key=lambda r: int(r["step"]))
        loaded.append({"name": run.info.run_name, "model": run.data.params.get("model", "?"),
                       "rows": rows})  # fmt: skip
    return loaded


def summarize(rows: list[dict]) -> dict[str, float]:
    total = np.array([float(r["total_ms"]) for r in rows])
    inference = np.array([float(r["inference_ms"]) for r in rows])
    queue = np.array([float(r["queue_ms"]) for r in rows])
    review = np.array([r["needs_review"] == "True" for r in rows])
    return {
        "requests": float(len(rows)),
        "total_p50_ms": float(np.percentile(total, 50)),
        "total_p95_ms": float(np.percentile(total, 95)),
        "inference_p50_ms": float(np.percentile(inference, 50)),
        "queue_p95_ms": float(np.percentile(queue, 95)),
        "review_rate": float(review.mean()),
    }


def table(runs: list[dict]) -> str:
    lines = [
        "| Run | Requêtes | Total p50 (ms) | Total p95 (ms) | Inférence p50 (ms) "
        "| Attente p95 (ms) | « À vérifier » |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        if not run["rows"]:
            continue
        s = summarize(run["rows"])
        lines.append(
            f"| {run['name']} | {s['requests']:.0f} | {s['total_p50_ms']:.0f} "
            f"| {s['total_p95_ms']:.0f} | {s['inference_p50_ms']:.0f} | {s['queue_p95_ms']:.0f} "
            f"| {s['review_rate']:.1%} |"
        )
    return "\n".join(lines)


def plot(runs: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        if not run["rows"]:
            continue
        label = run["name"].split(" — ")[0]
        steps = [int(r["step"]) for r in run["rows"]]
        inference = [float(r["inference_ms"]) for r in run["rows"]]
        total = np.array([float(r["total_ms"]) for r in run["rows"]])
        window = max(1, len(total) // 50)
        smooth = np.convolve(total, np.ones(window) / window, mode="same")
        ax1.plot(steps, smooth, label=label)
        ax2.hist(inference, bins=40, alpha=0.6, label=label)
    ax1.set_xlabel("Requête n°")
    ax1.set_ylabel("Temps total côté API (ms, moyenne glissante)")
    ax1.set_title("Historique des requêtes (paliers de charge successifs)")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Temps d'inférence du modèle (ms)")
    ax2.set_ylabel("Nombre de requêtes")
    ax2.set_title("Distribution du temps d'inférence")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=2, help="nombre de runs récents à comparer")
    parser.add_argument("--out", type=Path, default=Path("docs/images/api_history.png"))
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    runs = load_runs(args.runs)
    print(table(runs))
    plot(runs, args.out)
    print(f"\nGraphique : {args.out}")


if __name__ == "__main__":
    main()
