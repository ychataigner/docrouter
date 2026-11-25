"""Banc d'évaluation : mêmes documents, mêmes métriques pour tous les modèles, suivi dans MLflow.

    uv sync --group train
    docker compose up -d mlflow
    uv run docrouter-evaluate --model all

Chaque modèle donne un run MLflow (expérience `docrouter-classification`) avec :
accuracy, F1-macro, F1 par classe, précision par effet (propre / scan / photo) et par format,
courbe couverture/précision du seuil `a_verifier`, latence p50/p95, taille du modèle,
matrice de confusion et prédictions détaillées en artefacts.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from docrouter_training.eval import metrics as m
from docrouter_training.eval.backends import BACKEND_NAMES, Backend, make_backend
from docrouter_training.eval.dataset import Sample, dataset_fingerprint, load_image, read_manifest

logger = logging.getLogger("docrouter.evaluate")

EXPERIMENT = "docrouter-classification"
API_THRESHOLD = 0.6  # seuil actuel de l'API (DOCROUTER_CONFIDENCE_THRESHOLD)


def run_predictions(backend: Backend, samples: Sequence[Sample]) -> list[dict]:
    rows = []
    for i, sample in enumerate(samples, 1):
        image = load_image(sample.path)
        t0 = time.perf_counter()
        scores = backend.predict(image)
        elapsed = time.perf_counter() - t0
        predicted = max(scores, key=scores.__getitem__)
        rows.append(
            {
                "fichier": sample.path.name,
                "classe": sample.label,
                "prediction": predicted,
                "confiance": scores[predicted],
                "correct": predicted == sample.label,
                "profil": sample.profile,
                "format": sample.fmt,
                "latence_s": elapsed,
                **{f"score_{c.value}": scores[c] for c in m.CLASSES},
            }
        )
        if i % 20 == 0:
            logger.info("%s : %d/%d", backend.name, i, len(samples))
    return rows


def compute_metrics(rows: Sequence[dict], target_accuracy: float) -> tuple[dict, list]:
    y_true = [r["classe"] for r in rows]
    y_pred = [r["prediction"] for r in rows]
    result: dict[str, float] = {
        "accuracy": m.accuracy(y_true, y_pred),
        "f1_macro": m.f1_macro(y_true, y_pred),
        "n_documents": float(len(rows)),
    }
    for cls, rep in m.per_class(y_true, y_pred).items():
        result[f"f1_{cls.value}"] = rep.f1
        result[f"recall_{cls.value}"] = rep.recall
        result[f"precision_{cls.value}"] = rep.precision

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[f"accuracy_{r['profil']}"].append(r)
        groups[f"accuracy_{r['format']}"].append(r)
    for key, group in sorted(groups.items()):
        result[key] = sum(r["correct"] for r in group) / len(group)

    curve = m.coverage_curve([r["correct"] for r in rows], [r["confiance"] for r in rows])
    at_api = curve[round(API_THRESHOLD * 100)]
    result["coverage_at_api_threshold"] = at_api.coverage
    result["accuracy_at_api_threshold"] = at_api.accuracy
    best = m.recommend_threshold(curve, target_accuracy)
    if best:
        result["recommended_threshold"] = best.threshold
        result["coverage_at_recommended"] = best.coverage
    result.update(m.latency_stats([r["latence_s"] for r in rows]))
    return result, curve


def write_report(
    out: Path, rows: Sequence[dict], metrics: dict, curve: list, title: str, plots: bool
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows({k: str(v) for k, v in r.items()} for r in rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "coverage.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("seuil;couverture;precision\n")
        f.writelines(f"{p.threshold:.2f};{p.coverage:.4f};{p.accuracy:.4f}\n" for p in curve)
    if plots:
        _plot(out, rows, curve, title)


def _plot(out: Path, rows: Sequence[dict], curve: list, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = m.confusion_matrix([r["classe"] for r in rows], [r["prediction"] for r in rows])
    labels = [c.value.replace("_", " ") for c in m.CLASSES]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color=color)
    ax.set_title(f"{title} — matrice de confusion")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p.threshold for p in curve], [p.coverage for p in curve], label="rangés auto")
    ax.plot([p.threshold for p in curve], [p.accuracy for p in curve], label="précision")
    ax.axvline(API_THRESHOLD, color="grey", linestyle="--", label="seuil actuel")
    ax.set_xlabel("Seuil de confiance")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.set_title(f"{title} — seuil « à vérifier »")
    fig.tight_layout()
    fig.savefig(out / "coverage_curve.png", dpi=120)
    plt.close(fig)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"


def evaluate(
    backend: Backend,
    data_root: Path,
    split: str = "test",
    target_accuracy: float = 0.98,
    out_root: Path = Path("reports/eval"),
    use_mlflow: bool = True,
    plots: bool = True,
) -> dict:
    samples = read_manifest(data_root, split)
    if not samples:
        raise SystemExit(f"Aucun document « {split} » dans {data_root / 'manifest.csv'}")

    prep = backend.prepare(read_manifest(data_root, "train"), read_manifest(data_root, "val"))
    rows = run_predictions(backend, samples)
    result, curve = compute_metrics(rows, target_accuracy)
    result.update(backend.size())
    result.update({f"prep_{k}": v for k, v in prep.items()})

    out = out_root / backend.name
    write_report(out, rows, result, curve, backend.name, plots)

    if use_mlflow:
        import mlflow

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=backend.name):
            mlflow.set_tags({"split": split, "git_commit": _git_commit(), "stage": "comparaison"})
            mlflow.log_params(
                {
                    "backend": backend.name,
                    **backend.params(),
                    **dataset_fingerprint(data_root),
                    "split": split,
                    "target_accuracy": target_accuracy,
                }
            )
            mlflow.log_metrics(result)
            for point in curve[::5]:
                step = round(point.threshold * 100)
                mlflow.log_metric("coverage_curve", point.coverage, step=step)
            mlflow.log_artifacts(str(out))
            saved = getattr(backend, "saved_path", None)
            if saved:
                mlflow.log_artifact(str(saved), artifact_path="model")
    return result


COLUMNS = [
    ("accuracy", "Accuracy", "{:.1%}"),
    ("f1_macro", "F1-macro", "{:.3f}"),
    ("f1_facture", "F1 fact.", "{:.2f}"),
    ("f1_note_de_frais", "F1 NdF", "{:.2f}"),
    ("f1_bon_de_livraison", "F1 BL", "{:.2f}"),
    ("f1_autre", "F1 autre", "{:.2f}"),
    ("accuracy_photo", "Photo", "{:.0%}"),
    ("coverage_at_recommended", "Rangés auto", "{:.0%}"),
    ("latency_ms_p50", "p50 (ms)", "{:.0f}"),
    ("latency_ms_p95", "p95 (ms)", "{:.0f}"),
    ("size_mb_fp32", "Taille (Mo)", "{:.0f}"),
]


def comparison_table(results: dict[str, dict]) -> str:
    head = "| Modèle | " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    sep = "|---" * (len(COLUMNS) + 1) + "|"
    lines = [head, sep]
    for name, res in results.items():
        cells = [fmt.format(res[key]) if key in res else "—" for key, _, fmt in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model", nargs="+", default=["all"], choices=[*BACKEND_NAMES, "all"], metavar="MODELE"
    )
    parser.add_argument("--data", type=Path, default=Path("data/synth"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.98,
        help="précision visée sur les documents rangés automatiquement (choix du seuil)",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/eval"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/probes"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    names = list(BACKEND_NAMES) if "all" in args.model else args.model
    cache = args.data / "features" / dataset_fingerprint(args.data)["dataset_sha256"][:12]

    results = {}
    for name in names:
        logger.info("=== %s ===", name)
        backend = make_backend(name, cache_dir=cache, save_dir=args.models_dir)
        results[name] = evaluate(
            backend, args.data, args.split, args.target_accuracy, args.out, not args.no_mlflow
        )
        del backend

    table = comparison_table(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(
        f"\nRapports : {args.out}" + ("" if args.no_mlflow else " — MLflow : http://localhost:5000")
    )


if __name__ == "__main__":
    main()
