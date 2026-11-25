"""Banc d'évaluation : métriques, lecture du jeu, chaîne complète avec un faux modèle.

Aucun réseau ni dépendance lourde : ces tests tournent en CI. Les vrais modèles sont évalués
avec `uv run docrouter-evaluate` (groupe `train`).
"""

import json

import numpy as np
import pytest
from docrouter_core import DocumentType as T
from docrouter_training.data.generate import generate
from docrouter_training.eval import metrics as m
from docrouter_training.eval.dataset import dataset_fingerprint, load_image, read_manifest
from docrouter_training.evaluate import comparison_table, compute_metrics, evaluate

# --- Métriques ------------------------------------------------------------------------------

Y_TRUE = [T.FACTURE, T.FACTURE, T.NOTE_DE_FRAIS, T.BON_DE_LIVRAISON, T.AUTRE, T.AUTRE]
Y_PRED = [T.FACTURE, T.AUTRE, T.NOTE_DE_FRAIS, T.FACTURE, T.AUTRE, T.AUTRE]


def test_accuracy():
    assert m.accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)
    assert m.accuracy([], []) == 0.0


def test_confusion_matrix_rows_are_truth():
    matrix = m.confusion_matrix(Y_TRUE, Y_PRED)
    i = {c: k for k, c in enumerate(m.CLASSES)}
    assert matrix.sum() == len(Y_TRUE)
    assert matrix[i[T.FACTURE], i[T.AUTRE]] == 1  # une facture prise pour « autre »
    assert matrix[i[T.BON_DE_LIVRAISON], i[T.FACTURE]] == 1
    assert np.trace(matrix) == 4


def test_per_class_and_f1_macro():
    reports = m.per_class(Y_TRUE, Y_PRED)
    # facture : 1 vrai positif, 2 prédites, 2 réelles → P = R = F1 = 0,5
    assert reports[T.FACTURE].precision == pytest.approx(0.5)
    assert reports[T.FACTURE].recall == pytest.approx(0.5)
    # autre : 2 VP, 3 prédites, 2 réelles → P = 2/3, R = 1, F1 = 0,8
    assert reports[T.AUTRE].f1 == pytest.approx(0.8)
    assert reports[T.BON_DE_LIVRAISON].f1 == 0.0
    assert m.f1_macro(Y_TRUE, Y_PRED) == pytest.approx((0.5 + 1.0 + 0.0 + 0.8) / 4)


def test_perfect_prediction():
    assert m.f1_macro(Y_TRUE, Y_TRUE) == 1.0


def test_coverage_curve_and_threshold():
    correct = [True, True, True, False]
    confidence = [0.95, 0.9, 0.7, 0.5]
    curve = m.coverage_curve(correct, confidence)
    assert curve[0].coverage == 1.0 and curve[0].accuracy == 0.75
    assert curve[60].coverage == 0.75 and curve[60].accuracy == 1.0
    assert curve[100].coverage == 0.0
    best = m.recommend_threshold(curve, target_accuracy=1.0)
    assert best is not None
    assert 0.5 < best.threshold <= 0.7  # on écarte seulement l'erreur
    assert best.coverage == 0.75
    assert m.recommend_threshold(m.coverage_curve([False], [0.9]), 0.9) is None


def test_latency_stats():
    stats = m.latency_stats([0.1, 0.2, 0.3])
    assert stats["latency_ms_p50"] == pytest.approx(200)
    assert stats["latency_ms_mean"] == pytest.approx(200)
    assert m.latency_stats([]) == {}


# --- Jeu de données et chaîne complète ------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    generate(root, seed=3, counts={"train": 2, "val": 1, "test": 3})
    return root


def test_read_manifest(tiny_dataset):
    test = read_manifest(tiny_dataset, "test")
    assert len(test) == 3 * len(T)
    assert {s.label for s in test} == set(T)
    assert all(s.layout == 3 and s.path.exists() for s in test)
    assert len(read_manifest(tiny_dataset)) == (2 + 1 + 3) * len(T)
    assert "dataset_sha256" in dataset_fingerprint(tiny_dataset)


def test_load_image_like_the_api(tiny_dataset):
    for sample in read_manifest(tiny_dataset, "test")[:4]:
        image = load_image(sample.path)
        assert image.mode == "RGB"
        assert min(image.size) > 100


class OracleBackend:
    """Faux modèle, sans réseau : le test lui fait donner la bonne réponse."""

    name = "oracle"

    def params(self):
        return {"model_id": "oracle"}

    def size(self):
        return {"size_mb_fp32": 0.0}

    def prepare(self, train, val):
        self.seen = (len(train), len(val))
        return {"val_accuracy": 1.0}

    def predict(self, image):
        return dict.fromkeys(T, 0.1)


def test_evaluate_end_to_end_without_mlflow(tiny_dataset, tmp_path, monkeypatch):
    # L'oracle ne voit que l'image : on lui fait répondre la bonne classe dans l'ordre du manifest.
    labels = iter(s.label for s in read_manifest(tiny_dataset, "test"))
    backend = OracleBackend()

    def predict(image):
        label = next(labels)
        return {c: 0.9 if c == label else 0.1 / 3 for c in T}

    monkeypatch.setattr(backend, "predict", predict)

    result = evaluate(backend, tiny_dataset, out_root=tmp_path, use_mlflow=False, plots=False)

    assert backend.seen == (2 * len(T), 1 * len(T))
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0
    assert result["coverage_at_api_threshold"] == 1.0
    assert result["prep_val_accuracy"] == 1.0
    out = tmp_path / "oracle"
    assert (out / "predictions.csv").exists() and (out / "coverage.csv").exists()
    assert json.loads((out / "metrics.json").read_text())["accuracy"] == 1.0


def test_compute_metrics_groups_by_profile_and_format():
    rows = [
        {"classe": T.FACTURE, "prediction": T.FACTURE, "confiance": 0.9, "correct": True,
         "profil": "photo", "format": "pdf", "latence_s": 0.1},
        {"classe": T.AUTRE, "prediction": T.FACTURE, "confiance": 0.4, "correct": False,
         "profil": "scan", "format": "image", "latence_s": 0.2},
    ]  # fmt: skip
    result, curve = compute_metrics(rows, target_accuracy=1.0)
    assert result["accuracy_photo"] == 1.0 and result["accuracy_scan"] == 0.0
    assert result["accuracy_pdf"] == 1.0 and result["accuracy_image"] == 0.0
    assert result["coverage_at_api_threshold"] == 0.5
    assert len(curve) == 101


def test_comparison_table():
    table = comparison_table({"a": {"accuracy": 0.5, "f1_macro": 0.4}, "b": {}})
    assert table.splitlines()[0].startswith("| Modèle | Accuracy")
    assert "| a | 50.0% | 0.400 |" in table
    assert table.count("—") > 5
