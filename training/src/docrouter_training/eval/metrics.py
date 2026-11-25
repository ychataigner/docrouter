"""Métriques de classification, sans dépendance lourde (testées en CI)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from docrouter_core import DocumentType

CLASSES: tuple[DocumentType, ...] = tuple(DocumentType)


@dataclass
class ClassReport:
    precision: float
    recall: float
    f1: float
    support: int


def accuracy(y_true: Sequence[DocumentType], y_pred: Sequence[DocumentType]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def confusion_matrix(y_true: Sequence[DocumentType], y_pred: Sequence[DocumentType]) -> np.ndarray:
    """Lignes = vraie classe, colonnes = classe prédite, dans l'ordre de `CLASSES`."""
    index = {c: i for i, c in enumerate(CLASSES)}
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[index[t], index[p]] += 1
    return matrix


def per_class(
    y_true: Sequence[DocumentType], y_pred: Sequence[DocumentType]
) -> dict[DocumentType, ClassReport]:
    matrix = confusion_matrix(y_true, y_pred)
    reports = {}
    for i, cls in enumerate(CLASSES):
        tp = matrix[i, i]
        predicted, actual = matrix[:, i].sum(), matrix[i, :].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        reports[cls] = ClassReport(float(precision), float(recall), float(f1), int(actual))
    return reports


def f1_macro(y_true: Sequence[DocumentType], y_pred: Sequence[DocumentType]) -> float:
    """Moyenne des F1 des classes présentes dans la vérité terrain."""
    reports = [r for r in per_class(y_true, y_pred).values() if r.support]
    return sum(r.f1 for r in reports) / len(reports) if reports else 0.0


@dataclass
class CoveragePoint:
    threshold: float
    coverage: float  # part des documents rangés automatiquement (confiance ≥ seuil)
    accuracy: float  # précision sur ces documents-là (le reste part dans `a_verifier`)


def coverage_curve(
    correct: Sequence[bool], confidence: Sequence[float], steps: int = 100
) -> list[CoveragePoint]:
    """Compromis du seuil `a_verifier` : plus il est haut, moins on range, mais mieux on range."""
    correct_arr = np.asarray(correct, dtype=bool)
    conf_arr = np.asarray(confidence, dtype=float)
    points = []
    for k in range(steps + 1):
        threshold = k / steps
        kept = conf_arr >= threshold
        n_kept = int(kept.sum())
        points.append(
            CoveragePoint(
                threshold=threshold,
                coverage=n_kept / len(conf_arr) if len(conf_arr) else 0.0,
                accuracy=float(correct_arr[kept].mean()) if n_kept else 1.0,
            )
        )
    return points


def recommend_threshold(
    curve: Sequence[CoveragePoint], target_accuracy: float
) -> CoveragePoint | None:
    """Plus petit seuil dont les documents rangés atteignent la précision visée (couverture max)."""
    for point in curve:
        if point.coverage > 0 and point.accuracy >= target_accuracy:
            return point
    return None


def latency_stats(seconds: Sequence[float]) -> dict[str, float]:
    if not seconds:
        return {}
    ms = np.asarray(seconds) * 1000
    return {
        "latency_ms_mean": float(ms.mean()),
        "latency_ms_p50": float(np.percentile(ms, 50)),
        "latency_ms_p95": float(np.percentile(ms, 95)),
    }
