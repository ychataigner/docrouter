"""Modèles comparés sur le même jeu de test.

Deux familles :
- **zero-shot** : le modèle est utilisé tel quel, sans voir nos données (DiT pré-entraîné sur
  RVL-CDIP, CLIP qui compare l'image à des descriptions textuelles des classes) ;
- **sonde linéaire** (`probe-*`) : le réseau est gelé et sert d'extracteur de caractéristiques ;
  seule une régression logistique à 4 classes est entraînée sur `train` (choix de la
  régularisation sur `val`). Quelques secondes de calcul sur CPU une fois les vecteurs extraits.

torch / transformers / scikit-learn sont importés à la demande : `uv sync --group train`.
"""

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from docrouter_api.classifier import ImageEmbedder
from docrouter_api.head import LinearHead
from docrouter_core import DocumentType
from PIL import Image

from docrouter_training.eval.dataset import Sample, load_image
from docrouter_training.eval.metrics import CLASSES

logger = logging.getLogger(__name__)

DIT_ID = "microsoft/dit-base-finetuned-rvlcdip"
VIT_ID = "google/vit-base-patch16-224"
CLIP_ID = "openai/clip-vit-base-patch32"

Scores = dict[DocumentType, float]


class Backend(Protocol):
    name: str

    def params(self) -> dict[str, str]: ...

    def size(self) -> dict[str, float]: ...

    def prepare(self, train: Sequence[Sample], val: Sequence[Sample]) -> dict[str, float]: ...

    def predict(self, image: Image.Image) -> Scores: ...


def _size_of(*modules) -> dict[str, float]:
    n = sum(p.numel() for m in modules for p in m.parameters())
    return {"params_millions": n / 1e6, "size_mb_fp32": n * 4 / 1e6}


# --- Zero-shot ------------------------------------------------------------------------------


class DiTZeroShot:
    """Le classifieur actuellement en production dans l'API (même code)."""

    name = "dit-zeroshot"

    def __init__(self, model_id: str = DIT_ID) -> None:
        from docrouter_api.classifier import DiTClassifier

        self.model_id = model_id
        self._clf = DiTClassifier(model_id)

    def params(self) -> dict[str, str]:
        return {"model_id": self.model_id, "method": "zero-shot (RVL-CDIP : invoice → facture)"}

    def size(self) -> dict[str, float]:
        return _size_of(self._clf._model)

    def prepare(self, train, val) -> dict[str, float]:
        return {}

    def predict(self, image: Image.Image) -> Scores:
        return self._clf.predict(image).scores


CLIP_PROMPTS: dict[DocumentType, list[str]] = {
    DocumentType.FACTURE: [
        "a scanned invoice",
        "an invoice with a table of items, VAT and a total amount due",
        "a business invoice document",
    ],
    DocumentType.NOTE_DE_FRAIS: [
        "a photo of a shop receipt",
        "a restaurant receipt",
        "a narrow printed till receipt",
        "a fuel or parking receipt",
    ],
    DocumentType.BON_DE_LIVRAISON: [
        "a delivery note",
        "a packing slip listing delivered items and quantities",
        "a shipping document with a signature box",
    ],
    DocumentType.AUTRE: [
        "a letter",
        "an administrative form",
        "a certificate",
        "meeting minutes",
    ],
}


class CLIPZeroShot:
    """CLIP compare l'image à des phrases décrivant chaque classe : aucun entraînement."""

    name = "clip-zeroshot"

    def __init__(self, model_id: str = CLIP_ID) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.model_id = model_id
        self._torch = torch
        self._model = CLIPModel.from_pretrained(model_id).eval()
        self._processor = CLIPProcessor.from_pretrained(model_id)
        with torch.inference_mode():
            class_embeddings = []
            for cls in CLASSES:
                inputs = self._processor(text=CLIP_PROMPTS[cls], return_tensors="pt", padding=True)
                emb = _as_tensor(self._model.get_text_features(**inputs))
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(dim=0)
                class_embeddings.append(mean / mean.norm())
            self._text = torch.stack(class_embeddings)

    def params(self) -> dict[str, str]:
        return {"model_id": self.model_id, "method": "zero-shot (prompts texte)"}

    def size(self) -> dict[str, float]:
        return _size_of(self._model)

    def prepare(self, train, val) -> dict[str, float]:
        return {}

    def predict(self, image: Image.Image) -> Scores:
        with self._torch.inference_mode():
            inputs = self._processor(images=image, return_tensors="pt")
            emb = _as_tensor(self._model.get_image_features(**inputs))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = self._model.logit_scale.exp() * emb @ self._text.T
            probs = logits.softmax(dim=-1)[0].tolist()
        return dict(zip(CLASSES, probs, strict=True))


def _as_tensor(output):
    """Selon la version de transformers, get_*_features renvoie un tenseur ou une sortie."""
    return output if hasattr(output, "norm") else output.pooler_output


# --- Sondes linéaires -----------------------------------------------------------------------


def head_from_sklearn(pipeline, kind: str, backbone: str) -> LinearHead:
    """Convertit le pipeline scikit-learn (StandardScaler + LogisticRegression) en tête numpy."""
    scaler, logreg = pipeline[0], pipeline[-1]
    return LinearHead(
        backbone=backbone,
        kind=kind,
        classes=[CLASSES[i].value for i in logreg.classes_],
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coef=logreg.coef_.astype(np.float32),
        intercept=logreg.intercept_.astype(np.float32),
    )


class LinearProbe:
    def __init__(
        self,
        name: str,
        kind: str,
        model_id: str,
        cache_dir: Path | None = None,
        save_dir: Path | None = None,
        batch_size: int = 16,
    ) -> None:
        self.name, self.kind, self.model_id = name, kind, model_id
        self.cache_dir, self.save_dir, self.batch_size = cache_dir, save_dir, batch_size
        self._extractor = ImageEmbedder(kind, model_id)
        self._head: LinearHead | None = None
        self._chosen_c: float | None = None
        self.saved_path: Path | None = None

    def params(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "method": "sonde linéaire (backbone gelé + régression logistique)",
            "probe_C": str(self._chosen_c),
        }

    def size(self) -> dict[str, float]:
        return _size_of(self._extractor.model)

    def _features(self, samples: Sequence[Sample], split: str) -> np.ndarray:
        cache = self.cache_dir / f"{self.name}_{split}.npy" if self.cache_dir else None
        if cache and cache.exists():
            feats = np.load(cache)
            if len(feats) == len(samples):
                return feats
        chunks = []
        for start in range(0, len(samples), self.batch_size):
            batch = [load_image(s.path) for s in samples[start : start + self.batch_size]]
            chunks.append(self._extractor.embed(batch))
            logger.info("%s %s : %d/%d", self.name, split, start + len(batch), len(samples))
        feats = np.concatenate(chunks)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, feats)
        return feats

    def prepare(self, train: Sequence[Sample], val: Sequence[Sample]) -> dict[str, float]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        t0 = time.perf_counter()
        x_train, x_val = self._features(train, "train"), self._features(val, "val")
        extract_s = time.perf_counter() - t0
        y_train = np.array([CLASSES.index(s.label) for s in train])
        y_val = np.array([CLASSES.index(s.label) for s in val])

        t0 = time.perf_counter()
        best = None
        for c in (0.001, 0.01, 0.1, 1.0, 10.0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
            clf.fit(x_train, y_train)
            score = float((clf.predict(x_val) == y_val).mean())
            logger.info("%s C=%g : accuracy val %.3f", self.name, c, score)
            if best is None or score > best[0]:
                best = (score, c, clf)
        fit_s = time.perf_counter() - t0
        val_accuracy, self._chosen_c, pipeline = best
        # La tête est exportée en numpy et c'est elle qui prédit, ici comme dans l'API.
        self._head = head_from_sklearn(pipeline, self.kind, self.model_id)

        if self.save_dir:
            self.saved_path = self.save_dir / f"{self.name}.npz"
            self._head.save(self.saved_path)
        return {
            "val_accuracy": val_accuracy,
            "feature_extraction_s": extract_s,
            "head_training_s": fit_s,
        }

    def predict(self, image: Image.Image) -> Scores:
        probs = self._head.probabilities(self._extractor.embed([image]))[0]
        return {DocumentType(c): float(p) for c, p in zip(self._head.classes, probs, strict=True)}


def make_backend(name: str, cache_dir: Path | None, save_dir: Path | None) -> Backend:
    factories: dict[str, Callable[[], Backend]] = {
        "dit-zeroshot": DiTZeroShot,
        "clip-zeroshot": CLIPZeroShot,
        "probe-dit": lambda: LinearProbe("probe-dit", "dit", DIT_ID, cache_dir, save_dir),
        "probe-vit": lambda: LinearProbe("probe-vit", "vit", VIT_ID, cache_dir, save_dir),
        "probe-clip": lambda: LinearProbe("probe-clip", "clip", CLIP_ID, cache_dir, save_dir),
    }
    return factories[name]()


BACKEND_NAMES = ("dit-zeroshot", "clip-zeroshot", "probe-dit", "probe-vit", "probe-clip")
