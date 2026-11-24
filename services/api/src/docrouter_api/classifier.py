"""Classifieurs de documents.

L'API ne dépend que de l'interface `DocumentClassifier` : remplacer DiT par le modèle fine-tuné
(itération 2) puis par sa version ONNX (itération 4) ne touche ni aux routes ni au front.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol

from docrouter_core import DocumentType
from PIL import Image

logger = logging.getLogger(__name__)

# DiT pré-entraîné ne connaît que les 16 classes RVL-CDIP. En attendant le fine-tuning,
# on regroupe : `invoice` → facture, tout le reste → autre. Il ne sait donc pas encore
# reconnaître les notes de frais ni les bons de livraison.
RVLCDIP_TO_DOCTYPE: dict[str, DocumentType] = {"invoice": DocumentType.FACTURE}


@dataclass
class Prediction:
    scores: dict[DocumentType, float]
    raw_label: str | None = None

    @property
    def best(self) -> tuple[DocumentType, float]:
        doc_type = max(self.scores, key=self.scores.__getitem__)
        return doc_type, self.scores[doc_type]


class DocumentClassifier(Protocol):
    name: str

    def predict(self, image: Image.Image) -> Prediction: ...


def aggregate_scores(
    label_probs: dict[str, float], mapping: dict[str, DocumentType]
) -> dict[DocumentType, float]:
    """Additionne les probabilités des labels du modèle dans nos 4 classes."""
    scores = dict.fromkeys(DocumentType, 0.0)
    for label, prob in label_probs.items():
        scores[mapping.get(label.lower(), DocumentType.AUTRE)] += prob
    return scores


class DiTClassifier:
    def __init__(self, model_id: str) -> None:
        # Imports locaux : torch/transformers ne sont chargés que si ce classifieur est utilisé.
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        logger.info("Chargement du modèle %s", model_id)
        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageClassification.from_pretrained(model_id).eval()
        self.name = model_id

    def predict(self, image: Image.Image) -> Prediction:
        inputs = self._processor(images=image, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        probs = logits.softmax(dim=-1)[0].tolist()
        id2label = self._model.config.id2label
        label_probs = {id2label[i]: p for i, p in enumerate(probs)}
        raw_label = max(label_probs, key=label_probs.__getitem__)
        return Prediction(
            scores=aggregate_scores(label_probs, RVLCDIP_TO_DOCTYPE), raw_label=raw_label
        )


class DummyClassifier:
    """Classifieur simulé, sans IA : pour développer et tester le front.

    Sans scores imposés, la prédiction est tirée de l'empreinte de l'image : un même document
    donne toujours le même résultat, et des documents différents couvrent toutes les classes
    ainsi que des cas incertains (→ `a_verifier`).
    """

    name = "dummy"

    def __init__(self, scores: dict[DocumentType, float] | None = None) -> None:
        self._scores = scores

    def predict(self, image: Image.Image) -> Prediction:
        if self._scores is not None:
            return Prediction(scores=dict(self._scores), raw_label="dummy")
        digest = hashlib.sha256(image.tobytes()).digest()
        types = list(DocumentType)
        chosen = types[digest[0] % len(types)]
        confidence = 0.40 + 0.55 * digest[1] / 255  # entre 40 % et 95 %
        rest = (1 - confidence) / (len(types) - 1)
        scores = {t: confidence if t == chosen else rest for t in types}
        return Prediction(scores=scores, raw_label="simulation")


def build_classifier(kind: str, model_id: str) -> DocumentClassifier:
    if kind == "dummy":
        return DummyClassifier()
    try:
        return DiTClassifier(model_id)
    except ImportError as exc:
        raise RuntimeError(
            "Dépendances IA absentes : installer l'extra `ml` "
            "(`uv sync --group ml`, ou build Docker avec WITH_ML=true)."
        ) from exc
