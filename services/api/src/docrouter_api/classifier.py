"""Classifieurs de documents.

L'API ne dépend que de l'interface `DocumentClassifier` : passer de DiT à la sonde CLIP, puis à
sa version ONNX (itération 4), ne touche ni aux routes ni au front.

- `probe` : réseau de vision gelé + tête linéaire à 4 classes entraînée sur nos documents
  (choisi sur mesures : 97,5 % sur le jeu de test, voir docs/iterations.md) ;
- `dit` : DiT pré-entraîné sur RVL-CDIP, sans adaptation (35 %) ;
- `dummy` : tri simulé, sans IA, pour développer le front.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
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


class ImageEmbedder:
    """Réseau de vision gelé : une image → un vecteur qui résume le document.

    Utilisé à l'identique par l'entraînement de la tête (`training/`) et par l'API : les vecteurs
    vus en production sont calculés exactement comme ceux de l'entraînement.
    """

    def __init__(self, kind: str, model_id: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

        self.kind, self.model_id, self._torch = kind, model_id, torch
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        if kind == "clip":
            # Seule la partie « image » de CLIP est chargée (pas l'encodeur de texte).
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_id).eval()
        elif kind in ("vit", "dit"):
            self.model = AutoModel.from_pretrained(model_id).eval()
        else:
            raise ValueError(f"Type d'extracteur inconnu : {kind}")

    def embed(self, images: list[Image.Image]):
        """Renvoie un tableau numpy (n_images, dim)."""
        with self._torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            out = self.model(**inputs)
            if self.kind == "clip":
                feats = out.image_embeds
            elif self.kind == "vit":
                feats = out.last_hidden_state[:, 0]  # jeton [CLS]
            else:  # dit : moyenne des patchs (pooler de BEiT)
                pooled = out.pooler_output
                feats = pooled if pooled is not None else out.last_hidden_state[:, 1:].mean(1)
        return feats.float().numpy()


def resolve_file(spec: str, hint: str = "") -> Path:
    """Fichier local, ou `hf://<compte>/<dépôt>/<chemin>` téléchargé depuis le Hub (HF_TOKEN)."""
    if spec.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        owner, repo, filename = spec.removeprefix("hf://").split("/", 2)
        return Path(hf_hub_download(f"{owner}/{repo}", filename))
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {spec}" + (f" ({hint})" if hint else ""))
    return path


def sibling(spec: str, name: str) -> str:
    """Fichier voisin d'un autre (même dossier local ou même dossier du dépôt Hub)."""
    return spec.rsplit("/", 1)[0] + "/" + name if "/" in spec else name


class ProbeClassifier:
    """Réseau gelé + tête linéaire à 4 classes (fichier `.npz`, voir `head.py`).

    Le réseau tourne avec onnxruntime si `onnx_model` est donné (production), sinon avec PyTorch.
    """

    def __init__(self, head_spec: str, onnx_model: str | None = None) -> None:
        from docrouter_api.head import LinearHead

        head = LinearHead.load(
            resolve_file(
                head_spec, "la produire avec `uv run docrouter-evaluate --model probe-clip`"
            )
        )
        self._head = head
        self._classes = [DocumentType(c) for c in head.classes]
        if onnx_model:
            from docrouter_api.onnx_embedder import OnnxEmbedder

            hint = "le produire avec `uv run docrouter-export-onnx`"
            onnx_model = onnx_model.replace("\\", "/")
            self._embedder = OnnxEmbedder(
                resolve_file(onnx_model, hint),
                resolve_file(sibling(onnx_model, "preprocessor_config.json"), hint),
                resolve_file(sibling(onnx_model, "export_info.json"), hint),
            )
            if (self._embedder.kind, self._embedder.model_id) != (head.kind, head.backbone):
                raise ValueError(
                    f"Le modèle ONNX ({self._embedder.model_id}) ne correspond pas à la tête "
                    f"({head.backbone}) : ré-exporter ou ré-entraîner."
                )
            runtime = f"onnx {Path(onnx_model).stem}"
        else:
            self._embedder = ImageEmbedder(head.kind, head.backbone)
            runtime = "pytorch"
        logger.info("Sonde %s sur %s (%s)", head.kind, head.backbone, runtime)
        self.name = f"probe-{head.kind} ({head.backbone}, {runtime})"

    def predict(self, image: Image.Image) -> Prediction:
        probs = self._head.probabilities(self._embedder.embed([image]))[0]
        scores = {c: float(p) for c, p in zip(self._classes, probs, strict=True)}
        best = max(scores, key=scores.__getitem__)
        return Prediction(scores=scores, raw_label=best.value)


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


def build_classifier(
    kind: str, model_id: str, probe_head: str = "", onnx_model: str | None = None
) -> DocumentClassifier:
    """`onnx_model` vide → la sonde tourne avec PyTorch (modèle lourd de référence)."""
    if kind == "dummy":
        return DummyClassifier()
    try:
        if kind == "probe":
            return ProbeClassifier(probe_head, onnx_model or None)
        return DiTClassifier(model_id)
    except ImportError as exc:
        raise RuntimeError(
            "Dépendances IA absentes : installer l'extra `onnx` (production, onnxruntime) "
            "ou `ml` (PyTorch) — `uv sync --group ml`, ou build Docker avec ML_EXTRA=onnx|ml."
        ) from exc
