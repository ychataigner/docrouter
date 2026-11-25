"""Réseau de vision exporté en ONNX, exécuté par onnxruntime : ni PyTorch ni transformers.

Le dossier d'export (`docrouter-export-onnx`) contient :
- `model.onnx` (fp32) et `model_int8.onnx` (quantifié) : image normalisée → vecteur ;
- `preprocessor_config.json` : le prétraitement du modèle d'origine (config Hugging Face),
  reproduit ici en PIL + numpy ;
- `export_info.json` : le modèle d'origine et le type d'extracteur, pour vérifier que la tête
  de classification va bien avec ce réseau.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class ImagePreprocessor:
    """Équivalent numpy des « image processors » Hugging Face (CLIP, ViT, BEiT/DiT)."""

    resample: int
    shortest_edge: int | None  # redimensionne le petit côté (CLIP) ...
    size: tuple[int, int] | None  # ... ou à une taille fixe (hauteur, largeur) (ViT, DiT)
    crop: tuple[int, int] | None  # recadrage central (hauteur, largeur)
    rescale: float
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_config(cls, cfg: dict) -> "ImagePreprocessor":
        size = cfg.get("size", {})
        crop = cfg.get("crop_size") if cfg.get("do_center_crop") else None
        return cls(
            resample=int(cfg.get("resample", Image.Resampling.BICUBIC)),
            shortest_edge=size.get("shortest_edge"),
            size=(size["height"], size["width"]) if "height" in size else None,
            crop=(crop["height"], crop["width"]) if crop else None,
            rescale=float(cfg.get("rescale_factor", 1 / 255)) if cfg.get("do_rescale", True) else 1,
            mean=np.asarray(cfg.get("image_mean", [0.5] * 3), dtype=np.float32),
            std=np.asarray(cfg.get("image_std", [0.5] * 3), dtype=np.float32),
        )

    def __call__(self, image: Image.Image) -> np.ndarray:
        """Image PIL → tableau (3, H, W) float32, prêt pour le réseau."""
        image = image.convert("RGB")
        if self.shortest_edge:
            w, h = image.size
            short, long = (w, h) if w <= h else (h, w)
            new_long = int(self.shortest_edge * long / short)
            new_w, new_h = (
                (self.shortest_edge, new_long) if w <= h else (new_long, self.shortest_edge)
            )
            image = image.resize((new_w, new_h), resample=self.resample)
        elif self.size:
            image = image.resize((self.size[1], self.size[0]), resample=self.resample)
        if self.crop:
            ch, cw = self.crop
            w, h = image.size
            top, left = (h - ch) // 2, (w - cw) // 2
            image = image.crop((left, top, left + cw, top + ch))
        array = np.asarray(image, dtype=np.float32) * self.rescale
        array = (array - self.mean) / self.std
        return array.transpose(2, 0, 1)


class OnnxEmbedder:
    """Même rôle que `ImageEmbedder` (PyTorch), avec le modèle ONNX."""

    def __init__(self, model_path: Path, preprocessor_config: Path, export_info: Path) -> None:
        import onnxruntime as ort

        self.preprocess = ImagePreprocessor.from_config(
            json.loads(preprocessor_config.read_text(encoding="utf-8"))
        )
        info = json.loads(export_info.read_text(encoding="utf-8"))
        self.kind, self.model_id = info["kind"], info["model_id"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = np.stack([self.preprocess(image) for image in images])
        return self._session.run(None, {self._input: batch})[0]
