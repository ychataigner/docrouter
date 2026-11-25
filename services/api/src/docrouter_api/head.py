"""Tête de classification linéaire (la « sonde ») : quelques milliers de nombres, en numpy pur.

Elle est entraînée dans `training/` avec scikit-learn, puis exportée au format `.npz` **sans
pickle** : le fichier ne peut contenir que des tableaux, jamais du code. On peut donc le charger
sans risque, y compris depuis le Hub Hugging Face, et l'API n'a pas besoin de scikit-learn.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LinearHead:
    backbone: str  # modèle Hugging Face qui produit les vecteurs (ex. openai/clip-vit-base-patch32)
    kind: str  # façon d'extraire le vecteur : "clip", "vit" ou "dit"
    classes: list[str]
    mean: np.ndarray  # normalisation des vecteurs (StandardScaler)
    scale: np.ndarray
    coef: np.ndarray  # régression logistique multinomiale : (n_classes, dim)
    intercept: np.ndarray

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """(n, dim) → (n, n_classes), chaque ligne somme à 1."""
        z = ((features - self.mean) / self.scale) @ self.coef.T + self.intercept
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez(
                f,
                backbone=np.array(self.backbone),
                kind=np.array(self.kind),
                classes=np.array(self.classes),
                mean=self.mean,
                scale=self.scale,
                coef=self.coef,
                intercept=self.intercept,
            )

    @classmethod
    def load(cls, path: Path) -> "LinearHead":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                backbone=str(data["backbone"]),
                kind=str(data["kind"]),
                classes=[str(c) for c in data["classes"]],
                mean=data["mean"].astype(np.float32),
                scale=data["scale"].astype(np.float32),
                coef=data["coef"].astype(np.float32),
                intercept=data["intercept"].astype(np.float32),
            )
