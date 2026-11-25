"""Sonde : tête linéaire numpy (format .npz sans pickle) et chargement dans l'API.

Sans réseau ni PyTorch : l'extracteur d'images est remplacé par un faux qui renvoie des vecteurs.
"""

import numpy as np
import pytest
from docrouter_api.classifier import ProbeClassifier, build_classifier, resolve_head
from docrouter_api.head import LinearHead
from docrouter_api.settings import Settings
from docrouter_core import DocumentType
from PIL import Image

CLASSES = [c.value for c in DocumentType]


def make_head(dim: int = 8) -> LinearHead:
    rng = np.random.default_rng(0)
    return LinearHead(
        backbone="fake/backbone",
        kind="clip",
        classes=CLASSES,
        mean=rng.normal(size=dim).astype(np.float32),
        scale=rng.uniform(0.5, 2, size=dim).astype(np.float32),
        coef=rng.normal(size=(len(CLASSES), dim)).astype(np.float32),
        intercept=rng.normal(size=len(CLASSES)).astype(np.float32),
    )


def test_probabilities_are_a_softmax_of_the_linear_scores():
    head = make_head()
    x = np.random.default_rng(1).normal(size=(5, 8)).astype(np.float32)
    probs = head.probabilities(x)
    assert probs.shape == (5, len(CLASSES))
    np.testing.assert_allclose(probs.sum(axis=1), 1, rtol=1e-6)
    z = ((x - head.mean) / head.scale) @ head.coef.T + head.intercept
    np.testing.assert_array_equal(probs.argmax(axis=1), z.argmax(axis=1))


def test_probabilities_are_stable_for_large_scores():
    head = make_head()
    probs = head.probabilities(np.full((1, 8), 1e4, dtype=np.float32))
    assert np.isfinite(probs).all()


def test_save_and_load_roundtrip_without_pickle(tmp_path):
    head = make_head()
    path = tmp_path / "head.npz"
    head.save(path)
    loaded = LinearHead.load(path)
    assert (loaded.backbone, loaded.kind, loaded.classes) == ("fake/backbone", "clip", CLASSES)
    x = np.ones((2, 8), dtype=np.float32)
    np.testing.assert_allclose(loaded.probabilities(x), head.probabilities(x), rtol=1e-6)
    with np.load(path, allow_pickle=False) as data:  # aucun objet Python dans le fichier
        assert all(data[k].dtype != object for k in data.files)


def test_resolve_head_missing_file_explains_how_to_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="docrouter-evaluate"):
        resolve_head(str(tmp_path / "absent.npz"))
    (tmp_path / "ok.npz").write_bytes(b"x")
    assert resolve_head(str(tmp_path / "ok.npz")) == tmp_path / "ok.npz"


class FakeEmbedder:
    """Faux réseau : le « vecteur » de l'image est la valeur de son 1er pixel."""

    def __init__(self, kind, model_id):
        self.kind, self.model_id = kind, model_id

    def embed(self, images):
        return np.array([[float(im.getpixel((0, 0))[0])] for im in images], dtype=np.float32)


def test_probe_classifier_predicts_with_the_head(tmp_path, monkeypatch):
    # 1 seule dimension : pixel clair → facture (0,1 × 250 = 25), pixel noir → autre (biais 3).
    head = LinearHead(
        backbone="fake/backbone",
        kind="clip",
        classes=CLASSES,
        mean=np.zeros(1, np.float32),
        scale=np.ones(1, np.float32),
        coef=np.array([[0.1], [0.0], [0.0], [0.0]], np.float32),
        intercept=np.array([0.0, 0.0, 0.0, 3.0], np.float32),
    )
    head.save(tmp_path / "probe.npz")
    monkeypatch.setattr("docrouter_api.classifier.ImageEmbedder", FakeEmbedder)

    clf = build_classifier("probe", "unused", str(tmp_path / "probe.npz"))
    assert isinstance(clf, ProbeClassifier)
    assert clf.name == "probe-clip (fake/backbone)"

    bright = clf.predict(Image.new("RGB", (10, 10), (250, 250, 250)))
    dark = clf.predict(Image.new("RGB", (10, 10), (0, 0, 0)))
    assert bright.best[0] == DocumentType.FACTURE and bright.best[1] > 0.9
    assert bright.raw_label == "facture"
    assert dark.best[0] == DocumentType.AUTRE
    assert sum(dark.scores.values()) == pytest.approx(1)


def test_expected_model_shown_by_health():
    assert Settings(classifier="probe", probe_head="x.npz").expected_model == "x.npz"
    assert Settings(classifier="dit").expected_model.startswith("microsoft/dit")
    assert Settings(classifier="dummy").expected_model == "dummy"
