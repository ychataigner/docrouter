"""Chemin ONNX de l'API (production) : prétraitement numpy, onnxruntime, cohérence avec la tête.

Un mini-modèle ONNX (moyenne des pixels par canal) remplace CLIP : pas de téléchargement en CI.
Le test marqué `ml` compare le prétraitement numpy à celui de Hugging Face sur le vrai CLIP.
"""

import json

import numpy as np
import onnx
import pytest
from docrouter_api.classifier import ProbeClassifier, build_classifier
from docrouter_api.head import LinearHead
from docrouter_api.onnx_embedder import ImagePreprocessor, OnnxEmbedder
from docrouter_core import DocumentType
from onnx import TensorProto, helper
from PIL import Image

CLIP_LIKE = {
    "do_resize": True,
    "size": {"shortest_edge": 32},
    "do_center_crop": True,
    "crop_size": {"height": 32, "width": 32},
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.25, 0.25, 0.25],
    "resample": 3,
}


def test_preprocessor_resizes_short_side_then_crops_center():
    pre = ImagePreprocessor.from_config(CLIP_LIKE)
    out = pre(Image.new("RGB", (60, 120), "white"))  # portrait A4-like
    assert out.shape == (3, 32, 32) and out.dtype == np.float32
    np.testing.assert_allclose(out, (1 - 0.5) / 0.25, rtol=1e-6)  # blanc → 2,0


def test_preprocessor_fixed_size_without_crop():
    cfg = {"size": {"height": 24, "width": 16}, "image_mean": [0, 0, 0], "image_std": [1, 1, 1]}
    out = ImagePreprocessor.from_config(cfg)(Image.new("RGB", (100, 50), (255, 0, 0)))
    assert out.shape == (3, 24, 16)
    assert out[0].min() == pytest.approx(1.0) and out[1].max() == pytest.approx(0.0)


def test_preprocessor_center_crop_keeps_the_middle():
    image = Image.new("RGB", (32, 96), "black")
    image.paste((255, 255, 255), (0, 32, 32, 64))  # bande blanche au milieu
    out = ImagePreprocessor.from_config(CLIP_LIKE)(image)
    assert out.mean() == pytest.approx(2.0, abs=0.05)


@pytest.fixture
def onnx_dir(tmp_path):
    """Mini « réseau » : vecteur = moyenne de chaque canal (3 dimensions)."""
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["pixel_values"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["embeddings"]),
        ],
        "mini",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, ["batch", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, tmp_path / "model.onnx")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(CLIP_LIKE))
    (tmp_path / "export_info.json").write_text(
        json.dumps({"model_id": "fake/clip", "kind": "clip"})
    )
    return tmp_path


def test_onnx_embedder_runs_without_pytorch(onnx_dir):
    embedder = OnnxEmbedder(
        onnx_dir / "model.onnx",
        onnx_dir / "preprocessor_config.json",
        onnx_dir / "export_info.json",
    )
    feats = embedder.embed([Image.new("RGB", (40, 80), "white"), Image.new("RGB", (40, 80))])
    assert feats.shape == (2, 3)
    np.testing.assert_allclose(feats[0], 2.0, rtol=1e-5)  # blanc
    np.testing.assert_allclose(feats[1], -2.0, rtol=1e-5)  # noir
    assert (embedder.kind, embedder.model_id) == ("clip", "fake/clip")


def _head(tmp_path, backbone="fake/clip"):
    # Vecteur clair (> 0) → facture ; sombre (< 0) → autre.
    head = LinearHead(
        backbone=backbone,
        kind="clip",
        classes=[c.value for c in DocumentType],
        mean=np.zeros(3, np.float32),
        scale=np.ones(3, np.float32),
        coef=np.array([[3, 3, 3], [0, 0, 0], [0, 0, 0], [-3, -3, -3]], np.float32),
        intercept=np.zeros(4, np.float32),
    )
    head.save(tmp_path / "head.npz")
    return str(tmp_path / "head.npz")


def test_probe_classifier_with_onnx(onnx_dir, tmp_path):
    clf = build_classifier("probe", "unused", _head(tmp_path), str(onnx_dir / "model.onnx"))
    assert isinstance(clf, ProbeClassifier)
    assert clf.name == "probe-clip (fake/clip, onnx model)"
    assert clf.predict(Image.new("RGB", (50, 70), "white")).best[0] == DocumentType.FACTURE
    assert clf.predict(Image.new("RGB", (50, 70), "black")).best[0] == DocumentType.AUTRE


def test_head_and_onnx_model_must_match(onnx_dir, tmp_path):
    with pytest.raises(ValueError, match="ne correspond pas"):
        ProbeClassifier(_head(tmp_path, backbone="other/model"), str(onnx_dir / "model.onnx"))


def test_missing_onnx_model_explains_how_to_export(tmp_path):
    with pytest.raises(FileNotFoundError, match="docrouter-export-onnx"):
        ProbeClassifier(_head(tmp_path), str(tmp_path / "absent" / "model_int8.onnx"))


@pytest.mark.ml
def test_numpy_preprocessing_matches_hugging_face_for_clip():
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ours = ImagePreprocessor.from_config(processor.to_dict())
    rng = np.random.default_rng(0)
    for size in [(1240, 1754), (576, 1400), (1600, 1200)]:
        array = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        image = Image.fromarray(array)
        expected = processor(images=image, return_tensors="np")["pixel_values"][0]
        np.testing.assert_allclose(ours(image), expected, atol=1e-5)
