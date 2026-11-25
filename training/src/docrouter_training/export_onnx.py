"""Export du réseau de vision de la sonde en ONNX, puis quantization dynamique INT8.

    uv run docrouter-export-onnx                       # CLIP → models/onnx/clip/

Produit `model.onnx` (fp32), `model_int8.onnx` (poids en entiers 8 bits : ≈ 4× plus petit),
`preprocessor_config.json` et `export_info.json`, puis vérifie sur des documents de test que :
- le prétraitement numpy de l'API donne les mêmes pixels que celui de Hugging Face ;
- les vecteurs ONNX fp32 et INT8 restent proches de ceux de PyTorch (similarité cosinus).
Précision de bout en bout : `docrouter-evaluate --model probe-clip-onnx probe-clip-int8`.
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from docrouter_training.eval.backends import CLIP_ID
from docrouter_training.eval.dataset import load_image, read_manifest

logger = logging.getLogger("docrouter.export")


def _wrapper(embedder):
    """Module PyTorch : pixels normalisés → vecteur, avec la même sortie que `ImageEmbedder`."""
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = embedder.model

        def forward(self, pixel_values):
            out = self.model(pixel_values=pixel_values)
            if embedder.kind == "clip":
                return out.image_embeds
            if embedder.kind == "vit":
                return out.last_hidden_state[:, 0]
            return out.pooler_output

    return Wrapper().eval()


def export(kind: str, model_id: str, out: Path) -> dict:
    import onnx
    import torch
    from docrouter_api.classifier import ImageEmbedder
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.mkdir(parents=True, exist_ok=True)
    embedder = ImageEmbedder(kind, model_id)
    embedder.processor.save_pretrained(out)  # → preprocessor_config.json
    size = embedder.processor.crop_size if kind == "clip" else embedder.processor.size
    dummy = torch.randn(2, 3, size["height"], size["width"])

    t0 = time.perf_counter()
    torch.onnx.export(
        _wrapper(embedder),
        (dummy,),
        str(out / "model.onnx"),
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_shapes={"pixel_values": {0: torch.export.Dim("batch")}},
        dynamo=True,
        external_data=False,
    )
    export_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Réglage choisi sur mesures (docs/iterations.md, itération 4) :
    # - par défaut (une échelle par matrice) : 90,8 % au lieu de 97,5 % ;
    # - per_channel seul : modèle cassé (25 %) : sur les CPU x86 sans VNNI, les produits
    #   8 bits × 8 bits saturent ;
    # - per_channel + reduce_range (poids sur 7 bits) : 97,5 %, aucune perte.
    quantize_dynamic(
        out / "model.onnx",
        out / "model_int8.onnx",
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    quantize_s = time.perf_counter() - t0

    info = {
        "model_id": model_id,
        "kind": kind,
        "opset": onnx.load(out / "model.onnx", load_external_data=False).opset_import[0].version,
        "torch": torch.__version__,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "size_mb_fp32": round((out / "model.onnx").stat().st_size / 1e6, 1),
        "size_mb_int8": round((out / "model_int8.onnx").stat().st_size / 1e6, 1),
        "export_s": round(export_s, 1),
        "quantize_s": round(quantize_s, 1),
        "quantization": "dynamique INT8, per_channel, reduce_range",
    }
    (out / "export_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def verify(kind: str, model_id: str, out: Path, data: Path, n: int = 24) -> dict:
    from docrouter_api.classifier import ImageEmbedder
    from docrouter_api.onnx_embedder import OnnxEmbedder

    samples = read_manifest(data, "test")[:: max(1, len(read_manifest(data, "test")) // n)][:n]
    images = [load_image(s.path) for s in samples]
    torch_embedder = ImageEmbedder(kind, model_id)

    hf_pixels = torch_embedder.processor(images=images, return_tensors="np")["pixel_values"]
    onnx_fp32 = OnnxEmbedder(
        out / "model.onnx", out / "preprocessor_config.json", out / "export_info.json"
    )
    our_pixels = np.stack([onnx_fp32.preprocess(im) for im in images])
    reference = torch_embedder.embed(images)
    onnx_int8 = OnnxEmbedder(
        out / "model_int8.onnx", out / "preprocessor_config.json", out / "export_info.json"
    )
    cos_fp32 = _cosine(reference, onnx_fp32.embed(images))
    cos_int8 = _cosine(reference, onnx_int8.embed(images))
    return {
        "n_images": len(images),
        "pixels_max_abs_diff": float(np.abs(hf_pixels - our_pixels).max()),
        "cosine_fp32_min": float(cos_fp32.min()),
        "cosine_int8_mean": float(cos_int8.mean()),
        "cosine_int8_min": float(cos_int8.min()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kind", default="clip", choices=["clip", "vit", "dit"])
    parser.add_argument("--model-id", default=CLIP_ID)
    parser.add_argument("--out", type=Path, default=Path("models/onnx/clip"))
    parser.add_argument("--data", type=Path, default=Path("data/synth"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    info = export(args.kind, args.model_id, args.out)
    logger.info("Export : %s", info)
    if (args.data / "manifest.csv").exists():
        checks = verify(args.kind, args.model_id, args.out, args.data)
        logger.info("Vérifications : %s", checks)
        info["checks"] = checks
        (args.out / "export_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
