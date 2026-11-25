"""Dégradations « effet scan » et « effet photo » : un document trop propre ne ressemble pas aux
documents réels (scans de travers, photos de tickets prises au smartphone)."""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROFILES = ("propre", "scan", "photo")
# Probabilité de chaque profil : les tickets sont surtout photographiés, le reste surtout scanné.
WEIGHTS = {
    "note_de_frais": (0.10, 0.30, 0.60),
    "default": (0.25, 0.55, 0.20),
}


def pick_profile(doc_class: str, rng: random.Random) -> str:
    return rng.choices(PROFILES, weights=WEIGHTS.get(doc_class, WEIGHTS["default"]))[0]


def apply(img: Image.Image, profile: str, rng: random.Random) -> Image.Image:
    if profile == "scan":
        return scan(img, rng)
    if profile == "photo":
        return photo(img, rng)
    return img


def _np_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.getrandbits(32))


def _noise(img: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr += _np_rng(rng).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _tint(img: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Papier légèrement jauni / grisé (multiplication)."""
    arr = np.asarray(img, dtype=np.float32) * (np.array(color, dtype=np.float32) / 255)
    return Image.fromarray(arr.astype(np.uint8))


def _speckles(img: Image.Image, rng: random.Random, amount: float) -> Image.Image:
    arr = np.array(img)
    gen = _np_rng(rng)
    mask = gen.random(arr.shape[:2]) < amount
    arr[mask] = gen.integers(0, 90, size=(int(mask.sum()), 1))
    return Image.fromarray(arr)


def scan(img: Image.Image, rng: random.Random) -> Image.Image:
    paper = (rng.randint(235, 255), rng.randint(232, 255), rng.randint(220, 250))
    img = _tint(img, paper)
    img = img.rotate(
        rng.uniform(-2.5, 2.5), resample=Image.Resampling.BICUBIC, fillcolor=paper, expand=False
    )
    if rng.random() < 0.5:  # numérisation basse résolution
        w, h = img.size
        factor = rng.uniform(0.45, 0.8)
        img = img.resize((int(w * factor), int(h * factor))).resize((w, h))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.9)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.3))
    img = _noise(img, rng.uniform(2, 9), rng)
    if rng.random() < 0.4:
        img = _speckles(img, rng, rng.uniform(0.0002, 0.001))
    if rng.random() < 0.35:
        img = img.convert("L").convert("RGB")
    return img


def _perspective_coeffs(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list:
    """Coefficients de `Image.transform(PERSPECTIVE)` qui envoient `dst` sur `src`."""
    matrix = []
    for (x, y), (u, v) in zip(dst, src, strict=True):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b).tolist()


def photo(img: Image.Image, rng: random.Random) -> Image.Image:
    """Document posé sur une table et photographié : perspective, fond, ombre, éclairage."""
    w, h = img.size
    margin = int(max(w, h) * rng.uniform(0.06, 0.15))
    out_w, out_h = w + 2 * margin, h + 2 * margin

    table = rng.choice(
        [
            (120, 85, 60),
            (170, 140, 110),
            (70, 70, 75),
            (200, 200, 195),
            (40, 50, 70),
            (150, 155, 160),
        ]
    )
    background = Image.new("RGB", (out_w, out_h), table)
    background = _noise(background, rng.uniform(4, 12), rng).filter(ImageFilter.GaussianBlur(1.5))

    # Coins du document après perspective (léger décalage aléatoire de chaque coin)
    jitter = lambda: rng.uniform(-0.06, 0.06) * min(w, h)  # noqa: E731
    corners = [
        (margin + jitter(), margin + jitter()),
        (margin + w + jitter(), margin + jitter()),
        (margin + w + jitter(), margin + h + jitter()),
        (margin + jitter(), margin + h + jitter()),
    ]
    coeffs = _perspective_coeffs([(0, 0), (w, 0), (w, h), (0, h)], corners)

    paper = img.convert("RGBA")
    warped = paper.transform(
        (out_w, out_h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC
    )
    mask = warped.getchannel("A")
    shadow = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    shadow_mask = mask.filter(ImageFilter.GaussianBlur(18)).point(lambda p: int(p * 0.45))
    offset = (rng.randint(5, 25), rng.randint(5, 25))
    background.paste(shadow, offset, shadow_mask)
    background.paste(warped.convert("RGB"), (0, 0), mask)

    # Éclairage inégal : dégradé lumineux dans une direction aléatoire
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = np.cos(angle) * xx / out_w + np.sin(angle) * yy / out_h
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
    strength = rng.uniform(0.15, 0.4)
    light = 1 - strength / 2 + strength * ramp
    warm = np.array([1.0, rng.uniform(0.95, 1.0), rng.uniform(0.85, 0.98)], dtype=np.float32)
    arr = np.asarray(background, dtype=np.float32) * light[..., None] * warm
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))
    img = _noise(img, rng.uniform(3, 10), rng)
    # Photo de smartphone redimensionnée : côté le plus long ≈ 1600 px
    scale = 1600 / max(img.size)
    if scale < 1:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS
        )
    return img
