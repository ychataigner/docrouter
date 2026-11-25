"""Génère le jeu de documents fictifs français, étiquetés, avec leur vérité terrain.

    uv run docrouter-generate                       # 240 train + 60 val + 30 test par classe
    uv run docrouter-generate --train 20 --val 5 --test 5 --out data/synth_mini

Structure produite :
    data/synth/
      train/<classe>/*.jpg     gabarits 0-2
      val/<classe>/*.jpg       gabarits 0-2
      test/<classe>/*.jpg|pdf  gabarit 3 uniquement (jamais vu à l'entraînement)
      manifest.csv             une ligne par document (classe, gabarit, profil, vérité terrain)
      dataset_info.json        paramètres, effectifs, empreinte SHA-256 du manifeste

Le résultat est reproductible : même graine → mêmes documents (à polices système identiques).
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from docrouter_core import DocumentType
from faker import Faker

from docrouter_training.data import augment
from docrouter_training.data.templates import RENDERERS, TEST_LAYOUTS, TRAIN_LAYOUTS

GENERATOR_VERSION = "1.0"
SPLITS = ("train", "val", "test")
CLASSES = [t.value for t in DocumentType]
COLUMNS = [
    "fichier",
    "split",
    "classe",
    "gabarit",
    "profil",
    "fournisseur",
    "date",
    "numero",
    "numero_commande",
    "montant_ht",
    "tva",
    "montant_ttc",
    "categorie",
]
PDF_SHARE_IN_TEST = 0.25  # une partie du test en PDF, pour tester ce chemin dans l'API


@dataclass(frozen=True)
class Job:
    out: Path
    seed: int
    classe: str
    split: str
    index: int


def _doc_seed(job: Job) -> int:
    key = f"{job.seed}|{job.classe}|{job.split}|{job.index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


_FAKER: Faker | None = None


def _faker() -> Faker:
    global _FAKER
    if _FAKER is None:
        _FAKER = Faker("fr_FR")
    return _FAKER


def make_document(job: Job) -> dict[str, str]:
    """Génère, dégrade et enregistre un document ; renvoie sa ligne de manifeste."""
    seed = _doc_seed(job)
    rng = random.Random(seed)
    fake = _faker()
    fake.seed_instance(seed)

    layouts = TEST_LAYOUTS if job.split == "test" else TRAIN_LAYOUTS
    layout = rng.choice(layouts)
    img, truth = RENDERERS[job.classe](fake, rng, layout)
    profile = augment.pick_profile(job.classe, rng)
    img = augment.apply(img, profile, rng)

    as_pdf = job.split == "test" and rng.random() < PDF_SHARE_IN_TEST
    relative = Path(job.split) / job.classe / f"{job.classe}_{job.split}_{job.index:04d}"
    target = job.out / relative.with_suffix(".pdf" if as_pdf else ".jpg")
    target.parent.mkdir(parents=True, exist_ok=True)
    if as_pdf:
        img.save(target, "PDF", resolution=150)
    else:
        img.save(target, "JPEG", quality=95 if profile == "propre" else rng.randint(70, 92))

    return {
        "fichier": target.relative_to(job.out).as_posix(),
        "split": job.split,
        "classe": job.classe,
        "gabarit": str(layout),
        "profil": profile,
    } | truth


def build_jobs(out: Path, seed: int, counts: dict[str, int]) -> list[Job]:
    return [
        Job(out, seed, classe, split, i)
        for split in SPLITS
        for classe in CLASSES
        for i in range(counts[split])
    ]


def write_manifest(out: Path, rows: list[dict[str, str]]) -> str:
    rows = sorted(rows, key=lambda r: r["fichier"])
    path = out / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out: Path, seed: int, counts: dict[str, int], workers: int = 1) -> dict:
    jobs = build_jobs(out, seed, counts)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(make_document, jobs, chunksize=8))
    else:
        rows = [make_document(job) for job in jobs]
    digest = write_manifest(out, rows)
    info = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "per_class": counts,
        "total": len(rows),
        "counts": {
            f"{s}/{c}": n
            for (s, c), n in sorted(Counter((r["split"], r["classe"]) for r in rows).items())
        },
        "train_layouts": list(TRAIN_LAYOUTS),
        "test_layouts": list(TEST_LAYOUTS),
        "manifest_sha256": digest,
    }
    (out / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/synth"))
    parser.add_argument("--train", type=int, default=240, help="documents par classe")
    parser.add_argument("--val", type=int, default=60, help="documents par classe")
    parser.add_argument("--test", type=int, default=30, help="documents par classe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--overwrite", action="store_true", help="remplace un jeu existant dans --out"
    )
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        if not args.overwrite:
            sys.exit(f"{args.out} n'est pas vide : ajoutez --overwrite pour le régénérer.")
        for name in (*SPLITS, "manifest.csv", "dataset_info.json"):
            target = args.out / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    args.out.mkdir(parents=True, exist_ok=True)

    counts = {"train": args.train, "val": args.val, "test": args.test}
    total = sum(counts.values()) * len(CLASSES)
    print(f"Génération de {total} documents dans {args.out} ({args.workers} processus)…")
    start = time.perf_counter()
    info = generate(args.out, args.seed, counts, args.workers)
    print(
        f"Terminé en {time.perf_counter() - start:.0f} s : {info['total']} documents, "
        f"manifeste {info['manifest_sha256'][:12]}"
    )


if __name__ == "__main__":
    main()
