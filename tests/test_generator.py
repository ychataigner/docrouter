import csv
import random
from decimal import Decimal

import pytest
from docrouter_training.data import augment
from docrouter_training.data import fakedata as fd
from docrouter_training.data.generate import CLASSES, generate
from docrouter_training.data.templates import RENDERERS, TEST_LAYOUTS, TRAIN_LAYOUTS
from faker import Faker
from PIL import Image


@pytest.fixture
def fake():
    faker = Faker("fr_FR")
    faker.seed_instance(1)
    return faker


def test_siren_siret_and_vat_are_valid():
    rng = random.Random(0)
    for _ in range(50):
        siren = fd.make_siren(rng)
        siret = fd.make_siret(siren, rng)
        assert len(siren) == 9 and fd.luhn_ok(siren)
        assert len(siret) == 14 and siret.startswith(siren) and fd.luhn_ok(siret)
        assert fd.tva_intracom(siren) == f"FR{(12 + 3 * (int(siren) % 97)) % 97:02d}{siren}"


def test_french_money_format():
    assert fd.fmt_eur(Decimal("1234567.5")) == "1 234 567,50 €"
    assert fd.fmt_eur(Decimal("4.2"), " EUR") == "4,20 EUR"
    assert fd.fmt_rate(Decimal("5.5")) == "5,5 %"


def test_invoice_totals_are_consistent(fake):
    rng = random.Random(3)
    for _ in range(30):
        inv = fd.make_invoice(fake, rng)
        t = inv.totals
        assert t.ht == sum(line.total_ht for line in inv.lines)
        assert t.ttc == t.ht + t.tva
        assert t.tva == sum(t.tva_par_taux.values())


def test_receipt_vat_is_included_in_total(fake):
    rng = random.Random(4)
    for categorie in fd.RECEIPT_CATEGORIES:
        r = fd.make_receipt(fake, rng, categorie)
        assert r.ttc == sum(amount for _, amount in r.lignes)
        assert r.ht + r.tva == r.ttc
        assert abs(r.tva - r.ttc * r.taux / (100 + r.taux)) <= Decimal("0.01")


@pytest.mark.parametrize("classe", CLASSES)
@pytest.mark.parametrize("layout", TRAIN_LAYOUTS + TEST_LAYOUTS)
def test_every_layout_renders(fake, classe, layout):
    img, truth = RENDERERS[classe](fake, random.Random(layout), layout)
    assert img.mode == "RGB" and img.width >= 500 and img.height >= 300
    assert truth["fournisseur"] and truth["date"]


@pytest.mark.parametrize("profile", augment.PROFILES)
def test_augmentations_keep_an_rgb_image(profile):
    img = Image.new("RGB", (300, 400), "white")
    out = augment.apply(img, profile, random.Random(0))
    assert out.mode == "RGB" and out.width > 100


def _manifest(out):
    with (out / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def test_generate_small_dataset(tmp_path):
    info = generate(tmp_path, seed=7, counts={"train": 2, "val": 1, "test": 2})
    rows = _manifest(tmp_path)

    assert info["total"] == len(rows) == 5 * len(CLASSES)
    assert all((tmp_path / row["fichier"]).is_file() for row in rows)
    # Les gabarits du test ne sont jamais utilisés pour l'entraînement / la validation
    train_layouts = {row["gabarit"] for row in rows if row["split"] != "test"}
    test_layouts = {row["gabarit"] for row in rows if row["split"] == "test"}
    assert train_layouts.isdisjoint(test_layouts)
    for row in rows:
        if row["montant_ttc"]:
            assert Decimal(row["montant_ht"]) + Decimal(row["tva"]) == Decimal(row["montant_ttc"])


def test_generation_is_reproducible(tmp_path):
    counts = {"train": 1, "val": 0, "test": 1}
    first = generate(tmp_path / "a", seed=11, counts=counts)
    second = generate(tmp_path / "b", seed=11, counts=counts)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    other = generate(tmp_path / "c", seed=12, counts=counts)
    assert other["manifest_sha256"] != first["manifest_sha256"]
