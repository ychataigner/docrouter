"""Gabarits de mise en page des 4 classes.

Chaque classe a 4 mises en page structurellement différentes : 0, 1 et 2 servent à
l'entraînement, la 3 est réservée au jeu de test (spec §4.1 : aucun gabarit du test n'est vu à
l'entraînement, sinon le modèle reconnaîtrait la mise en page plutôt que le type de document).
Le style (police, couleur, tableau, taille) est en plus tiré au hasard pour chaque document.
"""

import random
from datetime import date
from decimal import Decimal

from faker import Faker
from PIL import Image

from docrouter_training.data import fakedata as fd
from docrouter_training.data.render import A4, TICKET_WIDTH, Canvas, Style

TRAIN_LAYOUTS = (0, 1, 2)
TEST_LAYOUTS = (3,)

MONTHS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]


def fmt_date(day: date, rng: random.Random) -> str:
    return rng.choice(
        [
            f"{day:%d/%m/%Y}",
            f"{day:%d/%m/%Y}",
            f"{day:%d.%m.%Y}",
            f"{day.day} {MONTHS[day.month - 1]} {day.year}",
        ]
    )


def initials(name: str) -> str:
    return "".join(word[0] for word in name.split()[:2] if word[0].isalpha())


def legal_footer(party: fd.Party) -> str:
    form = party.nom.split()[-1]
    return (
        f"{party.nom} – {form} au capital de {party.capital:,} €"
        f" – SIRET {fd.fmt_siret(party.siret)} – TVA {party.tva}"
    ).replace(",", " ")


Truth = dict[str, str]


def _amount_truth(ht: Decimal | None, tva: Decimal | None, ttc: Decimal | None) -> Truth:
    return {
        "montant_ht": f"{ht:.2f}" if ht is not None else "",
        "tva": f"{tva:.2f}" if tva is not None else "",
        "montant_ttc": f"{ttc:.2f}" if ttc is not None else "",
    }


# ================================================================================================
# Factures
# ================================================================================================


def _invoice_rows(inv: fd.Invoice, with_ref: bool, with_rate: bool) -> tuple[list, list]:
    columns = [("Désignation", 470, "l"), ("Qté", 90, "r"), ("PU HT", 160, "r")]
    if with_ref:
        columns.insert(0, ("Réf.", 140, "l"))
        columns[1] = ("Désignation", 380, "l")
    if with_rate:
        columns.append(("TVA", 90, "r"))
    columns.append(("Total HT", 170, "r"))
    rows = []
    for line in inv.lines:
        row = [line.designation, str(line.qty), fd.fmt_eur(line.pu_ht)]
        if with_ref:
            row.insert(0, line.ref)
        if with_rate:
            row.append(fd.fmt_rate(line.taux))
        row.append(fd.fmt_eur(line.total_ht))
        rows.append(row)
    return columns, rows


def _totals_block(c: Canvas, x_right: int, y: int, inv: fd.Invoice, boxed: bool) -> int:
    t = inv.totals
    rows = [("Total HT", fd.fmt_eur(t.ht))]
    rows += [(f"TVA {fd.fmt_rate(rate)}", fd.fmt_eur(v)) for rate, v in t.tva_par_taux.items()]
    rows.append(("Total TTC", fd.fmt_eur(t.ttc)))
    size = c.style.size + 2
    if boxed:
        c.box(
            [x_right - 460, y - 10, x_right, y + len(rows) * int(size * 1.6) + 10],
            outline=c.style.accent,
        )
    for i, (label, value) in enumerate(rows):
        bold = i == len(rows) - 1
        c.text((x_right - 440, y), label, size, bold)
        c.text((x_right - 20, y), value, size, bold, anchor="ra")
        y += int(size * 1.6)
    return y + 20


def render_invoice(fake: Faker, rng: random.Random, layout: int) -> tuple[Image.Image, Truth]:
    inv = fd.make_invoice(fake, rng)
    c = Canvas(Style.random(rng))
    em, cl, s = inv.emetteur, inv.client, c.style
    title = rng.choice(["FACTURE", "Facture", "FACTURE"])
    when = fmt_date(inv.date, rng)

    if layout == 0:  # classique : émetteur à gauche, titre à droite
        c.logo(80, 70, initials(em.nom), rng)
        c.lines(80, 180, [em.nom], s.size + 6, True)
        c.lines(80, 225, [*em.adresse, f"Tél. {em.tel}", em.email])
        c.text((1160, 70), title, 58, True, s.accent, "ra")
        c.lines(
            1160,
            160,
            [f"N° {inv.numero}", f"Date : {when}", f"Commande : {inv.commande}"],
            anchor="ra",
        )
        c.lines(700, 400, ["Facturé à :", cl.nom, *cl.adresse], bold=False)
        columns, rows = _invoice_rows(inv, with_ref=rng.random() < 0.5, with_rate=False)
        y = c.table(80, 620, columns, rows)
        y = _totals_block(c, 1160, y + 50, inv, boxed=rng.random() < 0.5)
        c.lines(80, y + 20, [inv.conditions, f"Échéance : {fmt_date(inv.echeance, rng)}"])
        c.text((620, 1660), legal_footer(em), 15, anchor="mm")

    elif layout == 1:  # bandeau de couleur
        c.box([0, 0, A4[0], 170], outline=s.accent, fill=s.accent)
        c.text((80, 85), em.nom, 40, True, (255, 255, 255), "lm")
        c.text((1160, 85), title, 44, True, (255, 255, 255), "rm")
        c.lines(80, 210, [*em.adresse, f"SIRET {fd.fmt_siret(em.siret)}"], s.size - 2)
        c.lines(80, 360, [f"Facture n° {inv.numero}", f"du {when}"], s.size + 4, True)
        c.lines(680, 340, ["Client", cl.nom, *cl.adresse, f"N° TVA : {cl.tva}"])
        columns, rows = _invoice_rows(inv, with_ref=False, with_rate=True)
        y = c.table(80, 600, columns, rows)
        y = _totals_block(c, 1160, y + 40, inv, boxed=True)
        c.box([80, y + 20, 700, y + 160], outline=s.accent)
        c.lines(
            100,
            y + 35,
            [
                "Règlement par virement :",
                f"IBAN FR76 {rng.randint(1000, 9999)} {rng.randint(1000, 9999)} "
                f"{rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(100, 999)}",
                inv.conditions,
            ],
            s.size - 2,
        )
        c.text((620, 1680), legal_footer(em), 15, anchor="mm")

    elif layout == 2:  # titre centré, deux encadrés
        c.text((620, 90), title, 52, True, s.accent, "mm")
        c.text((620, 150), f"N° {inv.numero} – {when}", s.size + 2, anchor="mm")
        c.box([80, 220, 590, 440])
        c.lines(100, 235, ["Émetteur", em.nom, *em.adresse, em.tel], s.size - 1)
        c.box([650, 220, 1160, 440])
        c.lines(670, 235, ["Destinataire", cl.nom, *cl.adresse], s.size - 1)
        c.text(
            (80, 490),
            f"Votre commande {inv.commande}  ·  Échéance {fmt_date(inv.echeance, rng)}",
            s.size,
        )
        columns, rows = _invoice_rows(inv, with_ref=True, with_rate=rng.random() < 0.5)
        y = c.table(80, 560, columns, rows)
        _totals_block(c, 1160, y + 40, inv, boxed=False)
        c.rule(80, 1600, 1160, 1)
        c.text((620, 1640), legal_footer(em), 15, anchor="mm")
        c.text(
            (620, 1670),
            "En cas de retard de paiement, indemnité forfaitaire de 40 €.",
            15,
            anchor="mm",
        )

    else:  # 3 — réservé au test : colonne latérale colorée
        c.box([0, 0, 330, A4[1]], outline=s.accent, fill=s.accent)
        white = (255, 255, 255)
        c.logo(60, 60, initials(em.nom), rng, 110)
        y = c.lines(40, 220, c.wrap(em.nom, 260, s.size + 2), s.size + 2, True, white)
        c.lines(40, y + 20, [*em.adresse, em.tel, em.email], s.size - 4, color=white)
        c.lines(40, 1450, ["SIRET", fd.fmt_siret(em.siret), "TVA", em.tva], s.size - 5, color=white)
        c.text((400, 80), title.lower().capitalize(), 64, True, s.accent)
        heads = ["Date", "Numéro", "Échéance", "Commande"]
        values = [when, inv.numero, fmt_date(inv.echeance, rng), inv.commande]
        for i, (h, v) in enumerate(zip(heads, values, strict=True)):
            x0 = 400 + i * 200
            c.box([x0, 200, x0 + 190, 280], width=1)
            c.text((x0 + 10, 210), h, s.size - 6, True)
            c.text((x0 + 10, 245), v, s.size - 6)
        c.lines(400, 330, ["Adressée à", cl.nom, *cl.adresse])
        columns = [
            ("Article", 430, "l"),
            ("Qté", 70, "r"),
            ("Prix", 130, "r"),
            ("Montant", 150, "r"),
        ]
        rows = [
            [ln.designation, str(ln.qty), fd.fmt_eur(ln.pu_ht), fd.fmt_eur(ln.total_ht)]
            for ln in inv.lines
        ]
        y = c.table(400, 560, columns, rows)
        t = inv.totals
        recap = [
            ["Montant HT", fd.fmt_eur(t.ht)],
            ["TVA", fd.fmt_eur(t.tva)],
            ["Net à payer", fd.fmt_eur(t.ttc)],
        ]
        c.table(780, y + 50, [("Récapitulatif", 220, "l"), ("", 180, "r")], recap)

    truth = {
        "fournisseur": em.nom,
        "date": inv.date.isoformat(),
        "numero": inv.numero,
        "numero_commande": inv.commande,
        "categorie": "",
    }
    return c.img, truth | _amount_truth(inv.totals.ht, inv.totals.tva, inv.totals.ttc)


# ================================================================================================
# Bons de livraison
# ================================================================================================


def _signatures(c: Canvas, y: int, labels: list[str]) -> None:
    width = (1080 - 40 * (len(labels) - 1)) // len(labels)
    for i, label in enumerate(labels):
        x0 = 80 + i * (width + 40)
        c.box([x0, y, x0 + width, y + 180], width=1)
        c.text((x0 + 12, y + 12), label, c.style.size - 3)


def render_delivery(fake: Faker, rng: random.Random, layout: int) -> tuple[Image.Image, Truth]:
    bl = fd.make_delivery(fake, rng)
    c = Canvas(Style.random(rng))
    ex, de, s = bl.expediteur, bl.destinataire, c.style
    title = rng.choice(["BON DE LIVRAISON", "Bon de livraison", "BON DE LIVRAISON"])
    when = fmt_date(bl.date, rng)

    if layout == 0:
        c.logo(80, 70, initials(ex.nom), rng)
        c.lines(80, 180, [ex.nom], s.size + 6, True)
        c.lines(80, 225, [*ex.adresse, f"Tél. {ex.tel}"])
        c.text((1160, 70), title, 44, True, s.accent, "ra")
        c.lines(
            1160,
            150,
            [f"BL n° {bl.numero}", f"Date de livraison : {when}", f"Commande : {bl.commande}"],
            anchor="ra",
        )
        c.lines(700, 380, ["Livré à :", de.nom, *de.adresse])
        columns = [
            ("Référence", 200, "l"),
            ("Désignation", 480, "l"),
            ("Commandé", 200, "r"),
            ("Livré", 200, "r"),
        ]
        rows = [[ln.ref, ln.designation, str(ln.commandee), str(ln.livree)] for ln in bl.lines]
        y = c.table(80, 600, columns, rows)
        c.lines(
            80,
            y + 40,
            [
                f"Nombre de colis : {bl.colis}   Poids : {bl.poids_kg} kg",
                f"Observations : {bl.observations}",
            ],
        )
        _signatures(c, 1400, ["Signature du livreur", "Signature et cachet du client"])

    elif layout == 1:
        c.box([0, 0, A4[0], 150], outline=s.accent, fill=s.accent)
        c.text((80, 75), ex.nom, 38, True, (255, 255, 255), "lm")
        c.text((1160, 75), title, 36, True, (255, 255, 255), "rm")
        info = [
            ("N° BL", bl.numero),
            ("Date", when),
            ("Commande", bl.commande),
            ("Transporteur", bl.transporteur),
        ]
        for i, (h, v) in enumerate(info):
            x0 = 80 + i * 270
            c.box([x0, 200, x0 + 255, 290], width=1)
            c.text((x0 + 10, 210), h, s.size - 5, True)
            c.text((x0 + 10, 250), v, s.size - 5)
        c.lines(80, 340, ["Expéditeur", *ex.adresse], s.size - 2)
        c.lines(680, 340, ["Destinataire", de.nom, *de.adresse], s.size - 2)
        columns = [
            ("Désignation", 560, "l"),
            ("Réf.", 180, "l"),
            ("Qté", 140, "r"),
            ("Unité", 200, "c"),
        ]
        rows = [
            [ln.designation, ln.ref, str(ln.livree), rng.choice(["pièce", "carton", "lot"])]
            for ln in bl.lines
        ]
        y = c.table(80, 540, columns, rows)
        c.text((80, y + 40), f"{bl.colis} colis – {bl.poids_kg} kg", s.size, True)
        c.paragraph(80, y + 90, f"Remarques : {bl.observations}", 1080)
        c.text((80, 1370), "Bon pour réception (date, nom et signature) :", s.size)
        c.rule(80, 1520, 700, 1)

    elif layout == 2:
        c.text((620, 90), title, 48, True, s.accent, "mm")
        c.text((620, 150), f"N° {bl.numero}  ·  {when}", s.size + 2, anchor="mm")
        c.box([80, 210, 590, 410])
        c.lines(100, 225, ["Fournisseur", ex.nom, *ex.adresse], s.size - 1)
        c.box([650, 210, 1160, 410])
        c.lines(670, 225, ["Adresse de livraison", de.nom, *de.adresse], s.size - 1)
        c.text(
            (80, 460),
            f"Réf. commande client : {bl.commande}   Transport : {bl.transporteur}",
            s.size,
        )
        columns = [
            ("Réf.", 160, "l"),
            ("Article", 520, "l"),
            ("Qté", 120, "r"),
            ("Conforme", 140, "c"),
            ("Reliquat", 140, "r"),
        ]
        rows = [
            [
                ln.ref,
                ln.designation,
                str(ln.livree),
                "Non" if ln.livree < ln.commandee else "Oui",
                str(ln.commandee - ln.livree),
            ]
            for ln in bl.lines
        ]
        y = c.table(80, 530, columns, rows)
        c.text((80, y + 40), bl.observations, s.size)
        _signatures(c, 1380, ["Préparé par", "Livré par", "Reçu par"])

    else:  # 3 — réservé au test
        c.box([0, 0, 330, A4[1]], outline=s.accent, fill=s.accent)
        white = (255, 255, 255)
        c.logo(60, 60, initials(ex.nom), rng, 110)
        y = c.lines(40, 220, c.wrap(ex.nom, 260, s.size + 2), s.size + 2, True, white)
        c.lines(40, y + 20, [*ex.adresse, ex.tel], s.size - 4, color=white)
        c.text((400, 80), "Bon de livraison", 54, True, s.accent)
        c.lines(
            400, 180, [f"Numéro : {bl.numero}", f"Livré le : {when}", f"Commande : {bl.commande}"]
        )
        c.lines(800, 180, ["Livré à", de.nom, *de.adresse])
        columns = [
            ("Désignation", 420, "l"),
            ("Réf.", 140, "l"),
            ("Livré", 100, "r"),
            ("Reliquat", 120, "r"),
        ]
        rows = [
            [ln.designation, ln.ref, str(ln.livree), str(ln.commandee - ln.livree)]
            for ln in bl.lines
        ]
        y = c.table(400, 420, columns, rows)
        c.lines(400, y + 40, [f"Colis : {bl.colis}", f"Transporteur : {bl.transporteur}"])
        c.text((400, 1450), "Signature du destinataire :", s.size)
        c.rule(400, 1600, 1100, 1)

    truth = {
        "fournisseur": ex.nom,
        "date": bl.date.isoformat(),
        "numero": bl.numero,
        "numero_commande": bl.commande,
        "categorie": "",
    }
    return c.img, truth | _amount_truth(None, None, None)


# ================================================================================================
# Tickets (notes de frais)
# ================================================================================================


def render_receipt(fake: Faker, rng: random.Random, layout: int) -> tuple[Image.Image, Truth]:
    r = fd.make_receipt(fake, rng)
    style = Style.random(rng, "mono")
    style.size = rng.randint(20, 24)
    c = Canvas(style, (TICKET_WIDTH, 2200))
    s, w = style, TICKET_WIDTH
    mid, left, right = w // 2, 30, w - 30
    dashes = "-" * 40
    when = f"{r.date:%d/%m/%Y}" if rng.random() < 0.7 else f"{r.date:%d/%m/%y}"
    hour = f"{r.heure:%H:%M}"

    def row(y, label, value, bold=False, size=None):
        c.text((left, y), label, size, bold)
        c.text((right, y), value, size, bold, anchor="ra")
        return y + int((size or s.size) * 1.45)

    def center(y, text, size=None, bold=False):
        c.text((mid, y), text, size, bold, anchor="ma")
        return y + int((size or s.size) * 1.45)

    euro = rng.choice([" EUR", " €", ""])
    y = 40
    if layout == 0:
        y = center(y, r.marchand, s.size + 6, True)
        for line in (r.rue, r.ville, f"Tel {r.tel}"):
            y = center(y, line)
        y = center(y + 10, f"{when}  {hour}")
        for line in r.extra:
            y = center(y, line)
        y = center(y, dashes)
        for label, amount in r.lignes:
            y = row(y, label[:26], fd.fmt_eur(amount, ""))
        y = center(y, dashes)
        y = row(y + 5, "TOTAL", fd.fmt_eur(r.ttc, euro), True, s.size + 6)
        y = row(y, f"dont TVA {fd.fmt_rate(r.taux)}", fd.fmt_eur(r.tva, ""))
        y = center(y + 15, f"{r.paiement}  {fd.fmt_eur(r.ttc, euro)}")
        y = center(y + 15, rng.choice(["Merci de votre visite !", "A bientot !", "Bonne route !"]))

    elif layout == 1:
        y = c.lines(left, y, [r.marchand.upper()], s.size + 4, True)
        y = c.lines(left, y, [r.rue.upper(), r.ville.upper(), f"SIRET {r.siret}"])
        y = c.lines(left, y + 10, [f"TICKET N° {r.numero}", f"LE {when} A {hour}", *r.extra])
        y = center(y, "=" * 40)
        for label, amount in r.lignes:
            y = row(y, label.upper()[:26], fd.fmt_eur(amount, ""))
        y = center(y, "=" * 40)
        y = row(y, "TOTAL TTC", fd.fmt_eur(r.ttc, euro), True, s.size + 4)
        y = row(y + 10, f"TVA {fd.fmt_rate(r.taux)}", "HT      TVA      TTC", size=s.size - 3)
        y = row(
            y,
            "",
            f"{fd.fmt_eur(r.ht, '')}  {fd.fmt_eur(r.tva, '')}  {fd.fmt_eur(r.ttc, '')}",
            size=s.size - 3,
        )
        y = center(y + 20, "CARTE BANCAIRE")
        y = center(y, f"************{rng.randint(1000, 9999)}")
        y = center(y, f"A {rng.randint(1000000, 9999999):07d}  AUTO {rng.randint(100000, 999999)}")
        y = center(y, "TICKET CLIENT - A CONSERVER", bold=True)

    elif layout == 2:  # facturette courte de terminal bancaire + rappel des montants
        y = center(y, "CARTE BANCAIRE", s.size + 2, True)
        y = center(y, "SANS CONTACT" if "CONTACT" in r.paiement else "CONTACT")
        y = center(y + 10, f"LE {when} A {hour}")
        y = center(y, r.marchand[:30])
        y = center(y, r.ville.upper()[:30])
        y = center(y, f"{rng.randint(1000000, 9999999)}")
        y = center(y, f"************{rng.randint(1000, 9999)}")
        y = center(y + 10, "MONTANT REEL", bold=True)
        y = center(y, f"{fd.fmt_eur(r.ttc, ' EUR')}", s.size + 8, True)
        y = center(y + 10, "DEBIT")
        y = center(y, "TICKET CLIENT")
        y = center(y + 20, dashes)
        y = center(y, f"Dont TVA {fd.fmt_rate(r.taux)} : {fd.fmt_eur(r.tva, ' EUR')}")
        y = center(y, f"Ref. {r.categorie.upper()} {r.numero}")

    else:  # 3 — réservé au test : bloc d'en-tête inversé, étoiles, code-barres
        c.box([left, y, right, y + 90], outline=s.ink, fill=s.ink)
        c.text((mid, y + 45), r.marchand[:24], s.size + 4, True, (255, 255, 255), "mm")
        y += 110
        y = center(y, f"{r.rue} - {r.ville}"[:40], s.size - 3)
        y = center(y, "*" * 40)
        y = row(y, "QTE ARTICLE", "MONTANT", True)
        for label, amount in r.lignes:
            y = row(y, label[:24], fd.fmt_eur(amount, ""))
        y = center(y, "*" * 40)
        y = row(y, "A PAYER", fd.fmt_eur(r.ttc, euro), True, s.size + 4)
        y = row(y, "TVA", f"{fd.fmt_rate(r.taux)} = {fd.fmt_eur(r.tva, '')}")
        y = row(y, "Reglement", r.paiement)
        y = center(y + 10, f"{when} {hour}  Caisse {rng.randint(1, 9)}  N°{r.numero}")
        x = left + 40
        for _ in range(60):  # code-barres décoratif
            bar = rng.choice([2, 3, 5])
            c.box([x, y + 20, x + bar, y + 110], outline=s.ink, fill=s.ink, width=1)
            x += bar + rng.choice([2, 3, 4])
            if x > right - 40:
                break
        y += 140

    img = c.img.crop((0, 0, w, min(y + 40, c.img.height)))
    truth = {
        "fournisseur": r.marchand,
        "date": r.date.isoformat(),
        "numero": r.numero,
        "numero_commande": "",
        "categorie": r.categorie,
    }
    return img, truth | _amount_truth(r.ht, r.tva, r.ttc)


# ================================================================================================
# Autres documents
# ================================================================================================


def _form_fields(c: Canvas, y: int, fake: Faker, rng: random.Random) -> int:
    fields = [
        "Nom",
        "Prénom",
        "Date de naissance",
        "Adresse",
        "Code postal / Ville",
        "Téléphone",
        "Courriel",
        "N° d'adhérent",
    ]
    for label in rng.sample(fields, k=rng.randint(5, len(fields))):
        c.text((80, y), f"{label} :", c.style.size)
        c.draw.line(
            [(420, y + c.style.size), (1160, y + c.style.size)], fill=(120, 120, 120), width=1
        )
        if rng.random() < 0.5:
            c.text((430, y - 2), fake.word().capitalize(), c.style.size, color=(30, 30, 140))
        y += 70
    y += 20
    for option in rng.sample(
        [
            "Première demande",
            "Renouvellement",
            "Modification",
            "Je certifie l'exactitude des informations",
        ],
        k=3,
    ):
        c.box([80, y, 110, y + 30], width=2)
        c.text((130, y), option, c.style.size)
        y += 55
    return y


def render_other(fake: Faker, rng: random.Random, layout: int) -> tuple[Image.Image, Truth]:
    doc = fd.make_other(fake, rng)
    c = Canvas(Style.random(rng))
    s, em = c.style, doc.emetteur
    when = f"{doc.ville}, le {fmt_date(doc.date, rng)}"
    body_x, body_w = 80, 1080

    if layout == 0:  # lettre classique
        y = c.lines(80, 80, [em.nom], s.size + 4, True)
        c.lines(80, y, [*em.adresse, em.tel])
        c.lines(700, 300, [doc.destinataire_nom, *doc.destinataire_adresse])
        c.text((700, 500), when, s.size)
        y = 600
    elif layout == 1:  # en-tête centré + filet
        c.text((620, 90), em.nom, s.size + 10, True, s.accent, "mm")
        c.text((620, 140), " – ".join(em.adresse), s.size - 2, anchor="mm")
        c.rule(80, 180, 1160, 3, s.accent)
        c.text((80, 220), f"Réf. : {doc.reference}", s.size - 2)
        c.text((1160, 220), when, s.size - 2, anchor="ra")
        y = 300
    elif layout == 2:  # logo + bloc destinataire encadré
        c.logo(80, 70, initials(em.nom), rng)
        c.lines(200, 80, [em.nom, *em.adresse], s.size - 1)
        c.box([650, 260, 1160, 420], width=1)
        c.lines(670, 280, [doc.destinataire_nom, *doc.destinataire_adresse])
        c.text((80, 470), when, s.size)
        y = 540
    else:  # 3 — réservé au test : bandeau + deux colonnes d'en-tête + pied de page
        c.box([0, 0, A4[0], 120], outline=s.accent, fill=s.accent)
        c.text((80, 60), em.nom, s.size + 8, True, (255, 255, 255), "lm")
        c.lines(80, 160, ["Expéditeur", *em.adresse], s.size - 2)
        c.lines(
            680, 160, ["Destinataire", doc.destinataire_nom, *doc.destinataire_adresse], s.size - 2
        )
        c.text((680, 330), when, s.size - 2)
        c.rule(0, 1680, A4[0], 2, s.accent)
        c.text((620, 1710), f"{em.nom} · {em.tel} · {em.email}", 16, anchor="mm")
        y = 420

    title = (
        doc.objet.upper()
        if doc.kind in ("attestation", "convocation", "formulaire", "compte_rendu")
        else f"Objet : {doc.objet}"
    )
    if doc.kind == "lettre":
        y = c.lines(body_x, y, [title], s.size + 2, True) + 20
        y = c.lines(body_x, y, ["Madame, Monsieur,"]) + 15
    else:
        c.text((620, y), title, s.size + 8, True, s.accent, "ma")
        y += 90

    if doc.kind == "formulaire":
        y = _form_fields(c, y, fake, rng)
    else:
        for paragraph in doc.paragraphes:
            y = c.paragraph(body_x, y, paragraph, body_w) + 18
            if y > 1350:
                break
    if doc.kind == "compte_rendu":
        y = c.lines(body_x, y + 10, [f"Présents : {fake.name()}, {fake.name()}, {fake.name()}"])
    c.text((900, min(y + 60, 1500)), doc.signataire, s.size, True, anchor="ma")

    truth = {
        "fournisseur": em.nom,
        "date": doc.date.isoformat(),
        "numero": doc.reference,
        "numero_commande": "",
        "categorie": doc.kind,
    }
    return c.img, truth | _amount_truth(None, None, None)


RENDERERS = {
    "facture": render_invoice,
    "bon_de_livraison": render_delivery,
    "note_de_frais": render_receipt,
    "autre": render_other,
}
