"""Contenu fictif mais cohérent : sociétés (SIRET/TVA valides), lignes, montants, tickets.

Toutes les valeurs sont générées : aucune donnée réelle. Les montants sont des `Decimal`
arrondis au centime, avec HT + TVA = TTC exactement (vérité terrain pour la partie 2).
"""

import random
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from decimal import ROUND_HALF_UP, Decimal

from faker import Faker

CENT = Decimal("0.01")
DATE_MIN = date(2025, 1, 1)
DATE_MAX = date(2026, 10, 31)


def money(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def fmt_eur(value: Decimal, symbol: str = " €", thousands: str = " ") -> str:
    """Format français : 1 234,56 €."""
    units, cents = f"{value:.2f}".split(".")
    groups = []
    while len(units) > 3:
        groups.insert(0, units[-3:])
        units = units[:-3]
    groups.insert(0, units)
    return f"{thousands.join(groups)},{cents}{symbol}"


def fmt_rate(rate: Decimal) -> str:
    return f"{rate.normalize():f}".replace(".", ",") + " %"


# --- Identifiants d'entreprise -------------------------------------------------------------


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _complete_luhn(prefix: str) -> str:
    """Ajoute le chiffre qui rend `prefix + chiffre` valide selon Luhn."""
    for check in "0123456789":
        if luhn_ok(prefix + check):
            return prefix + check
    raise AssertionError("inaccessible")


def make_siren(rng: random.Random) -> str:
    return _complete_luhn(str(rng.randint(10_000_000, 99_999_999)))


def make_siret(siren: str, rng: random.Random) -> str:
    return _complete_luhn(siren + f"{rng.randint(1, 9999):04d}")


def tva_intracom(siren: str) -> str:
    key = (12 + 3 * (int(siren) % 97)) % 97
    return f"FR{key:02d}{siren}"


def fmt_siret(siret: str) -> str:
    return f"{siret[:3]} {siret[3:6]} {siret[6:9]} {siret[9:]}"


# --- Parties ---------------------------------------------------------------------------------

LEGAL_FORMS = ["SAS", "SARL", "SA", "EURL", "SASU", "SNC"]


@dataclass
class Party:
    nom: str
    rue: str
    ville: str  # « 69003 Lyon »
    siret: str
    tva: str
    tel: str
    email: str
    capital: int

    @property
    def siren(self) -> str:
        return self.siret[:9]

    @property
    def adresse(self) -> list[str]:
        return [self.rue, self.ville]


def make_party(fake: Faker, rng: random.Random, kind: str = "société") -> Party:
    base = fake.last_name() if rng.random() < 0.5 else fake.company().split(" ")[0]
    activity = rng.choice(
        [
            "Fournitures",
            "Distribution",
            "Industrie",
            "Services",
            "Négoce",
            "Équipements",
            "Logistique",
            "Bureautique",
            "Matériaux",
            "Informatique",
            "Emballages",
            "Conseil",
        ]
    )
    nom = f"{base} {activity} {rng.choice(LEGAL_FORMS)}" if kind == "société" else base
    siren = make_siren(rng)
    domain = "".join(c for c in base.lower() if c.isalpha()) or "societe"
    return Party(
        nom=nom,
        rue=fake.street_address(),
        ville=f"{fake.postcode()} {fake.city()}",
        siret=make_siret(siren, rng),
        tva=tva_intracom(siren),
        tel=fake.phone_number(),
        email=f"contact@{domain}.fr",
        capital=rng.choice([1000, 5000, 10000, 37000, 50000, 100000, 250000]),
    )


def make_date(rng: random.Random, start: date = DATE_MIN, end: date = DATE_MAX) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


# --- Catalogues --------------------------------------------------------------------------------

CATALOGUES: dict[str, list[tuple[str, float, float]]] = {
    # désignation, prix unitaire HT min, max
    "bureau": [
        ("Ramette papier A4 80 g", 3.5, 6.0),
        ("Classeur à levier dos 80 mm", 1.8, 4.5),
        ("Boîte de stylos bille (50)", 9.0, 18.0),
        ("Cartouche d'encre noire", 18.0, 45.0),
        ("Agrafeuse de bureau", 6.0, 22.0),
        ("Bloc-notes A5 (lot de 10)", 7.0, 15.0),
        ("Enveloppes C5 (boîte de 500)", 14.0, 30.0),
        ("Surligneurs assortis (lot de 4)", 3.0, 7.0),
    ],
    "informatique": [
        ("Écran 24 pouces Full HD", 110.0, 220.0),
        ("Clavier sans fil", 18.0, 60.0),
        ("Souris optique USB", 8.0, 30.0),
        ("Câble HDMI 2 m", 6.0, 15.0),
        ("Disque SSD 1 To", 55.0, 120.0),
        ("Station d'accueil USB-C", 80.0, 190.0),
        ("Licence logiciel bureautique (1 an)", 60.0, 140.0),
        ("Casque audio USB", 25.0, 90.0),
    ],
    "materiaux": [
        ("Sac de ciment 35 kg", 7.0, 12.0),
        ("Plaque de plâtre BA13", 6.0, 11.0),
        ("Rail métallique 3 m", 2.5, 5.0),
        ("Vis à bois (boîte de 200)", 6.0, 14.0),
        ("Peinture acrylique blanche 10 L", 35.0, 80.0),
        ("Laine de verre (rouleau)", 18.0, 40.0),
        ("Parpaing 20x20x50", 1.2, 2.5),
        ("Planche pin 2 m", 4.0, 9.0),
    ],
    "alimentaire": [
        ("Café en grains 1 kg", 12.0, 25.0),
        ("Eau minérale (pack de 6)", 2.0, 4.0),
        ("Sucre en morceaux 1 kg", 1.5, 3.0),
        ("Thé assorti (boîte de 100)", 6.0, 14.0),
        ("Biscuits secs (carton)", 9.0, 20.0),
        ("Jus d'orange 1 L (lot de 6)", 7.0, 13.0),
    ],
    "entretien": [
        ("Essuie-mains (carton)", 18.0, 35.0),
        ("Produit multi-surfaces 5 L", 8.0, 18.0),
        ("Sacs poubelle 100 L (rouleau)", 4.0, 9.0),
        ("Gants nitrile (boîte de 100)", 6.0, 14.0),
        ("Savon liquide 5 L", 9.0, 20.0),
        ("Papier toilette (lot de 48)", 15.0, 30.0),
    ],
}

TVA_RATES = [Decimal("20"), Decimal("10"), Decimal("5.5"), Decimal("2.1")]
CATALOGUE_RATES = {"alimentaire": Decimal("5.5")}


@dataclass
class Line:
    ref: str
    designation: str
    qty: int
    pu_ht: Decimal
    taux: Decimal

    @property
    def total_ht(self) -> Decimal:
        return money(self.pu_ht * self.qty)


def make_lines(rng: random.Random, catalogue: str, n: int) -> list[Line]:
    items = rng.sample(CATALOGUES[catalogue], k=min(n, len(CATALOGUES[catalogue])))
    rate = CATALOGUE_RATES.get(catalogue, Decimal("20"))
    prefix = catalogue[:3].upper()
    return [
        Line(
            ref=f"{prefix}-{rng.randint(100, 9999)}",
            designation=name,
            qty=rng.choice([1, 1, 2, 3, 4, 5, 6, 10, 12, 20, 25, 50]),
            pu_ht=money(rng.uniform(low, high)),
            taux=rate,
        )
        for name, low, high in items
    ]


@dataclass
class Totals:
    ht: Decimal
    tva_par_taux: dict[Decimal, Decimal]
    tva: Decimal
    ttc: Decimal


def totals(lines: list[Line]) -> Totals:
    by_rate: dict[Decimal, Decimal] = {}
    for line in lines:
        by_rate[line.taux] = by_rate.get(line.taux, Decimal(0)) + line.total_ht
    tva_par_taux = {rate: money(base * rate / 100) for rate, base in by_rate.items()}
    ht = sum((line.total_ht for line in lines), Decimal(0))
    tva = sum(tva_par_taux.values(), Decimal(0))
    return Totals(ht=ht, tva_par_taux=tva_par_taux, tva=tva, ttc=ht + tva)


# --- Documents -------------------------------------------------------------------------------


@dataclass
class Invoice:
    emetteur: Party
    client: Party
    numero: str
    date: date
    echeance: date
    commande: str
    lines: list[Line]
    totals: Totals
    conditions: str


def make_invoice(fake: Faker, rng: random.Random) -> Invoice:
    emetteur, client = make_party(fake, rng), make_party(fake, rng)
    day = make_date(rng)
    lines = make_lines(rng, rng.choice(list(CATALOGUES)), rng.randint(1, 7))
    delay = rng.choice([0, 15, 30, 45, 60])
    numero = rng.choice(["FA-{y}-{n:04d}", "F{y}{n:05d}", "FACT/{y}/{n:03d}", "{y}-{n:05d}"])
    return Invoice(
        emetteur=emetteur,
        client=client,
        numero=numero.format(y=day.year, n=rng.randint(1, 9999)),
        date=day,
        echeance=day + timedelta(days=delay),
        commande=f"CMD-{rng.randint(1000, 99999)}",
        lines=lines,
        totals=totals(lines),
        conditions="Paiement à réception" if delay == 0 else f"Paiement à {delay} jours",
    )


@dataclass
class DeliveryLine:
    ref: str
    designation: str
    commandee: int
    livree: int


@dataclass
class DeliveryNote:
    expediteur: Party
    destinataire: Party
    numero: str
    date: date
    commande: str
    transporteur: str
    lines: list[DeliveryLine]
    colis: int
    poids_kg: float
    observations: str


def make_delivery(fake: Faker, rng: random.Random) -> DeliveryNote:
    lines = []
    for line in make_lines(rng, rng.choice(list(CATALOGUES)), rng.randint(2, 9)):
        livree = line.qty if rng.random() < 0.8 else max(0, line.qty - rng.randint(1, line.qty))
        lines.append(DeliveryLine(line.ref, line.designation, line.qty, livree))
    missing = [ln for ln in lines if ln.livree < ln.commandee]
    observations = (
        f"Reliquat de {len(missing)} référence(s) à livrer ultérieurement."
        if missing
        else rng.choice(["Livraison complète.", "Marchandise conforme.", "RAS.", ""])
    )
    numero = rng.choice(["BL-{n:05d}", "BL{y}-{n:04d}", "LIV/{n:05d}", "B{n:06d}"])
    day = make_date(rng)
    return DeliveryNote(
        expediteur=make_party(fake, rng),
        destinataire=make_party(fake, rng),
        numero=numero.format(y=day.year, n=rng.randint(1, 99999)),
        date=day,
        commande=f"CMD-{rng.randint(1000, 99999)}",
        transporteur=rng.choice(
            [
                "Transports Martin",
                "Geodis",
                "DB Schenker",
                "Heppner",
                "Livraison interne",
                "Kuehne+Nagel",
                "Transports Bernard",
            ]
        ),
        lines=lines,
        colis=rng.randint(1, 12),
        poids_kg=round(rng.uniform(1, 400), 1),
        observations=observations,
    )


# --- Tickets (notes de frais) --------------------------------------------------------------------

RECEIPT_CATEGORIES = ["restaurant", "carburant", "peage", "parking", "hotel", "taxi"]


@dataclass
class Receipt:
    categorie: str
    marchand: str
    rue: str
    ville: str
    siret: str
    tel: str
    date: date
    heure: time
    lignes: list[tuple[str, Decimal]]  # libellé, montant TTC
    taux: Decimal
    ttc: Decimal
    tva: Decimal
    paiement: str
    numero: str
    extra: list[str] = field(default_factory=list)

    @property
    def ht(self) -> Decimal:
        return self.ttc - self.tva


RESTAURANT_ITEMS = [
    ("Menu du jour", 13.5, 19.9),
    ("Plat du jour", 11.0, 17.5),
    ("Entrecôte frites", 18.0, 26.0),
    ("Salade composée", 10.0, 15.0),
    ("Croque-monsieur", 7.5, 11.0),
    ("Café", 1.8, 2.6),
    ("Dessert du jour", 5.5, 8.5),
    ("Eau minérale 50 cl", 3.0, 4.5),
    ("Pizza margherita", 10.0, 14.0),
    ("Burger maison", 14.0, 19.0),
    ("Verre de vin", 4.5, 7.0),
    ("Thé", 2.5, 3.5),
]


def make_receipt(fake: Faker, rng: random.Random, categorie: str | None = None) -> Receipt:
    categorie = categorie or rng.choice(RECEIPT_CATEGORIES)
    lignes: list[tuple[str, Decimal]] = []
    extra: list[str] = []
    taux = Decimal("20")
    if categorie == "restaurant":
        marchand = (
            rng.choice(
                [
                    "Le Bistrot",
                    "Brasserie",
                    "Café",
                    "Restaurant",
                    "Pizzeria",
                    "Le Comptoir",
                    "La Table",
                ]
            )
            + " "
            + fake.last_name()
        )
        taux = Decimal("10")
        covers = rng.randint(1, 4)
        for name, low, high in rng.sample(RESTAURANT_ITEMS, k=rng.randint(2, 6)):
            qty = rng.randint(1, covers)
            lignes.append((f"{qty} x {name}", money(qty * rng.uniform(low, high))))
        extra.append(f"Table {rng.randint(1, 40)}   Couverts {covers}")
    elif categorie == "carburant":
        marchand = rng.choice(["Station", "Relais", "Aire de"]) + " " + fake.city()
        produit = rng.choice(["Gazole", "SP95-E10", "SP98", "E85"])
        litres = Decimal(str(round(rng.uniform(15, 65), 2)))
        prix = Decimal(str(round(rng.uniform(1.55, 1.99), 3)))
        lignes.append((f"{produit} {litres} L x {prix}", money(litres * prix)))
        extra.append(f"Pompe {rng.randint(1, 12)}")
    elif categorie == "peage":
        marchand = rng.choice(["Autoroutes du Sud", "Autoroute A", "Sanef", "APRR", "Vinci"])
        if marchand == "Autoroute A":
            marchand += str(rng.randint(1, 89))
        lignes.append((f"Classe {rng.choice([1, 1, 2])}", money(rng.uniform(2.2, 48.0))))
        extra.append(f"Gare de {fake.city()}  Voie {rng.randint(1, 20)}")
    elif categorie == "parking":
        marchand = rng.choice(["Parking", "Parc relais", "Indigo"]) + " " + fake.city()
        heures = rng.randint(1, 30)
        lignes.append((f"Stationnement {heures} h", money(heures * rng.uniform(1.2, 3.8))))
        extra.append(f"Entrée {rng.randint(6, 12):02d}:{rng.randint(0, 59):02d}")
    elif categorie == "hotel":
        marchand = (
            rng.choice(["Hôtel", "Hôtel du", "Résidence", "Ibis", "Kyriad"]) + " " + (fake.city())
        )
        taux = Decimal("10")
        nuits = rng.randint(1, 4)
        lignes.append((f"Nuitée x {nuits}", money(nuits * rng.uniform(55, 160))))
        if rng.random() < 0.6:
            lignes.append(("Petit-déjeuner", money(nuits * rng.uniform(8, 16))))
        lignes.append(("Taxe de séjour", money(nuits * rng.uniform(0.8, 3.3))))
        extra.append(f"Chambre {rng.randint(1, 5)}{rng.randint(1, 30):02d}")
    else:  # taxi
        marchand = rng.choice(["Taxi", "Taxis", "VTC"]) + " " + fake.last_name()
        taux = Decimal("10")
        km = round(rng.uniform(3, 45), 1)
        lignes.append((f"Course {km} km", money(km * rng.uniform(1.6, 2.4) + 4.2)))
        extra.append(f"Véhicule n° {rng.randint(1, 999)}")

    ttc = sum((amount for _, amount in lignes), Decimal(0))
    tva = money(ttc * taux / (100 + taux))
    siren = make_siren(rng)
    return Receipt(
        categorie=categorie,
        marchand=marchand.upper() if rng.random() < 0.6 else marchand,
        rue=fake.street_address(),
        ville=f"{fake.postcode()} {fake.city()}",
        siret=make_siret(siren, rng),
        tel=fake.phone_number(),
        date=make_date(rng),
        heure=time(rng.randint(6, 23), rng.randint(0, 59)),
        lignes=lignes,
        taux=taux,
        ttc=ttc,
        tva=tva,
        paiement=rng.choice(["CB", "CB SANS CONTACT", "ESPECES", "CB", "CARTE BANCAIRE"]),
        numero=f"{rng.randint(1, 9999):04d}",
        extra=extra,
    )


# --- « Autres » documents ------------------------------------------------------------------------

OTHER_KINDS = ["lettre", "attestation", "convocation", "formulaire", "compte_rendu"]


@dataclass
class OtherDoc:
    kind: str
    emetteur: Party
    destinataire_nom: str
    destinataire_adresse: list[str]
    date: date
    ville: str
    objet: str
    paragraphes: list[str]
    signataire: str
    reference: str


def make_other(fake: Faker, rng: random.Random, kind: str | None = None) -> OtherDoc:
    kind = kind or rng.choice(OTHER_KINDS)
    person = fake.name()
    org = rng.choice(
        [
            f"Mairie de {fake.city()}",
            f"Lycée {fake.last_name()}",
            "Caisse d'assurance maladie",
            f"Association {fake.last_name()}",
            None,
        ]
    )
    emetteur = make_party(fake, rng)
    if org:
        emetteur.nom = org
    objets = {
        "lettre": rng.choice(
            [
                "Demande de rendez-vous",
                "Réclamation",
                "Changement d'adresse",
                "Résiliation de contrat",
                "Information",
            ]
        ),
        "attestation": rng.choice(
            [
                "Attestation de domicile",
                "Attestation employeur",
                "Attestation de stage",
                "Attestation de présence",
            ]
        ),
        "convocation": rng.choice(
            [
                "Convocation à un entretien",
                "Convocation à l'assemblée générale",
                "Convocation à une réunion",
            ]
        ),
        "formulaire": rng.choice(
            [
                "Demande d'inscription",
                "Fiche de renseignements",
                "Demande de remboursement de frais médicaux",
                "Bulletin d'adhésion",
            ]
        ),
        "compte_rendu": rng.choice(
            ["Compte rendu de réunion", "Procès-verbal", "Note d'information interne"]
        ),
    }
    return OtherDoc(
        kind=kind,
        emetteur=emetteur,
        destinataire_nom=person,
        destinataire_adresse=[fake.street_address(), f"{fake.postcode()} {fake.city()}"],
        date=make_date(rng),
        ville=fake.city(),
        objet=objets[kind],
        paragraphes=[
            fake.paragraph(nb_sentences=rng.randint(3, 6)) for _ in range(rng.randint(2, 5))
        ],
        signataire=fake.name(),
        reference=f"{fake.lexify('??').upper()}/{rng.randint(100, 9999)}",
    )
