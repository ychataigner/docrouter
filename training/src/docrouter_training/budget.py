"""Itération 9 — estimation budgétaire (FinOps) pour un volume mensuel de documents.

    uv run docrouter-budget                         # 1 million de documents / mois
    uv run docrouter-budget --docs 10000000 --markdown

Le calcul repose sur deux sources, séparées ci-dessous :
- MESURES : nos chiffres réels (tests de charge Locust de l'itération 5, comparaison PyTorch /
  ONNX de l'itération 4). Les remplacer si le modèle ou l'image changent.
- TARIFS : prix publics relevés le 25/11/2026 (USD, hors taxes, à vérifier avant engagement).
  Si un prix change, on modifie la constante et on relance : le budget reste reproductible.
"""

import argparse
import math
from dataclasses import dataclass

HOURS_PER_MONTH = 730
SECONDS_PER_MONTH = HOURS_PER_MONTH * 3600

# --- MESURES (docs/iterations.md) ----------------------------------------------------------
MEASURED = {
    # Débit soutenu d'un conteneur ONNX INT8, sans erreur (itération 5)
    "onnx_docs_per_s_2vcpu": 11.5,
    "onnx_docs_per_s_4vcpu": 20.5,
    # Temps d'une requête isolée côté serveur (prétraitement + modèle), p50, 2 vCPU
    "onnx_request_s": 0.106,
    # Modèle lourd (PyTorch), mesuré dans les mêmes conditions (itération 8 : 2 vCPU, Locust)
    "torch_docs_per_s_2vcpu": 2.95,
    "torch_request_s": 0.333,
    # Démarrage à froid jusqu'à « prêt » (itération 4)
    "onnx_cold_start_s": 4,
    "torch_cold_start_s": 24,
    # Images Docker (Go) et taille moyenne d'un document envoyé (Mo)
    "onnx_image_gb": 0.54,
    "torch_image_gb": 1.87,
    "avg_document_mb": 0.2,
    "response_kb": 0.6,
}

# --- TARIFS (USD, relevés le 25/11/2026) ---------------------------------------------------
PRICES = {
    # Google Cloud Run, facturation à la requête (arrondi aux 100 ms supérieures)
    "run_vcpu_s": 0.000024,
    "run_gib_s": 0.0000025,
    "run_per_million_requests": 0.40,
    "run_free_vcpu_s": 180_000,
    "run_free_gib_s": 360_000,
    "run_free_requests": 2_000_000,
    # Cloud Run, instance toujours allouée (min-instances = 1)
    "run_always_vcpu_s": 0.000018,
    "run_always_gib_s": 0.000002,
    # Compute Engine e2-standard-2 (2 vCPU, 8 Go), Paris europe-west9, à la demande
    "vm_e2_standard_2_h": 0.078,
    # Compute Engine e2-medium (2 vCPU partagés, 4 Go) : front Streamlit + MLflow
    "vm_e2_medium_h": 0.038,
    # Hugging Face Inference Endpoints : par cœur CPU et par heure, et GPU NVIDIA T4
    "hf_cpu_core_h": 0.03,
    "hf_t4_h": 0.50,
    # Divers : registre d'images (Go-mois), sortie réseau (Go)
    "registry_gb_month": 0.10,
    "egress_gb": 0.12,
}

SOURCES = {
    "Cloud Run": "https://cloud.google.com/run/pricing",
    "Compute Engine (e2, Paris)": "https://gcloud-compute.com/e2-standard-2.html",
    "Hugging Face Inference Endpoints": "https://huggingface.co/docs/inference-endpoints/pricing",
}


@dataclass
class Traffic:
    docs_per_month: int
    working_days: int = 22  # documents envoyés en jours ouvrés …
    hours_per_day: int = 10  # … pendant les heures de bureau
    peak_factor: float = 3.0  # pic = 3 × la moyenne des heures de bureau

    @property
    def office_rate(self) -> float:
        return self.docs_per_month / (self.working_days * self.hours_per_day * 3600)

    @property
    def peak_rate(self) -> float:
        return self.office_rate * self.peak_factor


@dataclass
class Scenario:
    name: str
    monthly_usd: float
    detail: str
    capacity_note: str

    def per_1000_docs(self, docs: int) -> float:
        return self.monthly_usd / docs * 1000


def _billed_seconds(request_s: float) -> float:
    """Cloud Run arrondit le temps facturé de chaque requête aux 100 ms supérieures."""
    return math.ceil(request_s * 10 - 1e-9) / 10


def cloud_run_on_demand(t: Traffic, request_s: float, vcpu: int = 2, gib: int = 2) -> float:
    """Borne haute : chaque requête facturée seule (pas de recouvrement entre requêtes)."""
    seconds = t.docs_per_month * _billed_seconds(request_s)
    cpu = max(0.0, seconds * vcpu - PRICES["run_free_vcpu_s"]) * PRICES["run_vcpu_s"]
    mem = max(0.0, seconds * gib - PRICES["run_free_gib_s"]) * PRICES["run_gib_s"]
    req = (
        max(0, t.docs_per_month - PRICES["run_free_requests"])
        / 1e6
        * PRICES["run_per_million_requests"]
    )
    return cpu + mem + req


def cloud_run_always_on(instances: int, vcpu: int = 2, gib: int = 2) -> float:
    per_instance = SECONDS_PER_MONTH * (
        vcpu * PRICES["run_always_vcpu_s"] + gib * PRICES["run_always_gib_s"]
    )
    return instances * per_instance


def instances_needed(t: Traffic, docs_per_s: float, minimum: int = 1) -> int:
    return max(minimum, math.ceil(t.peak_rate / docs_per_s))


def network_and_storage(t: Traffic, image_gb: float) -> float:
    egress_gb = t.docs_per_month * MEASURED["response_kb"] / 1e6
    return image_gb * PRICES["registry_gb_month"] + egress_gb * PRICES["egress_gb"]


def scenarios(t: Traffic) -> list[Scenario]:
    m, p = MEASURED, PRICES
    extra = network_and_storage(t, m["onnx_image_gb"])
    run_min = instances_needed(t, m["onnx_docs_per_s_2vcpu"])  # dimensionné sur le pic
    vm_ha = max(2, run_min)
    torch_run = instances_needed(t, m["torch_docs_per_s_2vcpu"])
    hf_endpoints = instances_needed(t, m["torch_docs_per_s_2vcpu"])
    side = p["vm_e2_medium_h"] * HOURS_PER_MONTH  # front + MLflow
    return [
        Scenario(
            "A. Cloud Run à la requête — ONNX INT8 (recommandé)",
            cloud_run_on_demand(t, m["onnx_request_s"]) + extra,
            "2 vCPU / 2 Gio, 0 instance au repos, facturation à la requête "
            f"({_billed_seconds(m['onnx_request_s']):.1f} s facturées par document)",
            f"démarrage à froid ≈ {m['onnx_cold_start_s']} s après une période sans trafic",
        ),
        Scenario(
            "B. Cloud Run, instances toujours prêtes — ONNX INT8",
            cloud_run_always_on(run_min) + extra,
            f"min-instances = {run_min} (pas de démarrage à froid), 2 vCPU / 2 Gio",
            f"{m['onnx_docs_per_s_2vcpu']:.1f} doc/s par instance ; autoscaling au-delà",
        ),
        Scenario(
            f"C. VM e2-standard-2 × {vm_ha} (haute dispo) — ONNX INT8",
            vm_ha * p["vm_e2_standard_2_h"] * HOURS_PER_MONTH + extra,
            "docker compose tel quel (API + front + MLflow), 2 VM pour la redondance",
            f"{vm_ha * m['onnx_docs_per_s_2vcpu']:.0f} doc/s au total, dimensionnement fixe",
        ),
        Scenario(
            "D. Cloud Run à la requête — modèle lourd PyTorch",
            cloud_run_on_demand(t, m["torch_request_s"])
            + network_and_storage(t, m["torch_image_gb"]),
            "même plateforme que A, image 1,87 Go",
            f"démarrage à froid ≈ {m['torch_cold_start_s']} s : en pratique il faut B "
            f"(≈ {cloud_run_always_on(torch_run):.0f} $) pour garder des temps corrects",
        ),
        Scenario(
            "E. Hugging Face Inference Endpoint CPU (géré)",
            hf_endpoints * 2 * p["hf_cpu_core_h"] * HOURS_PER_MONTH,
            f"{hf_endpoints} endpoint(s) 2 vCPU toujours allumé(s), PyTorch",
            "aucune infrastructure à gérer ; données envoyées chez un tiers (RGPD)",
        ),
        Scenario(
            "F. GPU NVIDIA T4 dédié (HF Endpoint)",
            p["hf_t4_h"] * HOURS_PER_MONTH,
            "1 GPU T4 allumé en permanence",
            "inutile ici : le modèle répond en 54 ms sur CPU ; GPU utile pour entraîner",
        ),
        Scenario(
            "+ Front Streamlit et MLflow (VM e2-medium)",
            side,
            "à ajouter à A, B, D, E ou F (déjà inclus dans C)",
            "pourrait être remplacé par un front serverless",
        ),
    ]


def report(t: Traffic) -> str:
    lines = [
        f"Volume : {t.docs_per_month:,} documents / mois".replace(",", " "),
        f"Débit moyen en heures de bureau : {t.office_rate:.2f} doc/s ; "
        f"pic estimé (× {t.peak_factor:g}) : {t.peak_rate:.2f} doc/s",
        "",
        "| Scénario | Coût / mois | Coût / 1 000 docs | Détail | Remarque |",
        "|---|---|---|---|---|",
    ]
    for s in scenarios(t):
        lines.append(
            f"| {s.name} | {s.monthly_usd:,.2f} $ | {s.per_1000_docs(t.docs_per_month):.4f} $ "
            f"| {s.detail} | {s.capacity_note} |"
        )
    return "\n".join(lines)


# Scénarios comparés selon le volume (le GPU, dont le débit n'a pas été mesuré, n'y figure pas)
SENSITIVITY = {
    "A. Cloud Run à la requête": 0,
    "B. Cloud Run toujours prêt": 1,
    "C. VM × N (HA)": 2,
    "E. HF Endpoint CPU": 4,
}


def _usd(value: float) -> str:
    return f"{value:,.0f} $".replace(",", " ")


def sensitivity(volumes: list[int]) -> str:
    lines = [
        "| Documents / mois | Pic (doc/s) | " + " | ".join(SENSITIVITY) + " |",
        "|---" * (len(SENSITIVITY) + 2) + "|",
    ]
    for v in volumes:
        t = Traffic(v)
        all_scenarios = scenarios(t)
        costs = [_usd(all_scenarios[i].monthly_usd) for i in SENSITIVITY.values()]
        volume = f"{v:,}".replace(",", " ")
        lines.append(f"| {volume} | {t.peak_rate:.1f} | " + " | ".join(costs) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--docs", type=int, default=1_000_000, help="documents par mois")
    parser.add_argument("--peak-factor", type=float, default=3.0)
    parser.add_argument("--markdown", action="store_true", help="ajoute la sensibilité au volume")
    args = parser.parse_args(argv)
    print(report(Traffic(args.docs, peak_factor=args.peak_factor)))
    if args.markdown:
        print("\n" + sensitivity([100_000, 1_000_000, 10_000_000, 100_000_000]))


if __name__ == "__main__":
    main()
