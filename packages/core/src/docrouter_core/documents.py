"""Types de documents gérés par DocRouter et règles de routage (spec §3)."""

from enum import StrEnum


class DocumentType(StrEnum):
    FACTURE = "facture"
    NOTE_DE_FRAIS = "note_de_frais"
    BON_DE_LIVRAISON = "bon_de_livraison"
    AUTRE = "autre"


LABELS: dict[DocumentType, str] = {
    DocumentType.FACTURE: "Facture",
    DocumentType.NOTE_DE_FRAIS: "Note de frais",
    DocumentType.BON_DE_LIVRAISON: "Bon de livraison",
    DocumentType.AUTRE: "Autre",
}

FOLDERS: dict[DocumentType, str] = {
    DocumentType.FACTURE: "factures",
    DocumentType.NOTE_DE_FRAIS: "notes_de_frais",
    DocumentType.BON_DE_LIVRAISON: "bons_de_livraison",
    DocumentType.AUTRE: "autres",
}

REVIEW_FOLDER = "a_verifier"


def destination_for(doc_type: DocumentType, confidence: float, threshold: float) -> str:
    """Dossier de destination : celui du type, ou `a_verifier` si le modèle n'est pas assez sûr."""
    if confidence < threshold:
        return REVIEW_FOLDER
    return FOLDERS[doc_type]
