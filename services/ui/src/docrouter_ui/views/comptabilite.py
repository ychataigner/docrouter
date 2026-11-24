"""Partie 2 — journal comptable (à venir) : OCR, extraction des champs, écritures (spec §7)."""

import pandas as pd
import streamlit as st

from docrouter_ui.common import folder_badge, get_registry
from docrouter_ui.nav import page

# Colonnes du futur export, inspirées du FEC (spec §7.4)
JOURNAL_COLUMNS = [
    "JournalCode",
    "EcritureDate",
    "CompteNum",
    "CompteLib",
    "PieceRef",
    "EcritureLib",
    "Debit",
    "Credit",
    "Statut",
    "FichierSource",
]

STEPS = [
    (
        ":material/document_scanner:",
        "1. Lecture (OCR)",
        "Le texte des factures et notes de frais rangées est extrait automatiquement.",
    ),
    (
        ":material/data_object:",
        "2. Extraction",
        "Fournisseur, date, n° de pièce, montants HT / TVA / TTC, avec contrôle HT + TVA = TTC.",
    ),
    (
        ":material/account_balance:",
        "3. Écritures",
        "Une écriture équilibrée par document, exportée en CSV et Excel.",
    ),
]


def render() -> None:
    st.title("Journal comptable")
    st.badge("À venir — partie 2", icon=":material/construction:", color="violet")
    st.caption(
        "À partir des documents triés en partie 1, DocRouter lira leur contenu et pré-remplira "
        "un journal comptable."
    )

    cols = st.columns(len(STEPS))
    for col, (icon, title, text) in zip(cols, STEPS, strict=True):
        with col.container(border=True, height="stretch"):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(text)

    docs = get_registry().all()
    ready = {
        folder: sum(d.folder == folder for d in docs) for folder in ("factures", "notes_de_frais")
    }
    with st.container(border=True):
        st.markdown("**Documents qui seront comptabilisés**")
        a, b = st.columns(2)
        a.markdown(f"{folder_badge('factures')} **{ready['factures']}**")
        b.markdown(f"{folder_badge('notes_de_frais')} **{ready['notes_de_frais']}**")
        st.caption(
            "Les bons de livraison ne génèrent pas d'écriture : ils serviront au rapprochement."
        )
        if not any(ready.values()):
            st.page_link(
                page("depot"), label="Déposer des documents", icon=":material/upload_file:"
            )

    st.markdown("**Aperçu du format d'export**")
    st.dataframe(pd.DataFrame(columns=JOURNAL_COLUMNS), hide_index=True, width="stretch")
