"""Déclaration des pages, accessible depuis les vues pour créer des liens entre elles."""

import streamlit as st


def build_pages() -> dict[str, list[st.Page]]:
    # Import local : les vues importent ce module pour leurs liens (`page(...)`).
    from docrouter_ui.views import comptabilite, depot, documents

    _PAGES.update(
        depot=st.Page(
            depot.render,
            title="Dépôt de documents",
            icon=":material/upload_file:",
            url_path="depot",
            default=True,
        ),
        documents=st.Page(
            documents.render,
            title="Mes documents",
            icon=":material/folder_open:",
            url_path="documents",
        ),
        comptabilite=st.Page(
            comptabilite.render,
            title="Journal comptable",
            icon=":material/account_balance:",
            url_path="comptabilite",
        ),
    )
    return {
        "Partie 1 · Routage": [_PAGES["depot"], _PAGES["documents"]],
        "Partie 2 · Comptabilité": [_PAGES["comptabilite"]],
    }


def page(name: str) -> st.Page:
    return _PAGES[name]


_PAGES: dict[str, st.Page] = {}
