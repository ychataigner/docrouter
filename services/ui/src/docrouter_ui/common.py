"""Éléments partagés entre les pages : client API, registre, libellés et couleurs des dossiers."""

import streamlit as st
from docrouter_core import FOLDERS, LABELS, REVIEW_FOLDER, HealthResponse

from docrouter_ui.client import ApiClient, ApiError
from docrouter_ui.registry import Registry
from docrouter_ui.settings import get_settings

FOLDER_LABELS = {FOLDERS[t]: LABELS[t] for t in FOLDERS} | {REVIEW_FOLDER: "À vérifier"}
FOLDER_ICONS = {
    "factures": ":material/receipt_long:",
    "notes_de_frais": ":material/payments:",
    "bons_de_livraison": ":material/local_shipping:",
    "autres": ":material/description:",
    REVIEW_FOLDER: ":material/help:",
}
# Couleurs de badge Streamlit (blue, green, orange, red, violet, gray…)
FOLDER_COLORS = {
    "factures": "blue",
    "notes_de_frais": "green",
    "bons_de_livraison": "orange",
    "autres": "gray",
    REVIEW_FOLDER: "red",
}
# Mêmes teintes en hexadécimal, pour les éléments dessinés à la main (barres de répartition)
FOLDER_HEX = {
    "factures": "#60A5FA",
    "notes_de_frais": "#4ADE80",
    "bons_de_livraison": "#FB923C",
    "autres": "#9CA3AF",
    REVIEW_FOLDER: "#F87171",
}

settings = get_settings()


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(
        settings.api_url,
        timeout=settings.api_timeout_s,
        api_key=settings.api_key.get_secret_value(),
    )


def get_registry() -> Registry:
    return Registry(settings.output_dir, list(FOLDER_LABELS))


def folder_badge(folder: str) -> str:
    """Badge coloré en Markdown pour un dossier."""
    color = FOLDER_COLORS.get(folder, "gray")
    return f":{color}-badge[{FOLDER_ICONS.get(folder, '')} {FOLDER_LABELS.get(folder, folder)}]"


def api_health() -> HealthResponse | None:
    """État de l'API, ou `None` si elle est injoignable (l'erreur est gardée en session)."""
    try:
        health = get_client().health()
    except ApiError as exc:
        st.session_state.api_error = str(exc)
        return None
    st.session_state.api_error = None
    return health


def render_sidebar_status() -> HealthResponse | None:
    health = api_health()
    with st.sidebar:
        if health is None:
            st.error(
                f"API indisponible\n\n{st.session_state.api_error}", icon=":material/cloud_off:"
            )
        elif health.model_ready:
            st.success("API connectée", icon=":material/cloud_done:")
        else:
            st.warning("Modèle non chargé", icon=":material/hourglass:")
        if health is not None:
            mode = "simulé (sans IA)" if health.model == "dummy" else health.model
            st.caption(
                f"Tri : **{mode}** · seuil {health.confidence_threshold:.0%} · "
                f"max {health.max_upload_mb} Mo · API v{health.version}"
            )
    return health
