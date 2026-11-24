"""Partie 1 — dépôt, classification, correction et rangement des documents."""

import mimetypes
from datetime import datetime

import streamlit as st
from docrouter_core import FOLDERS, LABELS, ClassifyResponse

from docrouter_ui.client import ApiError
from docrouter_ui.common import (
    FOLDER_LABELS,
    folder_badge,
    get_client,
    get_registry,
    settings,
)
from docrouter_ui.filing import file_document
from docrouter_ui.nav import page
from docrouter_ui.preview import make_thumbnail
from docrouter_ui.registry import FiledDocument

ACCEPTED_TYPES = ["pdf", "png", "jpg", "jpeg"]


def init_state() -> None:
    st.session_state.setdefault("documents", {})  # clé fichier → document analysé
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("filed", [])


def reset() -> None:
    st.session_state.documents = {}
    st.session_state.uploader_key += 1


def analyze(files) -> None:
    client = get_client()
    progress = st.progress(0.0, text="Analyse en cours…")
    for i, uploaded in enumerate(files, start=1):
        data = uploaded.getvalue()
        content_type = (
            uploaded.type or mimetypes.guess_type(uploaded.name)[0] or "application/octet-stream"
        )
        doc = {"name": uploaded.name, "data": data, "result": None, "error": None}
        try:
            result = client.classify(uploaded.name, data, content_type)
            doc["result"] = result
            doc["folder"] = result.classification.destination
        except ApiError as exc:
            doc["error"] = str(exc)
        st.session_state.documents[uploaded.file_id] = doc
        progress.progress(i / len(files), text=f"Analyse : {uploaded.name} ({i}/{len(files)})")
    progress.empty()


def render_summary(documents: dict) -> None:
    analysed = [d for d in documents.values() if d["result"]]
    counts = dict.fromkeys(FOLDER_LABELS, 0)
    for doc in analysed:
        counts[doc["folder"]] += 1
    cols = st.columns(len(FOLDER_LABELS) + 1)
    cols[0].metric("Documents", len(documents), border=True)
    for col, (folder, label) in zip(cols[1:], FOLDER_LABELS.items(), strict=True):
        col.metric(label, counts[folder], border=True)
    errors = len(documents) - len(analysed)
    if errors:
        st.warning(f"{errors} document(s) n'ont pas pu être analysés.", icon=":material/error:")


def render_document(key: str, doc: dict) -> None:
    with st.container(border=True):
        preview, info = st.columns([1, 3])
        thumbnail = make_thumbnail(doc["data"])
        if thumbnail:
            preview.image(thumbnail, width=160)
        else:
            preview.markdown(":material/draft: *Aperçu indisponible*")

        info.markdown(f"**{doc['name']}**")
        if doc["error"]:
            info.error(doc["error"], icon=":material/error:")
            return

        result: ClassifyResponse = doc["result"]
        clf = result.classification
        label = LABELS[clf.type]
        if clf.needs_review:
            info.warning(
                f"Prédiction incertaine : **{label}** ({clf.confidence:.0%}). Vérifiez le type.",
                icon=":material/help:",
            )
        else:
            info.markdown(f"{folder_badge(FOLDERS[clf.type])} confiance **{clf.confidence:.0%}**")
        info.progress(clf.confidence)

        folders = list(FOLDER_LABELS)
        doc["folder"] = info.selectbox(
            "Ranger dans",
            folders,
            index=folders.index(doc["folder"]),
            format_func=lambda f: f"{FOLDER_LABELS[f]}  ({f}/)",
            key=f"folder_{key}",
        )
        with info.expander("Détails"):
            st.json(
                {
                    "request_id": result.request_id,
                    "pages": result.page_count,
                    "modele": result.model,
                    "label_brut": clf.raw_label,
                    "scores": {LABELS[t]: s for t, s in clf.scores.items()},
                    "temps_ms": result.timings_ms,
                }
            )


def file_all(documents: dict) -> None:
    registry = get_registry()
    filed = []
    for doc in documents.values():
        result: ClassifyResponse | None = doc["result"]
        if result is None:
            continue
        clf = result.classification
        path = file_document(
            settings.output_dir, doc["folder"], doc["name"], result.request_id, doc["data"]
        )
        registry.add(
            FiledDocument(
                id=result.request_id,
                original_name=doc["name"],
                path=path.relative_to(settings.output_dir).as_posix(),
                folder=doc["folder"],
                filed_at=datetime.now(),
                predicted_type=clf.type,
                predicted_folder=clf.destination,
                confidence=clf.confidence,
                needs_review=clf.needs_review,
                model=result.model,
                page_count=result.page_count,
            )
        )
        filed.append((doc["name"], doc["folder"]))
    st.session_state.filed = filed
    reset()


def render_filed() -> None:
    filed = st.session_state.filed
    if not filed:
        return
    with st.container(border=True):
        st.success(f"{len(filed)} document(s) rangé(s).", icon=":material/check_circle:")
        for name, folder in filed:
            st.markdown(f"`{name}` → {folder_badge(folder)}")
        st.page_link(page("documents"), label="Voir mes documents", icon=":material/folder_open:")
    st.session_state.filed = []


def render() -> None:
    init_state()
    health = st.session_state.get("health")
    api_ok = health is not None and health.model_ready

    st.title("Dépôt de documents")
    st.caption(
        "Déposez vos factures, notes de frais, bons de livraison ou autres documents : "
        "ils sont classés automatiquement puis rangés dans le bon dossier."
    )
    render_filed()

    files = st.file_uploader(
        "Documents (PDF, PNG, JPEG)",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    to_analyse = [f for f in files or [] if f.file_id not in st.session_state.documents]
    if st.button(
        f"Analyser {len(to_analyse)} document(s)",
        type="primary",
        icon=":material/document_scanner:",
        disabled=not (api_ok and to_analyse),
    ):
        analyze(to_analyse)
        st.rerun()

    documents = st.session_state.documents
    if not documents:
        return

    st.subheader("Résultats")
    render_summary(documents)
    for key, doc in documents.items():
        render_document(key, doc)

    ready = sum(1 for d in documents.values() if d["result"])
    file_col, clear_col, _ = st.columns([2, 2, 4])
    if file_col.button(
        f"Ranger {ready} document(s)",
        type="primary",
        icon=":material/drive_file_move:",
        disabled=not ready,
        width="stretch",
    ):
        file_all(documents)
        st.rerun()
    if clear_col.button("Vider la liste", icon=":material/delete_sweep:", width="stretch"):
        reset()
        st.rerun()
