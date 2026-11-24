"""Partie 1 — espace « Mes documents » : documents rangés et façon dont ils ont été triés."""

import mimetypes

import pandas as pd
import streamlit as st
from docrouter_core import LABELS, REVIEW_FOLDER

from docrouter_ui.common import FOLDER_HEX, FOLDER_LABELS, folder_badge, get_registry, settings
from docrouter_ui.nav import page
from docrouter_ui.preview import make_thumbnail
from docrouter_ui.registry import FiledDocument

SORTS = {
    "Plus récents": lambda d: -d.filed_at.timestamp(),
    "Confiance croissante": lambda d: d.confidence if d.confidence is not None else 2.0,
    "Nom": lambda d: d.original_name.lower(),
}


def sorting_status(doc: FiledDocument) -> str:
    if doc.predicted_folder is None:
        return "Ajouté à la main"
    if doc.corrected:
        return "Corrigé"
    if doc.folder == REVIEW_FOLDER:
        return "À vérifier"
    return "Automatique"


def render_overview(docs: list[FiledDocument]) -> None:
    scored = [d.confidence for d in docs if d.confidence is not None]
    auto = [d for d in docs if sorting_status(d) == "Automatique"]
    corrected = [d for d in docs if d.corrected]
    pending = [d for d in docs if d.folder == REVIEW_FOLDER]

    cols = st.columns(4)
    cols[0].metric("Rangés", len(docs), border=True)
    cols[1].metric(
        "Tri auto",
        f"{len(auto) / len(docs):.0%}",
        border=True,
        help="Rangés dans le dossier proposé par le modèle, sans correction.",
    )
    cols[2].metric("Corrigés", len(corrected), border=True, help="Reclassés à la main.")
    cols[3].metric(
        "Confiance moy.", f"{sum(scored) / len(scored):.0%}" if scored else "—", border=True
    )
    if pending:
        st.warning(
            f"{len(pending)} document(s) attendent une vérification dans « À vérifier ».",
            icon=":material/help:",
        )

    with st.container(border=True):
        st.markdown("**Répartition par dossier**")
        for folder in FOLDER_LABELS:
            count = sum(d.folder == folder for d in docs)
            label_col, bar_col, count_col = st.columns([2, 6, 1], vertical_alignment="center")
            label_col.markdown(folder_badge(folder))
            bar_col.html(
                f"<div style='background:rgba(255,255,255,.06);border-radius:6px;height:12px'>"
                f"<div style='width:{count / len(docs):.0%};background:{FOLDER_HEX[folder]};"
                f"height:12px;border-radius:6px'></div></div>"
            )
            count_col.markdown(f"**{count}**")


def render_table(docs: list[FiledDocument]) -> FiledDocument | None:
    table = pd.DataFrame(
        {
            "Document": [d.original_name for d in docs],
            "Dossier": [FOLDER_LABELS.get(d.folder, d.folder) for d in docs],
            "Type prédit": [LABELS[d.predicted_type] if d.predicted_type else None for d in docs],
            "Confiance": [d.confidence for d in docs],
            "Tri": [sorting_status(d) for d in docs],
            "Rangé le": [d.filed_at for d in docs],
        }
    )
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Confiance": st.column_config.ProgressColumn(
                "Confiance", min_value=0, max_value=1, format="percent"
            ),
            "Rangé le": st.column_config.DatetimeColumn("Rangé le", format="DD/MM/YYYY HH:mm"),
        },
        key="documents_table",
    )
    rows = event.selection.rows
    return docs[rows[0]] if rows else None


def render_detail(doc: FiledDocument) -> None:
    registry = get_registry()
    path = settings.output_dir / doc.path
    data = path.read_bytes()

    with st.container(border=True):
        preview, info = st.columns([1, 2])
        thumbnail = make_thumbnail(data)
        if thumbnail:
            preview.image(thumbnail, width="stretch")
        else:
            preview.markdown(":material/draft: *Aperçu indisponible*")

        info.subheader(doc.original_name)
        info.markdown(f"Rangé dans {folder_badge(doc.folder)} · **{sorting_status(doc)}**")
        if doc.predicted_type is not None:
            proposed = folder_badge(doc.predicted_folder)
            info.markdown(
                f"Le modèle avait prédit **{LABELS[doc.predicted_type]}** "
                f"avec {doc.confidence:.0%} de confiance → proposé : {proposed}"
            )
        info.caption(
            f"Fichier : `{doc.path}` · Rangé le {doc.filed_at:%d/%m/%Y à %H:%M}"
            + (f" · {doc.page_count} page(s)" if doc.page_count else "")
            + (f" · modèle `{doc.model}`" if doc.model else "")
        )

        folders = list(FOLDER_LABELS)
        target = info.selectbox(
            "Reclasser dans",
            folders,
            index=folders.index(doc.folder),
            format_func=lambda f: FOLDER_LABELS[f],
            key=f"move_{doc.path}",
        )
        move_col, download_col = info.columns(2)
        if move_col.button(
            "Déplacer",
            icon=":material/drive_file_move:",
            disabled=target == doc.folder,
            width="stretch",
        ):
            registry.move(doc, target)
            st.toast(f"{doc.original_name} → {FOLDER_LABELS[target]}", icon=":material/check:")
            st.rerun()
        download_col.download_button(
            "Télécharger",
            data,
            file_name=doc.original_name,
            icon=":material/download:",
            mime=mimetypes.guess_type(doc.original_name)[0] or "application/octet-stream",
            width="stretch",
        )


def render() -> None:
    st.title("Mes documents")
    st.caption("Tous les documents rangés, et la façon dont ils ont été triés.")

    docs = get_registry().all()
    if not docs:
        st.info("Aucun document rangé pour l'instant.", icon=":material/inbox:")
        st.page_link(page("depot"), label="Déposer des documents", icon=":material/upload_file:")
        return

    render_overview(docs)

    st.subheader("Liste des documents")
    search_col, folder_col, sort_col = st.columns([2, 3, 1])
    search = search_col.text_input("Rechercher", placeholder="Nom du fichier…")
    folders = folder_col.multiselect(
        "Dossiers",
        list(FOLDER_LABELS),
        format_func=lambda f: FOLDER_LABELS[f],
        placeholder="Tous les dossiers",
    )
    sort = sort_col.selectbox("Trier par", list(SORTS))

    shown = [
        d
        for d in docs
        if (not folders or d.folder in folders) and search.lower() in d.original_name.lower()
    ]
    shown.sort(key=SORTS[sort])
    if not shown:
        st.info("Aucun document ne correspond aux filtres.", icon=":material/filter_alt_off:")
        return

    selected = render_table(shown)
    if selected is None:
        st.caption(
            ":material/touch_app: Sélectionnez une ligne pour voir le document et le reclasser."
        )
    else:
        render_detail(selected)
