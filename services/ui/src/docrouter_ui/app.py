"""Interface DocRouter : point d'entrée et navigation entre les deux parties du projet."""

import streamlit as st

from docrouter_ui.common import render_sidebar_status
from docrouter_ui.nav import build_pages

st.set_page_config(page_title="DocRouter", page_icon=":material/folder_open:", layout="wide")

st.logo(
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='180' height='36'>"
    "<text x='0' y='26' font-family='sans-serif' font-size='24' font-weight='700' "
    "fill='%236C8CFF'>Doc<tspan fill='%23E6E9F2'>Router</tspan></text></svg>"
)

navigation = st.navigation(build_pages())
st.session_state.health = render_sidebar_status()
navigation.run()
