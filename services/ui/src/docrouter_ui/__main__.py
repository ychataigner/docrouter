"""Point d'entrée `docrouter-ui` : lance Streamlit sur l'application du paquet."""

import sys
from pathlib import Path

from streamlit.web import cli

# Taille alignée sur DOCROUTER_MAX_UPLOAD_MB de l'API ; le menu « Deploy » n'a pas lieu d'être ici.
DEFAULT_OPTIONS = [
    "--server.maxUploadSize=10",
    "--client.toolbarMode=minimal",
    # Thème sombre DocRouter
    "--theme.base=dark",
    "--theme.primaryColor=#6C8CFF",
    "--theme.backgroundColor=#0F1320",
    "--theme.secondaryBackgroundColor=#1A2033",
    "--theme.textColor=#E6E9F2",
]


def main() -> None:
    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *DEFAULT_OPTIONS, *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
