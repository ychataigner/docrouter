"""Itération 6 — authentification par clé d'API et garde-fous sur les données reçues.

- **Une clé par client** (`DOCROUTER_API_KEYS="front:<clé>,compta:<clé>"`) : on sait qui appelle
  (journaux), et on révoque un client en retirant sa clé, sans toucher aux autres.
- Les clés arrivent par variable d'environnement (`.env`, jamais versionné) et sont comparées en
  temps constant (`secrets.compare_digest`).
- Les contrôles (clé, taille) ont lieu **avant la lecture du corps de la requête** : un appel
  non autorisé ou trop gros est refusé sans que le document soit reçu.
- Le document reçu reste **en mémoire** : Starlette écrit sinon sur disque tout fichier de plus
  de 1 Mo (fichier temporaire). Aucune image n'est donc écrite sur le serveur, même brièvement.
"""

import logging
import secrets
from dataclasses import dataclass

from starlette.formparsers import MultiPartParser

logger = logging.getLogger("docrouter.security")

API_KEY_HEADER = "X-API-Key"
MIN_KEY_LENGTH = 24
MULTIPART_OVERHEAD = 64 * 1024  # en-têtes multipart autour du fichier


@dataclass(frozen=True)
class ApiKeys:
    clients: dict[str, str]  # clé → nom du client

    @classmethod
    def parse(cls, raw: str) -> "ApiKeys":
        """`"front:abc…,compta:def…"` ; une clé sans nom devient `client-1`, `client-2`…"""
        clients: dict[str, str] = {}
        for i, entry in enumerate(e.strip() for e in raw.split(",") if e.strip()):
            name, _, key = entry.rpartition(":")
            key, name = key.strip(), name.strip() or f"client-{i + 1}"
            if len(key) < MIN_KEY_LENGTH:
                raise ValueError(
                    f"Clé d'API « {name} » trop courte ({len(key)} caractères, "
                    f"minimum {MIN_KEY_LENGTH}). En générer une : "
                    'python -c "import secrets; print(secrets.token_urlsafe(32))"'
                )
            clients[key] = name
        return cls(clients)

    def __bool__(self) -> bool:
        return bool(self.clients)

    def identify(self, key: str | None) -> str | None:
        """Nom du client si la clé est valide, sinon None (comparaison en temps constant)."""
        if not key:
            return None
        found = None
        for known, name in self.clients.items():
            if secrets.compare_digest(key.encode(), known.encode()):
                found = name
        return found


def keep_uploads_in_memory(max_upload_bytes: int) -> None:
    """Empêche Starlette d'écrire les fichiers reçus sur disque (au-delà de 1 Mo par défaut).

    Sans risque pour la mémoire : la taille de la requête est vérifiée avant sa lecture.
    """
    MultiPartParser.spool_max_size = max_upload_bytes + MULTIPART_OVERHEAD
