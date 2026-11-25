# DocRouter

API de routage de documents administratifs par vision par ordinateur : on dépose un document
(facture, note de frais, bon de livraison, autre), il est classé automatiquement puis rangé
dans le bon dossier. Spécifications complètes : [SPECIFICATIONS.md](SPECIFICATIONS.md).

> **État actuel (partie 1)** : le tri par IA est **optionnel**. Par défaut, l'API utilise un
> **tri simulé** (`dummy`, sans IA) pour développer et tester le front. Le modèle DiT
> s'active avec l'option `ml` (voir plus bas).

## Architecture

| Service | Rôle | Port |
|---|---|---|
| `api` (FastAPI) | Prétraitement + classification. Ne conserve aucun fichier. | 8000 (Swagger : `/docs`) |
| `ui` (Streamlit) | Dépôt, correction du type, rangement dans `output/<dossier>/` | 8501 |

Dossiers de sortie : `factures/`, `notes_de_frais/`, `bons_de_livraison/`, `autres/`,
`a_verifier/` (confiance < seuil, 60 % par défaut).

## Lancer avec Docker (mode front, sans IA)

```bash
docker compose up --build -d
```

- Front : http://localhost:8501
- API : http://localhost:8000/docs

Pour modifier le front sans rebuild : `docker compose watch` (le code de `services/ui` est
recopié dans le conteneur à chaque sauvegarde et la page se recharge).

Le tri simulé renvoie toujours le même résultat pour un même document, et couvre les 4 classes
ainsi que des cas incertains (→ `a_verifier`).

## Activer le tri par IA (DiT)

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml up --build -d   # Docker (image docrouter-api:ml)
uv sync --group ml                                                            # local
```

Au premier démarrage, l'API télécharge le modèle (~350 Mo) dans le volume `hf-cache`.

## Développement local (uv)

```bash
uv sync                                         # Python 3.11 + dépendances (sans IA)
uv run uvicorn docrouter_api.main:app --reload  # API (tri simulé)
uv run docrouter-ui                             # front
uv run pytest                                   # tests
uv run ruff check . && uv run ruff format .     # lint / format
```

## Structure

```
packages/core/   code partagé (types de documents, schémas)
services/api/    API FastAPI + Dockerfile (extra `ml` = DiT + PyTorch)
services/ui/     front Streamlit + Dockerfile
tests/           tests pytest
```

Ajouter une dépendance à un service : `uv add --package docrouter-api <paquet>`.
