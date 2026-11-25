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

## Activer le tri par IA (sonde CLIP)

Le modèle retenu est une **sonde linéaire sur CLIP** : la partie « image » de
`openai/clip-vit-base-patch32`, gelée, suivie d'une tête à 4 classes entraînée sur nos documents
(97,5 % sur le jeu de test, contre 35 % pour DiT non adapté ; voir la comparaison plus bas).

```bash
uv sync --group train && uv run docrouter-evaluate --model probe-clip   # produit models/probes/probe-clip.npz (14 Ko)
docker compose -f docker-compose.yml -f docker-compose.ml.yml up --build -d   # image docrouter-api:ml
```

Au premier démarrage, l'API télécharge CLIP (~600 Mo) dans le volume `hf-cache`. La tête est
montée depuis `models/probes/` (format `.npz` sans pickle). En local : `uv sync --group ml` et
`DOCROUTER_CLASSIFIER=probe`. L'ancien modèle reste disponible avec `DOCROUTER_CLASSIFIER=dit`.

## Générer les documents de test (fictifs)

```bash
uv run docrouter-generate                 # 1 320 documents dans data/synth (≈ 2 min)
uv run docrouter-generate --train 20 --val 5 --test 5 --out data/synth_mini
```

Factures, bons de livraison, tickets de caisse et « autres » documents français, entièrement
fictifs (Faker `fr_FR`) : 4 mises en page par classe, effets scan / photo, et vérité terrain
(fournisseur, date, n°, HT, TVA, TTC) dans `data/synth/manifest.csv`. Le gabarit n° 3 est réservé
au jeu de test (`data/synth/test/`), qu'on peut déposer directement dans le front.

## Comparer les modèles (MLflow)

```bash
uv sync --group train              # torch CPU, transformers, scikit-learn, client MLflow
docker compose up -d mlflow        # serveur de suivi : http://localhost:5000
uv run docrouter-evaluate          # les 5 modèles sur data/synth/test (≈ 10 min la 1re fois)
uv run docrouter-evaluate --model probe-dit --no-mlflow
```

Modèles comparés sur les mêmes 120 documents de test : `dit-zeroshot` (le modèle actuel de l'API),
`clip-zeroshot`, et des sondes linéaires (`probe-dit`, `probe-vit`, `probe-clip`) : réseau gelé +
régression logistique entraînée sur `train`. Chaque modèle donne un run MLflow (accuracy,
F1-macro et F1 par classe, précision par effet et par format, seuil « à vérifier » recommandé,
latence p50/p95, taille) avec la matrice de confusion et la courbe du seuil en artefacts.
Tableau récapitulatif : `reports/eval/comparison.md` ; têtes entraînées : `models/probes/`.

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
training/        génération de données, entraînement, évaluation (hors images Docker)
tests/           tests pytest
```

Ajouter une dépendance à un service : `uv add --package docrouter-api <paquet>`.
