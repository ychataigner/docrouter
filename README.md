# DocRouter

API de routage de documents administratifs par vision par ordinateur : on dépose un document
(facture, note de frais, bon de livraison, autre), il est classé automatiquement puis rangé
dans le bon dossier. Spécifications complètes : [SPECIFICATIONS.md](SPECIFICATIONS.md).

> **Partie 1** : tri par IA avec une sonde CLIP exécutée en ONNX INT8 (97,5 % sur le jeu de
> test, voir [docs/iterations.md](docs/iterations.md)). Sans la surcouche IA, l'API utilise un
> **tri simulé** (`dummy`) pour développer le front.

## Résultats clés

| | Valeur | Détails |
|---|---|---|
| Modèle | Sonde linéaire sur CLIP, exécutée en ONNX INT8 | 5 modèles comparés dans MLflow |
| Précision (jeu de test, mise en page jamais vue) | **97,5 %** (DiT pré-entraîné : 35 %) | [docs/iterations.md](docs/iterations.md) |
| Temps de réponse | ≈ 90 ms par document (modèle : 54 ms) | itérations 4 et 5 |
| Image Docker de l'API | **538 Mo** (PyTorch : 1,87 Go) | itération 4 |
| Débit | 11,5 doc/s à 2 vCPU (PyTorch : 2,95), 20,5 doc/s à 4 vCPU, 0 erreur | itérations 5 et 8 |
| Suivi | Historique des requêtes et comparaison des modèles dans MLflow | itération 8 |
| Sécurité | Clé d'API par client, aucun document écrit sur disque | itération 6 |
| Budget, 1 M documents / mois | **≈ 6 $/mois** pour l'API (≈ 34 $ avec front et MLflow) | [BUDGET.md](BUDGET.md) |

Parcours itération par itération (choix, mesures, problèmes rencontrés) :
[docs/iterations.md](docs/iterations.md). Spécifications : [SPECIFICATIONS.md](SPECIFICATIONS.md).

## Architecture

| Service | Rôle | Port |
|---|---|---|
| `api` (FastAPI) | Prétraitement + classification. Ne conserve aucun fichier. | 8000 (Swagger : `/docs`) |
| `ui` (Streamlit) | Dépôt, correction du type, rangement dans `output/<dossier>/` | 8501 |

Dossiers de sortie : `factures/`, `notes_de_frais/`, `bons_de_livraison/`, `autres/`,
`a_verifier/` (confiance < seuil, 60 % par défaut).

## Lancer avec Docker (mode front, sans IA)

Créer d'abord le fichier `.env` (jamais versionné) avec une clé d'API pour le front :

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # clé à recopier dans .env
#   DOCROUTER_API_KEYS=front:<clé>   (clés acceptées par l'API, une par client)
#   DOCROUTER_API_KEY=<clé>          (clé envoyée par le front)
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

En production, le réseau est **exporté en ONNX et quantifié en INT8** : l'API tourne avec
onnxruntime, sans PyTorch (image 538 Mo au lieu de 1,87 Go, RAM 231 Mo, même précision).

```bash
uv sync --group train
uv run docrouter-evaluate --model probe-clip   # tête → models/probes/probe-clip.npz (14 Ko)
uv run docrouter-export-onnx                   # réseau → models/onnx/clip/model_int8.onnx (90 Mo)
docker compose -f docker-compose.yml -f docker-compose.ml.yml up --build -d   # image docrouter-api:ml
```

Les deux fichiers sont montés depuis `models/` (non versionné). La tête est au format `.npz`
sans pickle. Variante PyTorch (modèle lourd de référence) et ancien modèle DiT : voir
`docker-compose.ml.yml` (`ML_EXTRA=ml`, `DOCROUTER_CLASSIFIER=dit`).

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

Modèles comparés sur les mêmes 120 documents de test : `dit-zeroshot` (le premier modèle de l'API),
`clip-zeroshot`, et des sondes linéaires (`probe-dit`, `probe-vit`, `probe-clip`) : réseau gelé +
régression logistique entraînée sur `train` ; `probe-clip-onnx` et `probe-clip-int8` mesurent
la version exportée, avec le code de l'API. Chaque modèle donne un run MLflow (accuracy,
F1-macro et F1 par classe, précision par effet et par format, seuil « à vérifier » recommandé,
latence p50/p95, taille) avec la matrice de confusion et la courbe du seuil en artefacts.
Tableau récapitulatif : `reports/eval/comparison.md` ; têtes entraînées : `models/probes/`.

## Suivi MLflow et historique des requêtes (itération 8)

MLflow tourne avec le compose (http://localhost:5000). L'API y envoie l'historique de ses
requêtes (expérience **`docrouter-api`**) : durées, confiance, type, client, taux « à vérifier »,
sans aucun contenu du document. Les autres expériences comparent les modèles
(`docrouter-classification`) et les campagnes de charge (`docrouter-load`).

```bash
uv run docrouter-api-history     # tableau et graphique des derniers runs de l'API
```

`DOCROUTER_MLFLOW_URI` vide désactive le suivi ; `API_RUN_LABEL` nomme le run, par exemple
pour comparer PyTorch et ONNX sous la même charge.

## Test de charge (itération 5)

```powershell
uv sync --group train
$env:API_CPUS = 2                                        # 4 : ajouter $env:API_INFERENCES = 2
docker compose -f docker-compose.yml -f docker-compose.ml.yml -f docker-compose.load.yml up -d api
$env:DOCROUTER_API_KEY = "<clé du front, voir .env>"
uv run locust -f load_test/locustfile.py --headless --host http://localhost:8000   # ≈ 3 min
uv run docrouter-load-report reports/load/requests.csv --label "2 vCPU" --cpus 2
```

Sans `--headless`, Locust ouvre une page web (http://localhost:8089) avec les graphiques en
direct. Résultat de référence : **≈ 11,5 doc/s à 2 vCPU, ≈ 20,5 doc/s à 4 vCPU**, sans erreur
jusqu'à 32 utilisateurs simultanés (détails : [docs/iterations.md](docs/iterations.md)).
Réglages utiles : `DOCROUTER_ONNX_THREADS` (2) et `DOCROUTER_MAX_CONCURRENT_INFERENCES`
(nombre de vCPU / 2).

## Sécurité et RGPD (itération 6)

**Authentification.** Les routes `/v1/*` exigent l'en-tête `X-API-Key`. Chaque client a sa clé
(`DOCROUTER_API_KEYS="front:<clé>,compta:<clé>"`) : les journaux indiquent qui appelle, et on
révoque un client en retirant sa clé, sans toucher aux autres. Sans clé configurée, l'API refuse
tout (elle ne s'ouvre pas par défaut). Les clés ne sont lues que dans l'environnement (`.env`,
non versionné), comparées en temps constant, et jamais journalisées. Dans Swagger, bouton
**Authorize**. `/health` et `/docs` restent publics.

**Aucune image conservée par l'API.**
- Le document est traité **en mémoire** : Starlette écrit normalement sur disque tout fichier
  reçu de plus de 1 Mo ; ce comportement est neutralisé (`security.py`), ce que vérifie un test.
- Clé et taille sont contrôlées **avant** la lecture du document : un appel refusé ne transmet rien.
- Le conteneur de l'API est en **lecture seule** (`read_only`, `/tmp` en mémoire, aucun
  privilège) : il ne peut physiquement pas enregistrer un document.
- Les journaux ne contiennent que des métadonnées : identifiant de requête, client, type prédit,
  confiance, durées. Ni image, ni nom de fichier, ni clé (vérifié par un test).

| Question RGPD | Réponse pour DocRouter |
|---|---|
| Données traitées | Factures, notes de frais, BL : noms, adresses, montants, parfois données de carte bancaire (tickets) |
| Finalité | Trier les documents pour les ranger dans le bon dossier |
| Conservation par l'API | **Aucune** : traitement en mémoire, durée de la requête |
| Conservation par le front | Les documents rangés (`output/`) restent chez l'utilisateur : c'est le service rendu. Il peut les supprimer à tout moment |
| Accès | Clé d'API par client, révocable ; aucun accès anonyme |
| Transferts | Aucun envoi à un service tiers : le modèle tourne dans le conteneur (pas d'API externe). Les seuls téléchargements sont ceux des poids du modèle |
| Chiffrement | En production, HTTPS obligatoire devant l'API (reverse proxy ou service cloud) ; les clés circulent dans un en-tête, donc jamais en clair hors HTTPS |
| Données de développement | Uniquement des documents **fictifs** (générateur) : aucune donnée personnelle réelle pour entraîner ou tester |

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
packages/core/            code partagé (types de documents, règles de rangement, schémas)
services/api/             API FastAPI + Dockerfile (extras : onnx = production, ml = PyTorch)
services/ui/              front Streamlit + Dockerfile
training/                 données, évaluation, export ONNX, rapport de charge, budget
load_test/                scénario Locust (itération 5)
tests/                    tests pytest (lancés par la CI GitHub Actions)
docs/iterations.md        journal : choix, mesures et problèmes, itération par itération
BUDGET.md                 estimation budgétaire (itération 9)
docker-compose*.yml       base (API + front + MLflow), .ml (IA), .load (test de charge)
```

Commandes du dossier `training/` (après `uv sync --group train`) :

| Commande | Rôle |
|---|---|
| `docrouter-generate` | génère les documents fictifs et leur vérité terrain |
| `docrouter-evaluate` | compare les modèles sur le jeu de test (MLflow) |
| `docrouter-export-onnx` | exporte le réseau en ONNX et le quantifie en INT8 |
| `docrouter-load-report` | analyse une campagne de charge Locust (MLflow) |
| `docrouter-budget` | recalcule le budget à partir des mesures et des tarifs |
| `docrouter-api-history` | résume l'historique des requêtes de l'API (MLflow) |

Ajouter une dépendance à un service : `uv add --package docrouter-api <paquet>`.
