# Journal des itérations

Pour chaque itération du sujet : la mission, ce qu'on a essayé (erreurs comprises), le résultat et
les mesures. Chaque itération terminée porte un tag Git (`git tag`).

---

## Étape 0 — Socle ✅

- Dépôt Git privé sur GitHub, tag `it1-it3` sur l'état initial.
- CI GitHub Actions ([.github/workflows/ci.yml](../.github/workflows/ci.yml)) : `ruff`, `pytest`
  et build des deux images à chaque push. Premier run vert en 1 min 13 s.
- Tests classés en catégories : `ml`, `slow`, `integration` (exclues par défaut).

## Itération 1 — Le Squelette ✅ (tag `it1-it3`)

- **Mission** : API web qui reçoit une image en POST, avec Swagger.
- **Réalisé** : FastAPI, `GET /health`, `POST /v1/classify` (PDF, PNG, JPEG ; 10 Mo max),
  réponses typées avec Pydantic (`packages/core`), Swagger sur `/docs`.
- **Choix** : l'API est **sans état** et ne garde aucun fichier (préparation de l'itération 6).
  C'est le client (Streamlit) qui range les documents.

## Itération 2 — Le Cerveau ✅ (modèle pré-entraîné, fine-tuning à venir)

- **Mission** : brancher un modèle Hugging Face et renvoyer une vraie prédiction.
- **Réalisé** : DiT (`microsoft/dit-base-finetuned-rvlcdip`) tourne dans Docker
  (`docker compose -f docker-compose.yml -f docker-compose.ml.yml up -d`). Le tri « simulé »
  (`DOCROUTER_CLASSIFIER=dummy`) reste disponible pour développer le front sans PyTorch.
- **Mesures** (CPU, 4 documents fictifs de `data/samples/`) :

  | Document | Prédiction | Confiance | Label brut DiT | Inférence |
  |---|---|---|---|---|
  | Facture | facture ✅ | 80 % | `invoice` | 132 ms |
  | Bon de livraison | autre ❌ | 98 % | `form` | 161 ms |
  | Ticket de restaurant | autre ❌ | 77 % | `budget` | 127 ms |
  | Courrier de mairie | autre ✅ | 99 % | `letter` | 132 ms |

  Image `docrouter-api:ml` : **1,87 Go** (contre 288 Mo sans IA). RAM du conteneur : **≈ 750 Mo**.
- **Constat** : DiT ne connaît que les 16 classes RVL-CDIP. Seule `invoice` correspond à l'une de
  nos classes : un BL ou un ticket ne peut **jamais** être reconnu, et il est classé « autre » avec
  une forte confiance, donc sans passer par « à vérifier ». Un fine-tuning sur nos 4 classes est
  indispensable.
- **Incident** : le disque C: s'est rempli pendant le build (Docker occupait 60 Go). Résultat : une
  image corrompue (fichiers Python vides) et une distribution WSL `docker-desktop` bloquée.
  Corrections : nettoyage (30 Go de cache de build), mise à jour de WSL, disque Docker déplacé sur
  D:, puis rebuild.

## Données — générateur de documents français ✅

- **Pourquoi** : aucun dataset public ne contient des factures, BL et tickets **français**. Sans
  données étiquetées, on ne peut ni adapter le modèle ni mesurer s'il fonctionne.
- **Réalisé** : paquet `training/`, commande `uv run docrouter-generate`.
  - Contenu fictif cohérent : SIRET valides (clé de Luhn), TVA intracommunautaire calculée,
    HT + TVA = TTC au centime près, taux 20 / 10 / 5,5 %.
  - 4 classes × **4 mises en page** structurellement différentes, et un style (police, couleur,
    tableau) tiré au hasard pour chaque document.
  - Effets **propre / scan** (rotation, papier jauni, bruit, basse résolution) / **photo**
    (perspective, fond, ombre, éclairage inégal). Les tickets sont surtout photographiés.
  - **Gabarit 3 réservé au test** : on mesure si le modèle reconnaît le *type* de document, et
    pas une mise en page déjà vue.
- **Résultat** : 1 320 documents (240 train + 60 val + 30 test par classe) en 2 min sur 11 cœurs,
  254 Mo. `manifest.csv` contient la vérité terrain (utile pour la partie 2) ; `dataset_info.json`
  contient la graine et l'empreinte SHA-256, qui seront enregistrées dans MLflow.
- **Limites** : les documents sont dessinés et restent plus réguliers que des vrais. À compléter
  par des photos de vrais tickets. Les gabarits de test de la facture et du BL partagent une colonne
  latérale : c'est un test difficile et voulu, puisque seul le contenu les distingue.

## Itération 3 — L'Isolement ✅ (tag `it1-it3`)

- **Mission** : Dockerfile optimisé.
- **Réalisé** : Dockerfiles multi-stage avec uv, utilisateur non-root, healthchecks,
  `docker-compose.yml` (API + front).
- **Erreur et correction** : le premier build durait plus de 50 min pour environ 2 Go d'images.
  Cause : PyTorch (814 Mo) et PyMuPDF, qui tirait onnxruntime. Solution : IA en option
  (`WITH_ML=true`) et passage à pypdfium2. Résultat : API 288 Mo, front 858 Mo, build en environ 8 min.

## Itération 4 — L'Optimisation ⏳
## Itération 5 — Le Flux ⏳
## Itération 6 — La Barrière ⏳
## Itération 7 — Le Grand Saut ⏳
## Itération 8 — Le Rétroviseur ⏳
## Itération 9 — La Facture ⏳
