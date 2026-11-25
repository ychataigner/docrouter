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

## Itération 2 — Le Cerveau ✅ (DiT pré-entraîné, puis sonde CLIP retenue sur mesures)

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

## Données — banc d'évaluation et comparaison des modèles ✅

- **Pourquoi** : choisir le modèle sur des mesures, avec les mêmes documents et les mêmes métriques
  pour tous, et garder une trace de chaque essai.
- **Réalisé** : `uv run docrouter-evaluate` (groupe `train`) et serveur **MLflow** dans
  `docker-compose.yml` (SQLite + artefacts dans le volume `mlflow-data`). Chaque run enregistre
  l'empreinte du jeu de données et le commit Git. Métriques : accuracy, F1-macro, F1 par classe,
  précision par effet et par format, courbe couverture/précision du seuil « à vérifier »,
  latence p50/p95, taille. Artefacts : matrice de confusion, courbe du seuil, prédictions, tête
  entraînée.
- **Sonde linéaire** : le réseau reste gelé et sert à résumer chaque image en un vecteur ; seule une
  régression logistique à 4 classes est entraînée (sur `train`, régularisation choisie sur `val`).
  Extraction ≈ 3 min par modèle sur CPU, entraînement de la tête < 1 s.
- **Résultats** (120 documents de test, **gabarit jamais vu**, CPU 12 cœurs) :

| Modèle | Accuracy | F1-macro | F1 fact. | F1 NdF | F1 BL | F1 autre | Photo | p50 (ms) | Taille (Mo) |
|---|---|---|---|---|---|---|---|---|---|
| dit-zeroshot (API actuelle) | 35,0 % | 0,227 | 0,37 | 0,00 | 0,00 | 0,54 | 21 % | 185 | 343 |
| clip-zeroshot | 59,2 % | 0,540 | 0,59 | 0,97 | 0,00 | 0,60 | 69 % | 106 | 605 |
| probe-dit | 78,3 % | 0,768 | 0,53 | 0,83 | 0,72 | 0,98 | 79 % | 171 | 343 |
| probe-vit | 72,5 % | 0,704 | 0,31 | 0,97 | 0,61 | 0,93 | 86 % | 174 | 346 |
| **probe-clip** | **97,5 %** | **0,975** | 0,95 | 1,00 | 0,95 | 1,00 | 97 % | **89** | 351 |

- **Constats** :
  - Sans adaptation, aucun modèle n'est utilisable (DiT 35 %, CLIP 59 %, BL jamais reconnu).
  - En validation (gabarits déjà vus), les trois sondes dépassent 99 % : le vrai juge est bien le
    gabarit inédit du test, où DiT et ViT chutent (78 %, 72 %) — ils ont appris les mises en page.
    CLIP généralise (97,5 %) : il a été entraîné sur des images avec du texte, son vecteur capte
    davantage le *contenu*.
  - Les 3 erreurs de probe-clip sont des factures prises pour des BL : exactement la difficulté
    voulue du gabarit de test (colonne latérale commune).
  - probe-clip est aussi le plus rapide (89 ms) ; il ne garde que la partie « image » de CLIP.
  - Seuil : avec probe-clip, 97,5 % des documents passent le seuil actuel (0,6) avec 98,3 % de
    bons rangements.
- **Prudence** : 30 documents par classe (une erreur = 0,8 point), documents synthétiques. Le seuil
  recommandé par le banc est calculé sur le test lui-même, donc optimiste. À confirmer sur
  de vrais documents (`data/test_reel/`).
- **Décision** : **probe-clip** devient le candidat pour l'API ; le fine-tuning complet n'est pas
  nécessaire à ce stade.

## Sonde CLIP branchée dans l'API ✅

- **Réalisé** : `DOCROUTER_CLASSIFIER=probe` (par défaut dans `docker-compose.ml.yml`).
  - L'extracteur d'images (`ImageEmbedder`) et la tête (`LinearHead`) vivent dans l'API ;
    l'entraînement les réutilise, donc les vecteurs de production sont calculés exactement comme
    à l'entraînement.
  - Seule la partie « image » de CLIP est chargée.
  - La tête (14 Ko) est en `.npz` **sans pickle** : elle ne peut pas contenir de code, et l'API
    n'a pas besoin de scikit-learn. Elle est montée depuis `models/probes/`, ou téléchargée
    depuis le Hub avec `hf://…`.
  - **Échauffement au démarrage** : la 1re inférence PyTorch coûtait 11 s sur la 1re requête.
- **Vérification de bout en bout** : les 120 documents de test envoyés à l'API conteneurisée
  donnent 117/120 corrects, soit les mêmes 3 erreurs que le banc (factures → BL, dont une
  envoyée dans `a_verifier`).
  - Latence par requête : p50 113 ms, p95 176 ms ; modèle seul : p50 84 ms.
  - RAM du conteneur : 1,16 Go (contre ≈ 750 Mo avec DiT). Image : 1,87 Go, inchangée.
  - Rangement : 30 autres, 31 BL, 26 factures, 30 notes de frais, 3 `a_verifier`.

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
