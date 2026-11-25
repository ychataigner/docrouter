# Plan de travail — DocRouter (suite du projet)

> État au 24/11/2026. Ce document liste ce qu'il reste à faire, **itération par itération du sujet**
> ([Projet vision par ordinateur.md](Projet%20vision%20par%20ordinateur.md)).
> [SPECIFICATIONS.md](SPECIFICATIONS.md) reste la référence fonctionnelle.

---

## 0. Où on en est

| Bloc | État |
|---|---|
| API FastAPI (`/health`, `/v1/classify`), prétraitement PDF/images, Swagger | ✅ fait (it. 1) |
| Front Streamlit (dépôt, « Mes documents », menu partie 2, thème sombre) | ✅ fait (hors sujet, bonus) |
| Docker (API légère 288 Mo, front 858 Mo), uv workspace, 27 tests, ruff | ✅ fait (it. 3, version sans IA) |
| Modèle | ⚠️ **tri simulé par défaut** (`dummy`). DiT est branché en option (`WITH_ML=true`) mais n'a **jamais été évalué**. Non fine-tuné, il ne connaît que `invoice` → il **ne peut jamais prédire** « note de frais » ni « bon de livraison » |
| Données d'entraînement / de test | ❌ rien |
| ONNX, MLflow, tests de charge, auth, cloud, budget | ❌ rien |
| Partie 2 (OCR, extraction, journal) | ❌ seulement la page « À venir » |
| Dépôt Git | ❌ **pas encore un dépôt Git**, alors que c'est le livrable final |

**Où tourne le modèle ?** En local : les poids sont téléchargés une fois depuis le Hub Hugging Face
(volume `hf-cache`), puis l'inférence se fait sur le CPU du conteneur. Aucune API HF n'est appelée.

---

## 1. Conformité au sujet, itération par itération

| It. | Ce que le sujet exige | Specs / plan précédent | Écart → correction |
|---|---|---|---|
| **1 Squelette** | API, POST d'une image, Swagger | ✅ fait | — |
| **2 Cerveau** | modèle **issu de Hugging Face**, prétraitement, prédiction **réelle** en JSON | DiT branché mais désactivé par défaut | Le livrable exige une vraie prédiction : il faut activer et **valider DiT dans Docker**. Le fine-tuning **n'est pas demandé** par le sujet, mais il est indispensable pour avoir nos 4 classes (voir §3, phase 2) |
| **3 Isolement** | Dockerfile **optimisé** | ✅ image légère. L'image avec IA (torch) n'a pas été mesurée récemment | Mesurer et documenter l'image IA. Après l'it. 4, l'image de prod n'embarque **que onnxruntime** (pas torch) |
| **4 Optimisation** | ONNX + **quantization**, code mis à jour pour utiliser le modèle optimisé | ✅ prévu | Le modèle optimisé devient **le modèle par défaut** de l'API |
| **5 Flux** | script externe d'afflux continu, test de charge. Concept : **pipeline de données temps réel**, fraîcheur des données | Locust prévu. Le « dossier surveillé » était dans ce que je proposais de **supprimer** en premier | ❌ **Erreur corrigée** : le dossier de réception surveillé correspond justement au concept de pipeline de données, on le garde. On ajoute la **boucle de rétroaction** (corrections → nouvelles données étiquetées) qui répond à « actualisation des données » |
| **6 Barrière** | **token d'authentification** obligatoire, **aucune image conservée** après inférence. Concepts : **chiffrement**, gestion des accès, RGPD | JWT + test « aucun fichier » prévus | Manquait : **chiffrement** (HTTPS en cloud, secrets hors du code), **gestion des accès** (clés par client, révocables), le front qui doit envoyer le token, et le spool disque de Starlette au-delà de 1 Mo |
| **7 Grand saut** | déployer le conteneur dans le cloud (si possible). Concept : **choix des ressources, y compris GPU** | Cloud Run / Azure prévu | Ajouter une **justification CPU vs GPU** : CPU pour l'inférence ONNX, GPU (Colab/Kaggle) pour le fine-tuning |
| **8 Rétroviseur** | MLflow : **tracker le temps de réponse**, comparer **modèle lourd (it. 2) vs optimisé (it. 4)**. Livrable : **dashboard MLflow avec l'historique des requêtes** | MLflow prévu, mais seulement pour l'**évaluation hors ligne** des modèles | ❌ **Manquait** : c'est l'**API en fonctionnement** qui doit logger chaque requête dans MLflow (latence par étape, modèle et version, classe, confiance), puis on compare les deux modèles sous le même test de charge. Il faut aussi tracer les **corrections manuelles** (§6.1 des specs) |
| **9 Facture** | budget pour **1 M d'images/mois**, documentation. Livrable : **repo Git** (code, Dockerfile, README, budget) | prévu | Ajouter `git init` dès maintenant, un **tag Git par itération**, et une page qui retrace chaque itération (mission → ce qu'on a essayé → résultat), car la méthode du sujet est « chercher, tester, se tromper » |
| **Prérequis HF** | compte + token, HF sert à **héberger et récupérer** les modèles | token dans `.env` ✅ | Pousser **notre modèle fine-tuné (et sa version ONNX) sur le Hub HF** (dépôt privé), que l'image Docker télécharge avec le token : c'est l'usage d'HF prévu par le sujet |
| **Thème C** | classer des scans (notes de frais, factures) | partie 1 ✅ | La **partie 2 (OCR + journal) dépasse le sujet** : c'est une extension, à faire après le parcours obligatoire (ou en parallèle si le temps le permet) |

---

## 2. Quels modèles pour quelle tâche

| Tâche | Candidats à comparer |
|---|---|
| **A. Classification** (partie 1, sujet) | ① baseline OCR + TF-IDF + régression logistique · ② DiT zero-shot (actuel) · ③ **DiT fine-tuné 4 classes** (cible) · ④ CLIP/SigLIP zero-shot · ⑤ LayoutXLM (plan B) |
| **B. OCR** (partie 2) | Tesseract `fra` · docTR |
| **C. Extraction de champs** (partie 2) | ① lecture du XML Factur-X, sans IA · ② regex + contrôles de cohérence · ③ Donut · ④ LLM/VLM avec sortie JSON (via API) |

**Local ou API Hugging Face ?**
- **Classification : en local, avec ONNX.** C'est le cœur des itérations 3, 4, 5 et 8. Un modèle hébergé
  ailleurs n'aurait rien à optimiser, et le document sortirait de l'infra (contraire à l'itération 6).
- **HF sert à héberger nos modèles** (Hub), pas à faire tourner l'inférence. C'est ce que dit le sujet.
- **Les API d'inférence HF servent de comparaison** pour l'extraction de la partie 2 (un VLM contre
  OCR + règles) et pour l'**itération 9** (coût à la requête contre serveur fixe, avec la limite RGPD
  à expliquer). DiT n'est probablement pas proposé en serverless : à vérifier.

---

## 3. Feuille de route

L'ordre suit les itérations. Chaque phase se termine par un **tag Git** et une entrée dans
`docs/iterations.md`.

### Phase 0 — Socle
- [x] `git init`, premier commit, `.gitignore` vérifié (`data/`, `models/`, `output/`, `.env`, `mlruns/`)
- [x] Tag `it1-it3` sur l'existant
- [x] CI GitHub Actions : `ruff` + `pytest` (tests rapides) + build des images API et front
- [x] Marqueurs pytest : `ml`, `slow`, `integration` (exclus par défaut)
- [x] `docs/iterations.md` : une section par itération (mission, essais, erreurs, résultat, mesures)

### Phase 1 — It. 2 + 3 « pour de vrai » : DiT réel dans Docker
- [x] Construire l'image avec IA, obtenir une vraie prédiction, mesurer (1,87 Go, ≈ 130 ms, ≈ 750 Mo de RAM)
- [x] Constat documenté : sans adaptation, DiT ne reconnaît que les factures (BL → `form`, ticket → `budget`)
- [x] Tag `it2`

### Phase 2 — Données, évaluation, fine-tuning (pas exigé par le sujet, mais nécessaire pour les 4 classes)
- [x] Générateur `docrouter-generate` (`training/`) : Faker `fr_FR` → dessin PIL, **4 gabarits par
      classe** (0-2 entraînement, 3 test), styles aléatoires, effets « propre / scan / photo »,
      **vérité terrain** dans `manifest.csv` (réutilisée en partie 2). 1 320 docs en 2 min, 254 Mo
- [ ] Classe `autre` : compléter avec un sous-ensemble RVL-CDIP / Cerfa publics (si la précision le demande)
- [x] Jeu de test figé : `data/synth/test/` (120 docs, gabarit 3 jamais vu à l'entraînement, 25 % en PDF)
- [ ] Compléter le test avec des **photos de vrais tickets** (numéros de carte masqués) → `data/test_reel/`
- [x] MLflow dans `docker-compose.yml` (backend SQLite + artefacts en volume)
- [x] `docrouter-evaluate --model <backend>` → MLflow : accuracy, F1-macro, matrice de confusion,
      courbe couverture/précision (pour fixer le seuil `a_verifier`), latence p50/p95, taille
- [x] **Comparer plusieurs modèles, pas seulement DiT**, avec le même jeu de test et les mêmes
      métriques dans MLflow : DiT zero-shot (actuel), CLIP/SigLIP zero-shot, OCR + TF-IDF (baseline),
      et au moins une autre architecture de vision adaptée (ViT ou EfficientNet).
      Fait : DiT et CLIP zero-shot, sondes DiT / ViT / CLIP → **probe-clip 97,5 %** (DiT 35 %).
      OCR + TF-IDF reporté à la partie 2 (Tesseract à installer)
- [x] Adaptation, en deux temps : ① **sonde linéaire** (backbone gelé, seule la tête à 4 classes est
      entraînée, sur CPU en quelques minutes) ; ② **fine-tuning complet** (GPU Colab/Kaggle)
      seulement si ① ne suffit pas. `training/finetune.py`, suivi MLflow
- [x] Le modèle retenu est choisi sur les mesures (précision, latence, taille) et le choix est justifié dans le journal
- [ ] Partie 2 : même démarche de comparaison pour l'OCR (Tesseract vs docTR) et l'extraction (règles vs Donut vs VLM)
- [x] Brancher probe-clip dans l'API (`DOCROUTER_CLASSIFIER=probe`, tête `.npz` sans pickle)
- [ ] Revoir le seuil `a_verifier` sur de vrais documents
- [ ] Modèle retenu poussé sur le **Hub HF** (privé), et l'API le télécharge avec `HF_TOKEN`
- [ ] Critères : accuracy ≥ 90 %, F1-macro ≥ 0,88 → c'est le **modèle lourd de référence** de l'it. 8

### Phase 3 — It. 4 : optimisation
- [x] Export ONNX (`torch.onnx`, `docrouter-export-onnx`) + quantization dynamique INT8
      (per_channel + reduce_range). Publication sur le Hub : à faire (`hf://` déjà géré par l'API)
- [x] `OnnxEmbedder` (onnxruntime + prétraitement numpy) : **backend par défaut**, l'image de prod perd torch
- [x] Comparaison lourd / optimisé : même accuracy (97,5 %), modèle 54 ms au lieu de 86 ms,
      image 538 Mo au lieu de 1,87 Go, RAM 231 Mo au lieu de 504 Mo
- [ ] Tag `it4`

### Phase 4 — It. 5 : flux
- [x] `load_test/` : Locust (montée en charge, p95, req/s, taux d'erreur) avec les docs du jeu de test
      → 11,5 doc/s à 2 vCPU, 20,5 à 4 vCPU, 0 erreur ; file d'attente des inférences ajoutée
- [ ] (facultatif) **Dossier de réception surveillé** (`watchdog`) : tout fichier déposé est envoyé à l'API et rangé
- [ ] (facultatif) **Boucle de rétroaction** : les corrections de « Mes documents » (`registre.jsonl`) sont exportées
      en jeu étiqueté pour le prochain fine-tuning
- [ ] Tag `it5`

### Phase 5 — It. 6 : sécurité et RGPD
- [x] Auth : clés d'API par client (révocables), `X-API-Key` sur `/v1/*`, bouton Authorize dans
      Swagger, clé envoyée par le front ; API fermée si aucune clé n'est configurée
- [x] Traitement 100 % en mémoire (spool disque de Starlette neutralisé, prouvé par un test),
      contrôles avant lecture du corps, conteneur en lecture seule
- [x] Logs sans contenu (ni image, ni nom de fichier, ni clé), secrets uniquement par variables d'environnement
- [x] Chiffrement : HTTPS en production (documenté), section RGPD du README
- [ ] Tag `it6`

### Phase 6 — It. 8 : MLflow en production
- [ ] L'API **logge chaque requête** dans MLflow (en tâche de fond, sans bloquer la réponse) :
      latence prétraitement / inférence / totale, modèle et version, classe, confiance, `a_verifier`,
      **sans aucun contenu du document**
- [ ] Même test Locust lancé sur le **modèle lourd** puis sur l'**ONNX** → deux expériences comparables
- [ ] Corrections manuelles du front loggées (taux d'erreur réel du modèle)
- [ ] **Livrable** : captures du dashboard MLflow (historique des requêtes, comparaison lourd / optimisé)
- [ ] Tag `it8`

### Phase 7 — It. 7 : cloud
- [ ] Déploiement de l'API (Cloud Run ou Azure Container Apps), secrets du fournisseur, HTTPS
- [ ] Le front pointe sur l'API distante ; MLflow reste en local ou sur une petite VM (à décider)
- [ ] Justification des ressources : CPU (inférence ONNX) vs GPU (entraînement seulement)
- [ ] Tag `it7`

### Phase 8 — It. 9 : FinOps et documentation
- [ ] `BUDGET.md` : 1 M docs/mois. Scénarios : CPU ONNX, modèle lourd, GPU, API HF à la requête,
      avec et sans OCR (partie 2), à partir des **mesures réelles** de Locust et MLflow
- [ ] README final (architecture, lancement, API, choix et résultats), `docs/iterations.md` complet
- [ ] Tag `it9` = rendu

### Extension — Partie 2 (après le parcours obligatoire)
- [ ] `ocr.py` (Tesseract vs docTR), `extraction.py` (Factur-X → regex → option VLM via API)
- [ ] `accounting.py`, `config/plan_comptable.yaml`, routes `/v1/extract`, `/v1/process`, `/v1/journal/export`
- [ ] Page « Journal comptable » réelle ; critères : TTC et date ≥ 85 %, 100 % d'écritures équilibrées
- [ ] Budget mis à jour (l'OCR est l'étape la plus coûteuse)

---

## 4. Tests à écrire

| Domaine | Tests unitaires | Autres |
|---|---|---|
| Prétraitement | PDF multipage, PDF chiffré/corrompu → 400, rotation EXIF, CMYK/RGBA/16 bits, bombe de décompression, formats refusés proprement | — |
| Classifieurs | mapping 16 → 4 classes complet et somme = 1, seuil `a_verifier` aux bornes, interface commune à tous les backends | **parité PyTorch / ONNX** (écart de logits < ε), marqueur `ml` |
| Qualité modèle | — | **non-régression** : F1-macro ≥ seuil sur un « golden set » de ~20 docs, `slow` |
| Données | générateur reproductible (seed), vérité terrain cohérente (HT + TVA = TTC), aucun gabarit commun train/test | — |
| Sécurité / RGPD (it. 6) | 401 sans token / token expiré / clé révoquée, **aucun fichier résiduel**, logs sans contenu | — |
| MLflow (it. 8) | le logging d'une requête contient les bons champs et aucun contenu ; une panne de MLflow ne casse pas l'API | — |
| Flux (it. 5) | le dossier surveillé envoie et range un fichier, export de la boucle de rétroaction | Locust : p95 < 1 s, ≥ 5 req/s |
| API | contrat (schémas, codes 400/401/413/422/503), `request_id` | intégration `docker compose`, marqueur `integration` |
| Front | registre et client (déjà faits), `streamlit.testing.AppTest` pour les pages | — |
| Partie 2 | montants FR, dates, SIRET (Luhn), TVA intracom, Factur-X, Σ débit = Σ crédit (`hypothesis`), export FEC | précision des champs sur le jeu généré, `slow` |

---

## 5. Mon avis

1. **Le sujet évalue l'industrialisation, pas la qualité du modèle.** On suit donc les itérations dans
   l'ordre, avec une trace écrite de chaque essai. Le fine-tuning est la seule entorse justifiée, parce
   que sans lui le routage en 4 classes ne marche pas.
2. **La principale erreur du plan précédent concernait l'itération 8** : MLflow doit suivre les
   **requêtes réelles de l'API**, pas seulement l'évaluation hors ligne. C'est le livrable.
3. **Seconde erreur, l'itération 5** : le dossier surveillé et la boucle de rétroaction sont exactement le
   « pipeline de données » demandé, on ne les supprime pas.
4. **Les données restent le principal risque** : il faut commencer le générateur tôt (phase 2).
5. **Si le temps manque**, on coupe d'abord : LayoutXLM, CLIP, Donut, le VLM via API, puis la partie 2.
   Les 9 itérations restent prioritaires.

**Prochaine étape proposée :** phase 0 (Git + CI + `docs/iterations.md`), puis phase 1 (DiT réel dans Docker).
