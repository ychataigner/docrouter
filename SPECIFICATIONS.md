# Spécifications — DocRouter : API de routage et de pré-comptabilisation de documents

> Thème C (L'Administratif) — Projet d'industrialisation d'une API de vision par ordinateur
> Version 0.1 — 30/10/2026

---

## 1. Contexte et objectifs

Dans une entreprise, les documents administratifs (factures fournisseurs, notes de frais, bons de livraison…) arrivent en vrac : scans, photos de smartphone, PDF. Leur tri et leur saisie en comptabilité sont manuels, lents et sources d'erreurs.

**DocRouter** automatise ce flux en deux parties :

| Partie | Objectif | Entrée | Sortie |
|---|---|---|---|
| **1 — Routage** | Classer automatiquement chaque document déposé et le ranger dans le bon dossier | Image / PDF | Type de document + score de confiance → document rangé dans `factures/`, `notes_de_frais/`, `bons_de_livraison/`, `autres/` |
| **2 — Pré-comptabilisation** | Lire (OCR) les documents classés, en extraire les informations clés et générer un **journal comptable** | Documents classés | Fichier d'écritures comptables (CSV + Excel) |

Le projet n'utilise **aucun document réel de l'entreprise** : uniquement des documents publics ou fictifs (voir §4).

---

## 2. Périmètre

### Dans le périmètre
- Classification de 4 classes de documents (§3).
- Formats : JPG, PNG, PDF (scanné ou natif), 1re page utilisée pour la classification.
- Documents **imprimés/dactylographiés** en français et en anglais.
- OCR + extraction de champs (fournisseur, date, n° de document, montants HT/TVA/TTC…).
- Génération d'un journal comptable simplifié (inspiré du FEC).
- Interface Streamlit de dépôt et de visualisation.
- Toute la chaîne d'industrialisation demandée (Docker, ONNX, charge, auth, cloud, MLflow, FinOps).

### Hors périmètre
- Documents **manuscrits** (OCR manuscrit trop peu fiable — éventuellement en bonus).
- Connexion à un vrai logiciel comptable (Sage, Cegid…) : on produit un fichier d'export.
- Validation comptable réelle (lettrage, rapprochement bancaire).
- Classification multi-documents dans un même PDF (un fichier = un document).

---

## 3. Types de documents (classes)

| Classe | Label API | Exemples | Dossier de destination |
|---|---|---|---|
| Facture | `facture` | Facture fournisseur, facture d'achat | `factures/` |
| Note de frais | `note_de_frais` | Ticket de caisse, reçu restaurant, ticket de péage/carburant, reçu de taxi | `notes_de_frais/` |
| Bon de livraison | `bon_de_livraison` | BL, bordereau de livraison | `bons_de_livraison/` |
| Autre | `autre` | Lettre, formulaire, CV, email imprimé… | `autres/` |

**Règle de confiance** : si le score de la classe prédite est `< 0,60` (seuil configurable), le document est rangé dans `a_verifier/` et signalé dans l'interface pour validation manuelle.

---

## 4. Données

Aucun dataset public ne contient exactement ces 4 classes : le dataset d'entraînement sera **composé** à partir de plusieurs sources.

| Classe | Source(s) envisagée(s) | Remarque |
|---|---|---|
| `facture` | **RVL-CDIP** (classe *invoice*), datasets Kaggle d'images de factures, factures fictives générées | RVL-CDIP = 400k documents scannés, 16 classes, dispo sur Hugging Face |
| `note_de_frais` | **SROIE** (tickets de caisse annotés), **CORD** (reçus, dispo sur Hugging Face) | SROIE fournit aussi les champs annotés (société, date, total) → utile pour évaluer la partie 2 |
| `bon_de_livraison` | **Générés** (script Python : Faker + ReportLab/Jinja→PDF, puis dégradation « effet scan » : rotation, bruit, flou) | Aucune source publique fiable → génération synthétique |
| `autre` | RVL-CDIP (classes *letter*, *form*, *email*, *resume*, *memo*…) | Échantillonnage équilibré |

- Volume cible : **~500 à 1 000 images par classe**, split 70 / 15 / 15 (train / val / test).
- Les documents générés ont une **vérité terrain connue** (montants, dates…) → mesure automatique de la qualité d'extraction.

### 4.1 Jeu de test français

Les datasets publics étant anglophones, un jeu de test **100 % français** (~30 documents par classe, ~120 au total) est constitué à part. C'est sur lui que se font l'évaluation finale et la démo.

| Classe | Source |
|---|---|
| `facture` | Factures générées (Faker `fr_FR`) avec des **gabarits différents de ceux de l'entraînement** ; exemples de factures Factur-X publiés par le FNFE-MPE (à vérifier) ; modèles gratuits Word/Excel/LibreOffice remplis avec des données fictives |
| `note_de_frais` | **Photos de vrais tickets personnels** (restaurant, carburant, péage, parking), numéros de carte masqués |
| `bon_de_livraison` | BL générés avec des gabarits non vus à l'entraînement ; modèles gratuits de BL remplis |
| `autre` | Courriers, attestations, formulaires administratifs publics (ex. Cerfa vierges ou fictifs) |

- Une partie des documents générés est **imprimée puis photographiée/scannée** au smartphone pour reproduire les conditions réelles.
- Vérité terrain dans `data/test_fr/labels.csv` : `fichier;classe;fournisseur;date;numero;montant_ht;tva;montant_ttc`.
- **Aucun gabarit du jeu de test n'est utilisé à l'entraînement**, sinon le modèle reconnaîtrait la mise en page plutôt que le type de document.

---

## 5. Architecture

```
┌────────────────────┐   HTTP (multipart + token)   ┌──────────────────────────────────┐
│  Streamlit (UI)    │ ───────────────────────────► │  API FastAPI (conteneur Docker)  │
│  - dépôt fichiers  │                              │                                  │
│  - visualisation   │ ◄─────────────────────────── │  1. Prétraitement (PDF→image,    │
│  - rangement dans  │        JSON (type, champs,   │     redimensionnement, niveaux   │
│    les dossiers    │        écritures)            │     de gris)                     │
│  - export journal  │                              │  2. Classifieur (ONNX)           │
└────────┬───────────┘                              │  3. OCR                          │
         │                                          │  4. Extraction de champs         │
         ▼                                          │  5. Génération des écritures     │
  Dossiers locaux                                   │  → aucun fichier conservé (RGPD) │
  factures/ notes_de_frais/ …                       └───────────────┬──────────────────┘
  journal_comptable.csv / .xlsx                                     │ métriques
                                                                    ▼
                                                             MLflow (tracking)
```

**Choix structurant (RGPD — itération 6)** : l'API est **sans état** : elle traite le fichier en mémoire et ne le conserve jamais. C'est le **client** (Streamlit ou script) qui range physiquement les fichiers dans les dossiers à partir de la réponse de l'API, et qui agrège les écritures dans le journal.

---

## 6. Partie 1 — Routage des documents

### 6.1 Interface Streamlit
- Zone de **glisser-déposer multi-fichiers** (JPG, PNG, PDF ; 10 Mo max par fichier).
- Pour chaque document : miniature, type prédit, score de confiance, dossier de destination.
- Possibilité de **corriger** manuellement le type avant rangement (la correction est tracée dans MLflow pour mesurer les erreurs du modèle).
- Bouton « Ranger » → copie des fichiers dans `output/<classe>/` avec un nom normalisé : `AAAAMMJJ_<classe>_<id>.<ext>`.
- Tableau récapitulatif : nombre de documents par classe, nombre « à vérifier ».
- Thème sombre, menu latéral en deux sections : **Partie 1 · Routage** (Dépôt, Mes documents) et **Partie 2 · Comptabilité** (Journal comptable).
- Espace **« Mes documents »** : tous les documents rangés avec leur type prédit, la confiance et le mode de tri (automatique, corrigé, à vérifier) ; filtres, aperçu, téléchargement et reclassement. Ces informations sont tenues par le client dans `output/registre.jsonl` (l'API reste sans état).

### 6.2 Modèle de classification
- **Modèle de base** : `microsoft/dit-base-finetuned-rvlcdip` (Document Image Transformer, Hugging Face), déjà entraîné sur des documents scannés.
- **Fine-tuning** sur nos 4 classes (remplacement de la tête de classification).
- Alternative légère pour comparaison : `google/vit-base-patch16-224` ou un `EfficientNet` fine-tuné.
- Optimisation (itération 4) : export **ONNX** + quantization dynamique **INT8**, inférence via `onnxruntime`.
- **Pourquoi pas un modèle « français » ?** DiT/ViT classent sur l'apparence visuelle (mise en page, tableaux, format ticket), pas sur le texte : la langue a peu d'impact. Le vrai écart est le domaine (RVL-CDIP = vieux scans anglais N&B), corrigé en ajoutant des documents français au fine-tuning. Le français compte en revanche dans la partie 2 (OCR, mots-clés).
- **Plan B** : `microsoft/layoutxlm-base` (image + texte OCR, multilingue dont français) si DiT confond trop factures et bons de livraison — au prix d'une latence plus élevée (OCR avant classification).

### 6.3 Critères d'acceptation
- Accuracy ≥ **90 %** et F1-macro ≥ **0,88** sur le jeu de test.
- Écart d'accuracy entre modèle PyTorch et modèle ONNX INT8 ≤ **2 points**.
- Temps de classification ≤ **500 ms** par image sur CPU (modèle optimisé).

---

## 7. Partie 2 — Extraction et journal comptable

### 7.1 OCR
- Moteur principal : **docTR** (open source, bon sur documents imprimés) ou **Tesseract** (`fra+eng`) en solution de repli plus légère.
- Prétraitement : redressement (deskew), binarisation, augmentation du contraste.
- Bonus : **Donut** (`naver-clova-ix/donut-base-finetuned-cord-v2`) pour l'extraction directe des tickets de caisse sans OCR séparé, à comparer avec OCR + règles.

### 7.2 Champs extraits

| Champ | Facture | Note de frais | Bon de livraison | Méthode |
|---|:-:|:-:|:-:|---|
| Fournisseur / émetteur | ✅ | ✅ | ✅ | 1res lignes du document + SIRET/TVA intracom |
| Date du document | ✅ | ✅ | ✅ | Regex (formats JJ/MM/AAAA, JJ-MM-AA, « 12 mars 2026 »…) |
| N° de document | ✅ | — | ✅ | Regex sur mots-clés (`Facture n°`, `BL n°`, `Réf.`) |
| N° de commande | ✅ | — | ✅ | Regex (`Commande n°`, `PO`) |
| Montant HT | ✅ | ⚪ | — | Mots-clés `Total HT`, `Sous-total` |
| TVA (montant + taux) | ✅ | ⚪ | — | Mots-clés `TVA`, taux 5,5 / 10 / 20 % |
| Montant TTC | ✅ | ✅ | — | Mots-clés `Total TTC`, `Net à payer`, `TOTAL` |
| Catégorie de dépense | — | ✅ | — | Mots-clés (restaurant, carburant, péage, taxi, hôtel…) |

✅ obligatoire — ⚪ si présent — — non applicable

**Contrôles de cohérence** : `HT + TVA = TTC` (tolérance 0,02 €), date valide et non future. En cas d'échec ou de champ obligatoire manquant → statut `a_verifier`.

### 7.3 Génération des écritures comptables
Une écriture équilibrée (Σ débit = Σ crédit) par document :

| Type | Journal | Débit | Crédit |
|---|---|---|---|
| Facture fournisseur | `AC` (achats) | `607` Achats de marchandises (HT) + `44566` TVA déductible | `401<FOURNISSEUR>` Fournisseur (TTC) |
| Note de frais | `NDF` | `625x` selon catégorie (`6251` déplacements, `6256` missions, `6257` réceptions/restaurant) HT + `44566` TVA | `421` Personnel — rémunérations dues (TTC) |
| Bon de livraison | — | *Pas d'écriture* (pas de valeur comptable) | — |

Les bons de livraison servent au **rapprochement** : si une facture et un BL ont le même n° de commande, la ligne du journal porte la référence du BL (colonne `piece_liee`). Les comptes sont paramétrables dans un fichier `config/plan_comptable.yaml`.

### 7.4 Format du journal (CSV `;` UTF-8 + XLSX)
Colonnes inspirées du FEC (Fichier des Écritures Comptables) :

```
JournalCode;JournalLib;EcritureNum;EcritureDate;CompteNum;CompteLib;PieceRef;PieceDate;EcritureLib;Debit;Credit;Statut;Confiance;FichierSource;PieceLiee
AC;Achats;AC0001;20261015;607000;Achats marchandises;FA-2026-118;20261012;Fournisseur ACME - FA-2026-118;100,00;0,00;OK;0,94;20261015_facture_001.pdf;BL-5521
AC;Achats;AC0001;20261015;445660;TVA déductible;FA-2026-118;20261012;Fournisseur ACME - FA-2026-118;20,00;0,00;OK;0,94;20261015_facture_001.pdf;BL-5521
AC;Achats;AC0001;20261015;401ACME;Fournisseur ACME;FA-2026-118;20261012;Fournisseur ACME - FA-2026-118;0,00;120,00;OK;0,94;20261015_facture_001.pdf;BL-5521
```

### 7.5 Critères d'acceptation
- Précision d'extraction du **montant TTC ≥ 85 %** et de la **date ≥ 85 %** (jeu de test à vérité terrain : SROIE + factures générées).
- 100 % des écritures générées sont équilibrées.
- Toute écriture incertaine est marquée `a_verifier` (aucune écriture « silencieusement fausse » sur des champs manquants).

---

## 8. Contrat d'API (FastAPI)

| Méthode | Route | Description | Itération |
|---|---|---|---|
| `GET` | `/health` | État du service et version du modèle chargé | 1 |
| `POST` | `/v1/classify` | Fichier → type + scores | 1–2 |
| `POST` | `/v1/extract` | Fichier (+ type optionnel) → texte OCR + champs extraits | Partie 2 |
| `POST` | `/v1/process` | Fichier → classification + extraction + écritures (tout-en-un) | Partie 2 |
| `POST` | `/v1/journal/export` | Liste de résultats JSON → fichier CSV ou XLSX | Partie 2 |
| `POST` | `/auth/token` | Obtention d'un token JWT | 6 |

Toutes les routes `/v1/*` exigent `Authorization: Bearer <token>` à partir de l'itération 6. Documentation Swagger sur `/docs`.

**Exemple — `POST /v1/process`**
```json
{
  "request_id": "a1f3c9e2",
  "filename": "scan_0042.pdf",
  "classification": {
    "type": "facture",
    "confidence": 0.94,
    "scores": {"facture": 0.94, "note_de_frais": 0.03, "bon_de_livraison": 0.02, "autre": 0.01},
    "destination": "factures"
  },
  "extraction": {
    "fournisseur": "ACME SAS",
    "date": "2026-10-12",
    "numero_document": "FA-2026-118",
    "numero_commande": "CMD-7781",
    "montant_ht": 100.00,
    "tva": {"taux": 20.0, "montant": 20.00},
    "montant_ttc": 120.00,
    "coherence_ok": true
  },
  "ecritures": [
    {"journal": "AC", "compte": "607000", "libelle": "ACME SAS - FA-2026-118", "debit": 100.00, "credit": 0.00},
    {"journal": "AC", "compte": "445660", "libelle": "ACME SAS - FA-2026-118", "debit": 20.00, "credit": 0.00},
    {"journal": "AC", "compte": "401ACME", "libelle": "ACME SAS - FA-2026-118", "debit": 0.00, "credit": 120.00}
  ],
  "statut": "OK",
  "timings_ms": {"preprocess": 40, "classify": 180, "ocr": 950, "extract": 12}
}
```

**Codes d'erreur** : `400` format non supporté / fichier corrompu, `401` token absent ou invalide, `413` fichier trop volumineux, `422` paramètres invalides, `500` erreur interne.

---

## 9. Exigences non fonctionnelles

| Domaine | Exigence |
|---|---|
| **Performance** | p95 `/v1/classify` < 1 s, `/v1/process` < 3 s par page sur CPU ; ≥ 5 req/s sur classify avec <br/>1 conteneur |
| **Sécurité** | Auth JWT (ou clé d'API), secrets via variables d'environnement (`.env` non versionné), token Hugging Face jamais dans le code |
| **RGPD** | Aucun fichier écrit sur disque côté serveur (traitement en mémoire, `tempfile` supprimé immédiatement si nécessaire), logs sans contenu OCR ni montants nominatifs, seules des métriques techniques partent dans MLflow |
| **Portabilité** | Image Docker multi-stage, < 2 Go, démarrage < 30 s |
| **Observabilité** | Logs structurés JSON, `request_id` par requête, métriques dans MLflow |
| **Qualité** | Tests `pytest` (API, extraction regex, équilibre des écritures), lint `ruff` |

---

## 10. Correspondance avec les itérations du module

| It. | Livrable dans DocRouter |
|---|---|
| 1 — Squelette | FastAPI + `/health` + `/v1/classify` qui accepte un fichier (réponse factice) + Swagger |
| 2 — Cerveau | DiT (Hugging Face) branché sur `/v1/classify` avec une vraie prédiction ; puis DiT fine-tuné sur nos 4 classes (non exigé par le sujet, mais nécessaire : DiT seul ne connaît ni les notes de frais ni les BL), publié sur le Hub HF |
| 3 — Isolement | `Dockerfile` multi-stage + `docker-compose.yml` (API + Streamlit + MLflow) |
| 4 — Optimisation | Export ONNX + quantization INT8 du classifieur, comparaison taille / latence / accuracy ; l'ONNX devient le modèle par défaut (image sans PyTorch) |
| 5 — Flux | Script `load_test/` (Locust) envoyant des documents en continu ; « dossier de réception » surveillé ; boucle de rétroaction (corrections → données étiquetées pour le ré-entraînement) |
| 6 — Barrière | JWT ou clés d'API révocables, traitement 100 % en mémoire, test automatisé vérifiant qu'aucun fichier ne reste après inférence, HTTPS et secrets hors du code |
| 7 — Grand saut | Déploiement (ex. Azure Container Apps, Google Cloud Run ou AWS App Runner) ; justification CPU (inférence) vs GPU (entraînement) |
| 8 — Rétroviseur | L'API logge **chaque requête** dans MLflow (latence par étape, modèle, confiance, `a_verifier`, sans contenu) ; même test de charge sur modèle lourd vs ONNX ; corrections manuelles ; dashboard = livrable |
| 9 — Facture | Estimation pour 1 M de documents/mois (CPU ONNX vs modèle lourd vs GPU vs API HF ; avec ou sans OCR), README final, un tag Git par itération |

La **partie 2** (OCR, extraction, journal) dépasse le thème C du sujet : c'est une extension, réalisée après le parcours des 9 itérations.

---

## 11. Organisation du dépôt

Dépendances gérées avec **uv** en *workspace* : un `pyproject.toml` par paquet, un `uv.lock` unique (versions figées). Chaque image Docker n'installe que son paquet (le front n'embarque pas PyTorch).

```
VisionParOrdinateur/
├── pyproject.toml           # racine du workspace uv + config ruff / pytest
├── uv.lock
├── packages/core/           # docrouter-core : types de documents, schémas Pydantic (partagé)
├── services/
│   ├── api/                 # docrouter-api : FastAPI + Dockerfile
│   │   └── src/docrouter_api/
│   │       ├── main.py          # routes
│   │       ├── settings.py      # config par variables DOCROUTER_*
│   │       ├── preprocessing.py # PDF→image, normalisation
│   │       ├── classifier.py    # interface + DiT (puis fine-tuné / ONNX)
│   │       ├── auth.py          # (it. 6)
│   │       ├── ocr.py / extraction.py / accounting.py   # (partie 2)
│   └── ui/                  # docrouter-ui : Streamlit + Dockerfile
├── training/                # dataset, fine-tuning, export ONNX (hors images Docker)
├── load_test/               # (it. 5)
├── config/plan_comptable.yaml   # (partie 2)
├── tests/
├── docker-compose.yml
├── .env.example
├── BUDGET.md                # (it. 9)
└── README.md
```

---

## 12. Stack technique

Python 3.11 · uv (workspace) · ruff · FastAPI · Uvicorn · Pydantic · Hugging Face `transformers` / `datasets` · PyTorch · ONNX Runtime / `optimum` · docTR ou Tesseract · pypdfium2 · Pillow / OpenCV · pandas / openpyxl · Streamlit · MLflow · Docker · pytest · Locust

---

## 13. Risques

| Risque | Impact | Parade |
|---|---|---|
| Pas de dataset de bons de livraison | Classe mal apprise | Génération synthétique + augmentations « effet scan » |
| Biais de domaine (RVL-CDIP = vieux documents anglais N&B) | Mauvaise généralisation sur docs français récents | Mélanger avec des documents générés en français, augmentations couleur/photo |
| OCR faible sur photos de tickets froissés | Montants mal lus | Prétraitement, contrôles de cohérence, statut `a_verifier` |
| Image Docker lourde (PyTorch + OCR) | Coût cloud, démarrage lent | Image d'inférence avec ONNX Runtime uniquement (sans PyTorch) |
| OCR = étape la plus lente | Latence de `/v1/process` | Séparer `/classify` (rapide) et `/process` ; traitement asynchrone en option |

---

## 14. Points ouverts

- Modèle de classification final : DiT vs ViT (vs LayoutXLM si besoin) — à trancher sur le jeu de test français (§4.1).
- OCR : docTR vs Tesseract — à trancher selon qualité / poids de l'image Docker.
- Fournisseur cloud de l'itération 7.
- Bonus possibles : support du manuscrit, découpage des PDF multi-documents, traitement asynchrone par file (Redis / Celery).
