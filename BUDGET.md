# DocRouter — Estimation budgétaire (itération 9, FinOps)

> **En bref.** Pour **1 million de documents par mois**, l'API coûte **≈ 6 $/mois** sur Cloud Run
> (serverless, facturation à la requête), soit **0,006 $ pour 1 000 documents**. Avec le front et
> MLflow sur une petite VM, le total est d'**≈ 34 $/mois**, soit 0,03 $ pour 1 000 documents.
> C'est l'optimisation ONNX de l'itération 4 qui rend ce scénario possible : un démarrage en 4 s
> au lieu de 24 s permet d'éteindre l'API quand personne ne l'utilise.

Montants en dollars US, hors taxes. Tarifs publics relevés le **25/11/2026**, à revérifier avant
tout engagement. Le calcul est **reproductible** :

```bash
uv run docrouter-budget                      # 1 million de documents / mois
uv run docrouter-budget --docs 10000000 --markdown
```

Les mesures et les tarifs sont des constantes commentées dans
[`training/src/docrouter_training/budget.py`](training/src/docrouter_training/budget.py) : si un
prix change, on modifie la constante et on relance. Les calculs sont couverts par des tests
(`tests/test_budget.py`).

## 1. Hypothèses de trafic

| Hypothèse | Valeur | Pourquoi |
|---|---|---|
| Volume | 1 000 000 documents / mois | Énoncé du sujet |
| Répartition | 22 jours ouvrés × 10 h | Documents d'entreprise, envoyés en heures de bureau |
| Débit moyen en heures de bureau | **1,26 doc/s** | 1 000 000 / (22 × 10 × 3 600) |
| Pic | **3,8 doc/s** (× 3) | Arrivées groupées : début de matinée, fin de mois |
| Taille d'un document | 0,2 Mo en moyenne | Jeu de test (JPEG, PDF, photos) |

## 2. Ce qu'on a mesuré (pas supposé)

| Mesure | Valeur | Source |
|---|---|---|
| Débit soutenu, ONNX INT8, 2 vCPU | **11,5 doc/s**, 0 erreur | Itération 5 (Locust) |
| Débit soutenu, ONNX INT8, 4 vCPU | **20,5 doc/s**, 0 erreur | Itération 5 |
| Requête isolée (prétraitement + modèle) | 106 ms (ONNX), ≈ 117 ms (PyTorch) | Itérations 4 et 5 |
| Démarrage jusqu'à « prêt » | **4 s** (ONNX) / 24 s (PyTorch) | Itération 4 |
| Image Docker | **538 Mo** (ONNX) / 1,87 Go (PyTorch) | Itération 4 |
| RAM | 231 Mo au repos, ≤ 850 Mo sous charge | Itérations 4 et 5 |

Un seul conteneur de 2 vCPU traite 11,5 doc/s : le pic de 3,8 doc/s n'en occupe qu'un tiers.
En théorie, il pourrait traiter ≈ 30 millions de documents par mois.

## 3. Tarifs utilisés

| Service | Tarif | Source |
|---|---|---|
| Cloud Run, à la requête | 0,000024 $/vCPU·s ; 0,0000025 $/Gio·s ; 0,40 $/million de requêtes ; gratuits chaque mois : 180 000 vCPU·s, 360 000 Gio·s, 2 M requêtes. Temps facturé arrondi aux **100 ms supérieures** | [Cloud Run pricing](https://cloud.google.com/run/pricing) |
| Cloud Run, instance toujours allouée | 0,000018 $/vCPU·s ; 0,000002 $/Gio·s | idem |
| VM e2-standard-2 (2 vCPU, 8 Go), Paris | ≈ 0,078 $/h (≈ 57 $/mois) | [gcloud-compute.com](https://gcloud-compute.com/e2-standard-2.html) |
| VM e2-medium (front + MLflow) | ≈ 0,038 $/h (≈ 28 $/mois) | estimation, même source |
| Hugging Face Inference Endpoints | 0,03 $ par cœur CPU et par heure ; GPU T4 : 0,50 $/h | [HF Endpoints pricing](https://huggingface.co/docs/inference-endpoints/pricing) |
| Registre d'images / sortie réseau | 0,10 $/Go-mois / 0,12 $/Go | tarifs Google Cloud usuels |

## 4. Scénarios pour 1 million de documents par mois

| Scénario | Coût / mois | Coût / 1 000 docs | Détail | Remarque |
|---|---|---|---|---|
| **A. Cloud Run à la requête — ONNX INT8 (recommandé)** | **5,51 $** | **0,0055 $** | 2 vCPU / 2 Gio, 0 instance au repos ; 0,2 s facturées par document | démarrage à froid ≈ 4 s après une période sans trafic |
| B. Cloud Run, instances toujours prêtes — ONNX INT8 | 105,25 $ | 0,1052 $ | min-instances = 1, 2 vCPU / 2 Gio | pas de démarrage à froid ; autoscaling au-delà |
| C. VM e2-standard-2 × 2 (haute dispo) — ONNX INT8 | 114,01 $ | 0,1140 $ | `docker compose` tel quel (API + front + MLflow) | 23 doc/s au total, dimensionnement fixe |
| D. Cloud Run à la requête — modèle lourd PyTorch | 5,64 $ | 0,0056 $ | même plateforme que A, image 1,87 Go | démarrage à froid ≈ 24 s : en pratique, il faut B (≈ 105 $) |
| E. Hugging Face Inference Endpoint CPU (géré) | 43,80 $ | 0,0438 $ | 1 endpoint 2 vCPU toujours allumé, PyTorch | rien à administrer, mais documents envoyés chez un tiers (RGPD) |
| F. GPU NVIDIA T4 dédié | 365,00 $ | 0,3650 $ | 1 GPU allumé en permanence | inutile : le modèle répond en 54 ms sur CPU |
| + Front Streamlit et MLflow (VM e2-medium) | 27,74 $ | 0,0277 $ | à ajouter à A, B, D, E ou F | déjà inclus dans C |

Détail du scénario A, calculable à la main :
- 1 000 000 documents × 0,2 s facturées = 200 000 s d'instance.
- CPU : 200 000 s × 2 vCPU = 400 000 vCPU·s ; moins 180 000 gratuits = 220 000 × 0,000024 $ = **5,28 $**.
- Mémoire : 400 000 Gio·s ; moins 360 000 gratuits = 40 000 × 0,0000025 $ = **0,10 $**.
- Requêtes : 1 M, dans la part gratuite (2 M) = **0 $**.
- Registre d'images (0,54 Go) et sortie réseau (réponses JSON de 0,6 Ko, soit 0,6 Go) = **0,13 $**.

C'est une **borne haute** : chaque document est facturé seul, alors que des requêtes
simultanées se partagent le même temps d'instance.

## 5. Sensibilité au volume

| Documents / mois | Pic (doc/s) | A. Cloud Run à la requête | B. Cloud Run toujours prêt | C. VM × N (HA) | E. HF Endpoint CPU |
|---|---|---|---|---|---|
| 100 000 | 0,4 | 0 $ (part gratuite) | 105 $ | 114 $ | 44 $ |
| **1 000 000** | 3,8 | **6 $** | 105 $ | 114 $ | 44 $ |
| 10 000 000 | 37,9 | 105 $ | 421 $ | 229 $ | 219 $ |
| 100 000 000 | 378,8 | 1 101 $ | 3 476 $ | 1 886 $ | 2 015 $ |

Lecture : le serverless (A) reste le moins cher à tous les volumes étudiés. Au-delà de
≈ 10 millions de documents par mois, des VM réservées (engagement 1 à 3 ans, −25 à −55 %) ou un
cluster deviennent compétitifs : le scénario C ne compte que le tarif à la demande. Le GPU ne
figure pas dans ce tableau : son débit n'a pas été mesuré.

## 6. Ce que l'optimisation a rapporté

- **ONNX INT8 et serverless vont ensemble.** Par document, le modèle lourd coûterait presque
  autant que l'ONNX (scénario D) : les deux tiennent dans le même arrondi de 100 ms. La
  différence est ailleurs : **24 s de démarrage à froid** contre 4 s. Avec PyTorch, il faut
  garder une instance allumée en permanence (≈ 105 $/mois) pour que les utilisateurs
  n'attendent pas ; avec ONNX, l'API peut s'éteindre la nuit et le week-end (≈ 6 $/mois).
  L'optimisation divise donc la facture par **≈ 19**.
- **Pas de GPU.** Sur CPU, l'inférence prend 54 ms. Un GPU T4 coûterait ≈ 365 $/mois pour
  quelques millisecondes gagnées. Le GPU ne serait utile que pour entraîner, ce que la sonde
  linéaire évite : entraînement en quelques minutes sur CPU, donc coût négligeable.
- **Image 3,5 × plus légère** : moins de stockage, et des démarrages et mises à l'échelle plus
  rapides.

## 7. Recommandation

| Poste | Choix | Coût / mois |
|---|---|---|
| API de classification | Cloud Run, à la requête, ONNX INT8, 2 vCPU / 2 Gio, `min-instances=0`, `max-instances=3`, `DOCROUTER_ONNX_THREADS=2` | ≈ 6 $ |
| Front + MLflow | VM e2-medium (ou Cloud Run pour le front) | ≈ 28 $ |
| Modèle | Hub Hugging Face (dépôt privé) : stockage inclus | 0 $ |
| **Total** | | **≈ 34 $ / mois** (≈ 0,03 $ / 1 000 documents) |

Si le démarrage à froid de 4 s gêne (premier document du matin), on passe à `min-instances=1`
seulement en heures de bureau (≈ 32 $/mois de plus, programmé), plutôt qu'en permanence.

## 8. Maîtrise des coûts

- **Plafond d'instances** (`max-instances=3`, soit ≈ 35 doc/s) : un pic anormal ou une attaque ne
  peut pas faire exploser la facture. Au-delà, les requêtes attendent, comme dans le test de
  charge, sans erreur.
- **Alertes budgétaires** du fournisseur à 50 %, 90 % et 100 % d'un budget mensuel (ex. 50 $).
- **Clés d'API par client** (itération 6) : on sait qui consomme, et on peut couper un client.
- **Suivi** : MLflow enregistre les temps de réponse ; une hausse de latence ou de volume se
  voit avant la facture.
- **Journaux** : métadonnées uniquement (≈ 0,3 Go/mois), dans la part gratuite.

## 9. Limites et points à revoir

- Les tarifs sont des **prix publics à la demande**, relevés à une date donnée, hors taxes et
  hors remises négociées. Les vérifier dans le simulateur du fournisseur avant tout engagement.
- Le facteur de pic (× 3) est une **hypothèse** : à confirmer avec le trafic réel (MLflow).
- Débits mesurés **en local**, sur des documents synthétiques, avec un réseau local ; un vCPU
  cloud peut être un peu plus lent qu'un cœur du PC de test.
- L'itération 7 (déploiement cloud) n'a pas été réalisée (consigne) : ce budget est une
  projection à partir des mesures locales, pas une facture constatée.
- **Partie 2 (perspective)** : ajouter l'OCR et l'extraction des montants multiplierait le
  calcul par document par ≈ 10 à 20 (de l'ordre de 1 à 2 s de CPU par page, non mesuré), soit
  environ 50 à 110 $/mois pour 1 million de documents en scénario A : cela reste modeste.
