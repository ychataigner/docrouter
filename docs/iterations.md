# Journal des itérations

Pour chaque itération du sujet : la mission, ce qu'on a essayé (erreurs comprises), le résultat et
les mesures. Chaque itération terminée porte un tag Git (`git tag`).

---

## Itération 1 — Le Squelette ✅ (tag `it1-it3`)

- **Mission** : API web qui reçoit une image en POST, avec Swagger.
- **Réalisé** : FastAPI, `GET /health`, `POST /v1/classify` (PDF, PNG, JPEG ; 10 Mo max),
  réponses typées avec Pydantic (`packages/core`), Swagger sur `/docs`.
- **Choix** : l'API est **sans état** et ne garde aucun fichier (préparation de l'itération 6).
  C'est le client (Streamlit) qui range les documents.

## Itération 2 — Le Cerveau ⏳

- **Mission** : brancher un modèle Hugging Face et renvoyer une vraie prédiction.
- **État** : DiT (`microsoft/dit-base-finetuned-rvlcdip`) est branché mais optionnel. Le tri
  « simulé » (`DOCROUTER_CLASSIFIER=dummy`) sert à développer le front sans PyTorch.
- **Constat** : DiT pré-entraîné ne connaît que 16 classes RVL-CDIP. Seule `invoice` correspond à
  nos classes, donc les notes de frais et les bons de livraison ne peuvent pas être reconnus.
  Un fine-tuning est nécessaire.

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
