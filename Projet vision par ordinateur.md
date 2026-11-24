# Projet d'industrialisation d'une API de vision par ordinateur

## Contexte du projet

Bienvenue ! Tout au long de ce module sur l'Industrialisation de l'IA dans le Cloud, vous allez construire, déployer et optimiser un produit d'IA de A à Z.

Votre mission est de transformer un modèle de Computer Vision (qui tourne actuellement de manière isolée) en un véritable microservice web, robuste, optimisé et prêt à absorber un trafic important dans le cloud.

## Prérequis : Hugging Face

Nous utiliserons la plateforme Hugging Face pour l'hébergement et la récupération des modèles pré-entraînés.

* Créez un compte gratuit sur Hugging Face.
* Dans les paramètres de votre compte, générez un "Access Token" (clé d'API) qui vous permettra de télécharger les modèles directement via votre code Python.

## Choix de la mission

Avant de commencer à coder, votre équipe doit choisir un cas d'usage parmi les 4 scénarios ci-dessous. Le code de l'API et l'infrastructure Cloud seront les mêmes, seul le modèle utilisé et la logique métier changeront.

* **Thème A (L'Usine) :** API de Contrôle Qualité. L'API reçoit des photos de pièces sur une ligne de montage et doit détecter les défauts.
* **Thème B (La Marketplace) :** API de Catégorisation. L'API reçoit des photos d'objets réalisées par des utilisateurs et doit identifier la catégorie de l'objet pour pré-remplir une annonce e-commerce.
* **Thème C (L'Administratif) :** API de Routage de Documents. L'API reçoit des scans (notes de frais, factures) et doit les classifier.
* **Thème D (L'Inclusion Web) :** API d'Accessibilité. L'API reçoit une image et génère automatiquement une description textuelle (Image Captioning).
* **Thème E (A vous décrire une application utilisant le traitement d'image) :** Validez impérativement votre idée avec le formateur avant de commencer.

## Méthodologie : Apprentissage Itératif

Ce projet est découpé en 9 itérations. Pour chaque étape :

* **Mission :** Vous vous lancez dans la pratique avec vos connaissances actuelles. L'objectif est de chercher, tester et faire des erreurs.

* **Application :** Vous finalisez l'itération en appliquant les concepts d'industrialisation liés à l'étape en cours.

---

## Les itérations

### Itération 1 : Le Squelette (Prise en main)

L'objectif est d'avoir une base fonctionnelle pour communiquer avec votre future IA.

* **Mission :** Créez une API web basique qui expose un point d'entrée pour recevoir une image via une requête HTTP POST.

* **Concepts associés :** Comprendre et appliquer les concepts d'API pour intégrer les modèles de machine learning avec diverses applications cloud.

* **Livrable :** Un code Python fonctionnel et l'accès à la documentation Swagger locale.

### Itération 2 : Le Cerveau (Intégration du modèle)

Il est temps d'intégrer le modèle de vision.

* **Mission :** Connectez votre API à un modèle de vision issu de Hugging Face. L'API doit prendre l'image, la prétraiter, l'envoyer au modèle et renvoyer la prédiction sous format JSON.

* **Concepts associés :** Élaborer des stratégies de serving de modèles d'IA, en utilisant des solutions avancées pour garantir la disponibilité et la scalabilité des modèles en production.

* **Livrable :** Une API capable de renvoyer une prédiction réelle basée sur une image uploadée.

### Itération 3 : L'Isolement (Conteneurisation)

Votre code fonctionne sur votre machine, mais fonctionnera-t-il sur celle de votre collègue ou sur le Cloud ?

* **Mission :** Packager l'API de l'itération précédente à l'aide de Docker.

* **Concepts associés :** Comprendre l'utilisation et la manipulation de Docker pour containeriser les applications d'IA, assurant ainsi une portabilité et un déploiement cohérent sur différentes plateformes.

* **Livrable :** Un Dockerfile optimisé et fonctionnel.

### Itération 4 : L'Optimisation (Faire mieux avec moins)

L'hébergement Cloud coûte cher, il faut optimiser les ressources.

* **Mission :** Appliquez des techniques d'optimisation sur votre modèle pour réduire son poids et accélérer le temps d'inférence (par exemple via ONNX).

* **Concepts associés :** Optimiser les modèles d'apprentissage automatique pour la production via des techniques comme la quantization, assurant une performance optimale sur le cloud.

* **Livrable :** Le code mis à jour utilisant la version optimisée du modèle.

### Itération 5 : Le Flux (Pipelines de données)

Un système en production reçoit des flux continus.

* **Mission :** Mettez en place un script externe simulant un afflux d'images en continu vers votre API pour tester sa résistance à la charge.
* **Concepts associés :** Assurer la collecte et l'actualisation automatisées des données via des pipelines DATA temps réel, garantissant la fraîcheur et la fiabilité des données utilisées par les modèles d'IA.

### Itération 6 : La Barrière (Sécurité & RGPD)

Vous gérez des données potentiellement sensibles (visages, documents).

* **Mission :** Ajoutez un système d'authentification à votre API. Mettez en place une règle pour vous assurer qu'aucune image n'est conservée sur le serveur après inférence.

* **Concepts associés :** Concevoir une architecture de données robuste et sécurisée en tenant compte des aspects multiples comme le chiffrement, la gestion des accès et le respect des normes légales (ex: RGPD).

* **Livrable :** Une API sécurisée exigeant un token d'authentification pour fonctionner.

### Itération 7 : Le Grand Saut (Déploiement Cloud - si possible)

Il est temps de rendre votre API accessible à tous.

* **Mission :** Déployez votre conteneur Docker sur un fournisseur Cloud.

* **Concepts associés :** Maîtriser le déploiement de modèles d'IA dans le cloud, en exploitant les infrastructures modernes et en choisissant les ressources adaptées, y compris la sélection de machines GPU pour répondre aux besoins spécifiques d'apprentissage.

### Itération 8 : Le Rétroviseur (MLOps avec MLflow)

Comment savoir si l'optimisation a dégradé les prédictions ou si le modèle est performant ?

* **Mission :** Intégrez MLflow à votre projet. Vous devez "tracker" le temps de réponse et comparer les métriques du modèle lourd (Itération 2) avec celles du modèle optimisé (Itération 4).

* **Concepts associés :** Mettre en œuvre les méthodes de MLOps pour automatiser le cycle de vie des projets d'IA, en utilisant des bibliothèques telles que MLflow pour assurer le tracking et la gestion des modèles.

* **Livrable :** Un dashboard MLflow affichant l'historique des requêtes et métriques.

### Itération 9 : La Facture (FinOps & Documentation finale)

Un projet n'est jamais terminé sans documentation et sans budget validé.

* **Mission :** Calculez une estimation budgétaire de votre solution Cloud pour un traitement de masse (ex: 1 million d'images par mois). Finalisez la documentation technique.

* **Concepts associés :** Planifier les ressources et estimer le budget nécessaire pour projeter efficacement des solutions d'IA dans le cloud, avec un contrôle rigoureux des délais et des coûts. Rédiger une documentation détaillée et accessible pour garantir une transmission claire des procédures d'implémentation et de maintenance des solutions IA dans le cloud.

* **Livrable final :** Un repository Git complet (Code, Dockerfile, README.md, fichier d'estimation budgétaire).
