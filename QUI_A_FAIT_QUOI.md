# SeinTinelle : qui a fait quoi

Projet E3 (ESIEE Paris, 2025-2026) : un site web d'aide à la sensibilisation au cancer du sein, avec un module de détection (mammographie, échographie) et un module de prédiction du risque à long terme. Équipe de six étudiants, encadrée par Mme Imen Kachouri.

Ce dépôt privé contient le code de l'application (`app.py`, `frontend/`, `Mirai/`) et les notebooks d'entraînement (`notebooks/`).

## Sources

1. La section « Répartition du travail » du rapport d'équipe « SeinTinelle : Rapport du projet ». Elle se dit établie d'après les comptes rendus d'avancement hebdomadaires et les contributions au dépôt GitHub et au site. C'est une déclaration de l'équipe.
2. La section « Team » du README de l'équipe (noms et filières).

L'historique de commits du code ne permet pas de vérifier qui a écrit quel fichier (un seul commit initial). Les attributions ci-dessous sont donc celles du rapport, et la part exacte de chacun dans le code est À COMPLÉTER.

## Répartition

| Membre | Filière (README) | Rôle d'après le rapport |
|---|---|---|
| Lilou Parrin | Data Science & AI | Site web : pages « Consulter un médecin », « À propos de nous », « FAQ », refonte de l'accueil et de la page de détection. Intégration technique des trois modèles (mammographie, échographie, risque futur) avec le backend Flask et affichage des résultats. Partie éthique du rapport, affiche du projet, participation au logo. |
| Nathan Gautier | Data Science & AI | État de l'art de la prédiction (comparaison ML / deep learning), démarches d'accès à des bases de mammographies (OMI-DB, ADMANI, PLCO). Frontend : page « Restez Informées », première version de « À propos », popup de prévention, refonte visuelle. Vidéo (plan de tournage, coordination), plan de rapport, maquette du poster. |
| Maxime Lenfant | Artificial Intelligence and Cybersecurity | Création du site et implémentation du premier modèle de détection (mammographies). Dépôt GitHub commun, règles de contribution, guide Git. Explication de l'AUC-ROC et veille sur des projets comparables. |
| Ayoub Mernissi | Artificial Intelligence and Cybersecurity | Module de prédiction du risque à venir : état de l'art ciblé, sélection et import d'AsymMirai, extension multimodale (âge, IMC, antécédents familiaux, densité BI-RADS), benchmark RSNA. Code du module d'échographie en collaboration avec Rayan. Rapport (partie prédiction, introduction, couverture), script de la vidéo, logo et supports de communication. |
| Rayan Oulboub (chef de projet) | Artificial Intelligence and Cybersecurity | En binôme avec Ayoub : modèle de risque à 1-5 ans et modèle d'échographie (DenseNet-121 sur BreastMNIST, Focal Loss, TTA, oversampling, Grad-CAM). Démarches d'accès à CSAW. Rapport (introduction, échographie, prédiction, bibliographie, annexes), logo et marketing, vidéo (script, voix off). |
| Antoine Navarro | Informatique | État de l'art deep learning / machine learning pour le diagnostic du cancer du sein, références (apprentissage fédéré, late fusion), notebook Kaggle ADMANI. Participation à la création du site web et à l'implémentation du premier modèle de détection. |

## Où est le code de chacun

| Dossier / fichier | Contenu | Origine |
|---|---|---|
| `app.py`, `frontend/`, `requirements.txt` | Application Flask et site web | Équipe (pas d'attribution par fichier) |
| `Mirai/` | Fork d'AsymMirai (licence MIT, voir `Mirai/LICENSE`) | Donnelly et al., https://github.com/jdonnelly36/AsymMirai |
| `notebooks/asymmirai_multimodal.py` | Modèle d'ensemble imagerie + clinique, interface Gradio | Ayoub Mernissi |
| `notebooks/detection_rsna_v1.py` | Benchmark RSNA v1 | Ayoub Mernissi |
| `notebooks/cnn_cbis_ddsm_et_random_forest_wisconsin.py` | CNN CBIS-DDSM (interrompu) et Random Forest Wisconsin | Ayoub Mernissi |
| `notebooks/echographie_breastmnist_densenet121.py` | Échographie DenseNet-121 sur BreastMNIST | Rayan Oulboub (notebook du Drive), module développé avec Ayoub d'après le rapport |
| `notebooks/detection_1.py` | Détection MammoScreen (Hugging Face) | Dossier de Lilou Parrin |
| `breast_cancer_densenet121_final.keras` | Poids du modèle d'échographie | Équipe |

Les notebooks sont des scripts « # %% » convertis depuis les notebooks Colab, sans sorties ni images.

## Limites

- Les rôles ci-dessus ne sont recoupés par aucune autre trace que le rapport, hors mes propres notebooks (dont je suis propriétaire sur le Drive).
- Le rapport attribue le code du module d'échographie à Ayoub « en collaboration avec Rayan » et le décrit comme développé par Rayan en binôme avec Ayoub. Le partage exact est À COMPLÉTER avec les intéressés.
- Le dépôt est privé et sans licence à la racine : ne pas le rendre public sans l'accord de tous les co-auteurs.
