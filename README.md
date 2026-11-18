# SeinTinelle — détection et prédiction du cancer du sein par IA

**Application web (site web)** réalisée en projet de groupe E3 (ESIEE Paris, 2025-2026), présentée au Jour des Projets. Six étudiants, encadrement : Mme Imen Kachouri.

> **Dépôt privé.** Il contient le code de plusieurs coéquipiers. Aucun poids de modèle n'est inclus (voir « Poids des modèles »). Ne pas le rendre public sans l'accord écrit de tous les co-auteurs. Aucune licence n'est posée à la racine (tous droits réservés par défaut).
>
> **Prototype étudiant, pas un dispositif médical.** Rien ici ne constitue un diagnostic ni une validation clinique.

## Le projet

SeinTinelle est une application web : un site (pages HTML/CSS/JS dans `frontend/`) relié à un backend Flask (`app.py`). Elle rassemble trois modèles d'IA répartis en deux modules :

1. **Détection sur mammographies** : le modèle MammoScreen (Hugging Face, `ianpan/mammoscreen`, avec `ianpan/mammo-crop` pour le recadrage), appelé par le backend. Vues CC et MLO.
2. **Détection sur échographies** : un DenseNet-121 (transfert depuis ImageNet, 100 dernières couches dégelées) entraîné sur BreastMNIST (MedMNIST v2), avec Focal Loss, suréchantillonnage de la classe maligne, TTA ×7 et Grad-CAM.
3. **Prédiction du risque** : AsymMirai (Donnelly et al., Radiology 2024, fork du modèle Mirai) sur 4 vues mammographiques, combiné par fusion tardive à 4 variables cliniques (âge, IMC, antécédents familiaux, densité BI-RADS).

## Organisation du dépôt

| Élément | Contenu |
|---|---|
| `app.py` | Backend Flask : routes et chargement des trois modèles |
| `frontend/` | Pages HTML/CSS/JS : accueil, détection, prédiction, consultation de praticiens, FAQ, à propos, page de test de cohérence |
| `Mirai/` | AsymMirai, code de tiers sous licence MIT (voir `Mirai/LICENSE`) |
| Poids des modèles | Non inclus dans le dépôt, voir « Poids des modèles » plus bas |
| `notebooks/` | Notebooks Colab convertis en scripts, sans sorties ni images (détail ci-dessous) |
| `QUI_A_FAIT_QUOI.md` | Répartition du travail dans l'équipe et origine du code |

### Notebooks

| Script | Origine |
|---|---|
| `asymmirai_multimodal.py` | Ayoub Mernissi : AsymMirai + données cliniques, interface Gradio, batch sur cohorte synthétique, métriques de survie |
| `detection_rsna_v1.py` | Ayoub Mernissi : benchmark RSNA (données cliniques, ResNet50, fusion tardive) |
| `cnn_cbis_ddsm_et_random_forest_wisconsin.py` | Ayoub Mernissi : CNN sur CBIS-DDSM (exécution interrompue) et Random Forest sur Wisconsin |
| `echographie_breastmnist_densenet121.py` | Notebook de Rayan Oulboub (module échographie) |
| `detection_1.py` | Notebook de l'équipe (dossier de Lilou Parrin) : essai de MammoScreen |

## Qui a fait quoi

Le détail, avec les sources, est dans [QUI_A_FAIT_QUOI.md](QUI_A_FAIT_QUOI.md). En résumé, d'après le rapport du projet :

- **Lilou Parrin** : pages du site, intégration des trois modèles avec Flask, partie éthique, affiche.
- **Nathan Gautier** : état de l'art de la prédiction, démarches d'accès aux données, pages du frontend, refonte visuelle.
- **Maxime Lenfant** : site et premier modèle de détection, règles de travail GitHub.
- **Ayoub Mernissi** : choix et import d'AsymMirai puis extension multimodale, benchmark RSNA, code échographie avec Rayan, parties prédiction et introduction du rapport.
- **Rayan Oulboub** : chef de projet ; modèle de risque et échographie DenseNet-121 avec Ayoub, accès aux données CSAW.
- **Antoine Navarro** : état de l'art deep learning / machine learning, notebook Kaggle ADMANI, participation au site et au premier modèle de détection.

La part exacte de chacun dans l'échographie reste **À COMPLÉTER** : le dépôt d'origine n'a qu'un seul commit initial, sans attribution par fichier.

## Lancer l'application en local

```bash
pip install -r requirements.txt
python app.py
```

Ouvrir ensuite `frontend/index.html` dans un navigateur.

Avant de lancer `app.py`, il faut placer les poids des modèles aux emplacements attendus par le code (voir « Poids des modèles »). Sans eux, les modules concernés ne se chargent pas.

`app.py` démarre en mode debug sur `0.0.0.0` avec CORS ouvert : à utiliser uniquement en local, jamais exposé sur un réseau.

## Résultats et performances

Seuls figurent ici des chiffres que j'ai pu vérifier à la source, avec leur origine. **Notebook** : sortie enregistrée dans le notebook d'origine, relue cellule par cellule. **Source d'origine** : chiffre relu dans la publication ou la fiche du modèle, mesuré par ses auteurs et non par l'équipe. Aucun entraînement ni calcul n'a été relancé pour ce dépôt.

### Détection sur mammographies (MammoScreen, mesuré par l'auteur du modèle)

Source : fiche du modèle `ianpan/mammoscreen` sur Hugging Face. Entraînement sur CBIS-DDSM puis RSNA (challenge Kaggle), découpage 80 / 10 / 10 répété 3 fois, évaluation sur les 10 % de test.

| Mesure | Valeur |
|---|---|
| AUC, découpages 1 / 2 / 3 | 0,9464 / 0,9467 / 0,9422 (moyenne 0,9451, écart-type 0,002) |
| Sensibilité / spécificité, seuil 0,0072 | 98,1 % / 65,4 % (± 7,2) |
| Sensibilité / spécificité, seuil 0,0127 | 94,3 % / 78,7 % (± 0,9) |
| Sensibilité / spécificité, seuil 0,0184 | 90,5 % / 84,8 % (± 2,7) |

La fiche ne donne pas la taille du jeu de test. Ces valeurs décrivent le modèle publié, pas une mesure faite par l'équipe.

### Détection sur échographies (DenseNet-121, BreastMNIST)

Source : notebook `notebooks/echographie_breastmnist_densenet121.py`. Jeu de test de 156 images (42 bénignes, 114 malignes), TTA ×7, seuil fixé à la main à 0,735 dans le code.

| Exécution enregistrée | Accuracy | AUC | Précision | Rappel | F1 |
|---|---|---|---|---|---|
| Cellule 15 | 87,82 % | 0,8782 | 90,60 % | 92,98 % | 91,77 % |
| Cellule 21 (autre exécution) | 89,10 % | 0,9229 | 90,08 % | 95,61 % | 92,77 % |

Détail de la cellule 15 : classe bénigne précision 0,79 et rappel 0,74 (42 images), classe maligne précision 0,91 et rappel 0,93 (114 images). Le modèle repère donc mieux les cas malins que les cas bénins. Les deux exécutions donnent des valeurs différentes pour le même notebook, et le test est petit : à lire comme un ordre de grandeur (accuracy autour de 88-89 %), pas comme une performance établie.

### Prédiction du risque à 1-5 ans (AsymMirai, mesuré par ses auteurs)

Source : Donnelly et al., « AsymMirai: Interpretable Mammography-based Deep Learning Model for 1-5-year Breast Cancer Risk Prediction », Radiology 2024, 310(3):e232780. Évaluation sur la base EMBED (16 314 patientes pour l'horizon à 1 an).

| Horizon | AUC d'AsymMirai (IC à 95 %) | AUC de Mirai, pour comparaison |
|---|---|---|
| 1 an | 0,79 (0,73-0,85) | 0,84 |
| 3 ans | 0,68 (0,65-0,71) | 0,72 |
| 5 ans | 0,66 (0,63-0,69) | 0,71 |

Notre extension (fusion avec l'âge, l'IMC, les antécédents familiaux et la densité BI-RADS) n'a aucune performance mesurée : l'équipe n'a pas obtenu l'accès aux bases longitudinales (CSAW, EMBED), et le méta-modèle XGBoost n'a jamais été entraîné. La page interne `test_coherence.html` vérifie seulement que le score augmente avec l'âge, l'IMC, la densité et les antécédents ; ce n'est pas une mesure de performance.

### Benchmark RSNA (notebook `detection_rsna_v1.py`)

Source : sorties du notebook. Les lignes 2 à 5 portent sur un jeu équilibré de 1000 patientes (500 cancers, 500 sains), découpage 80 / 20 : 200 patientes de test.

| Modèle | AUC | Cellule |
|---|---|---|
| Random Forest, âge et implant (jeu équilibré) | 0,6082 | 3 |
| ResNet50 sur l'imagerie | 0,4735 | 4 |
| EfficientNet sur l'imagerie | 0,4101 | 4 |
| Fusion tardive (méta-modèle) | 0,5594 | 6 |
| XGBoost / Random Forest / SVM, âge et implant (benchmark clinique initial) | 0,6350 / 0,6647 / 0,3873 | 2 |

Les modèles d'imagerie testés ici sont à ou sous 0,5 : ils ne montrent aucun signal exploitable sur ce protocole, et la fusion n'apporte rien par rapport au modèle clinique seul.

### Données tabulaires (Wisconsin, notebook `cnn_cbis_ddsm_et_random_forest_wisconsin.py`)

Random Forest (100 arbres) sur 569 échantillons (357 bénins, 212 malins), découpage 80 / 20 (455 / 114) : accuracy 97,37 %, AUC 0,9929 (cellule 14). Ce jeu contient des mesures cellulaires, pas des images : il ne dit rien de la performance sur mammographies. Le CNN sur CBIS-DDSM du même notebook a été interrompu et n'a donné aucune métrique.

### Écarts avec le rapport du projet

Le rapport affirme une précision de 92 % pour l'échographie : aucune sortie de notebook ne la confirme, elle n'est pas reprise ici. Il donne un seuil de 0,699 alors que le code utilise 0,735, une AUC d'AsymMirai à 1 an de 0,69 dans une section (la publication donne 0,79), et des poids de fusion de 0,70 / 0,30 alors que le code utilise 0,50 / 0,50.

## Provenance et licences

- Code de l'application : travail collectif de l'équipe SeinTinelle. Ce dépôt a un historique neuf : l'attribution par fichier n'y figure pas, la répartition est dans `QUI_A_FAIT_QUOI.md`.
- AsymMirai : Donnelly, Moffett, Barnett, Trivedi, Schwartz, Lo, Rudin, Radiology 2024 ; code sous licence MIT (voir `Mirai/LICENSE`). Mirai original : Yala et al. (2021).
- MammoScreen : modèle tiers appelé via Hugging Face, licence Apache-2.0 d'après sa fiche. mammo-crop : modèle tiers du même auteur, licence Apache-2.0 d'après sa fiche (la fiche ne précise pas la licence des données d'entraînement RSNA).
- BreastMNIST : Yang et al. (2023), MedMNIST v2. Le README de MedMNIST indique le code sous Apache-2.0 et les jeux de données sous CC BY 4.0 (sauf DermaMNIST) ; il demande aussi de citer les articles des données sources.
- Aucune donnée patient dans ce dépôt.

## Poids des modèles

Ces fichiers ne sont volontairement pas dans le dépôt. Ce que j'ai pu vérifier sur les pages officielles :

| Fichier attendu par le code | Statut | Pourquoi il n'est pas inclus |
|---|---|---|
| `Mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p` (Mirai, 50 Mo) | Droits contradictoires | Le README de `yala/Mirai` dit que le modèle entraîné et le code sont sous licence MIT et renvoie vers une image Docker pour récupérer les poids. Le rapport de l'équipe écrit au contraire que ces poids ne sont pas redistribuables sans accord institutionnel. Dans le doute, je ne les redistribue pas : on les télécharge à la source. |
| `Mirai/snapshots/trained_asymmirai.pt` (AsymMirai, 45 Mo) | Licence non précisée | Le dépôt `jdonnelly36/AsymMirai` ne contient pas ce fichier : son README renvoie vers un lien de téléchargement et dit seulement que le modèle est « open to access and use » avec attribution au groupe. Aucune licence n'est donnée pour les poids, et rien n'est dit sur l'effet des conditions d'accès à la base EMBED. |
| `breast_cancer_densenet121_final.keras` (échographie, 53 Mo) | Travail de l'équipe | Modèle entraîné par l'équipe sur BreastMNIST. Pas de licence posée, et le fichier est le travail de plusieurs personnes : je ne le publie pas sans leur accord. |

Les poids de MammoScreen et de mammo-crop sont téléchargés automatiquement depuis Hugging Face au démarrage (`snapshot_download` dans `app.py`).

Pour faire tourner l'application : récupérer les poids AsymMirai (et Mirai si besoin) via les liens du README de leurs dépôts d'origine, les placer dans `Mirai/snapshots/`, et mettre le fichier `.keras` à la racine du dépôt.
