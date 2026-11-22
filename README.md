# Breast Cancer Detection and Risk Prediction (SeinTinelle)

**SeinTinelle** is a web application built as an E3 group project (ESIEE Paris, 2025-2026) and presented at the Project Day. Team of six students, supervised by Mrs. Imen Kachouri.

> **Student prototype, not a medical device.** Nothing here constitutes a diagnosis or a clinical validation.
>
> No model weights are included in this repository (see "Model weights"). No license file is set at the root (all rights reserved by default). The code is the work of several teammates; credits are in [WHO_DID_WHAT.md](WHO_DID_WHAT.md).

## The project

SeinTinelle is a web application: a website (HTML/CSS/JS pages in `frontend/`) connected to a Flask backend (`app.py`). It combines three AI models split across two modules:

1. **Mammography detection**: the MammoScreen model (Hugging Face, `ianpan/mammoscreen`, with `ianpan/mammo-crop` for cropping), called by the backend. CC and MLO views.
2. **Ultrasound detection**: a DenseNet-121 (ImageNet transfer learning, last 100 layers unfrozen) trained on BreastMNIST (MedMNIST v2), with Focal Loss, oversampling of the malignant class, 7x TTA and Grad-CAM.
3. **Risk prediction**: AsymMirai (Donnelly et al., Radiology 2024, a fork of the Mirai model) on 4 mammographic views, combined by late fusion with 4 clinical variables (age, BMI, family history, BI-RADS density).

The website itself is in French (the target audience of the project).

## Repository layout

| Item | Content |
|---|---|
| `app.py` | Flask backend: routes and loading of the three models |
| `frontend/` | HTML/CSS/JS pages: home, detection, prediction, practitioner lookup, FAQ, about, consistency test page |
| `Mirai/` | AsymMirai, third-party code under the MIT license (see `Mirai/LICENSE`) |
| Model weights | Not included in the repository, see "Model weights" below |
| `notebooks/` | Colab notebooks converted to scripts, without outputs or images (details below) |
| `WHO_DID_WHAT.md` | Work split within the team and origin of the code |

### Notebooks

| Script | Origin |
|---|---|
| `asymmirai_multimodal.py` | Ayoub Mernissi: AsymMirai + clinical data, Gradio interface, batch run on a synthetic cohort, survival metrics |
| `detection_rsna_v1.py` | Ayoub Mernissi: RSNA benchmark (clinical data, ResNet50, late fusion) |
| `cnn_cbis_ddsm_et_random_forest_wisconsin.py` | Ayoub Mernissi: CNN on CBIS-DDSM (run interrupted) and Random Forest on Wisconsin |
| `echographie_breastmnist_densenet121.py` | Rayan Oulboub's notebook (ultrasound module) |
| `detection_1.py` | Team notebook (Lilou Parrin): MammoScreen trial |

## Who did what

The details, with sources, are in [WHO_DID_WHAT.md](WHO_DID_WHAT.md). In short, according to the team:

- **Lilou Parrin**: website pages, integration of the three models with Flask, ethics section, poster.
- **Nathan Gautier**: state of the art on prediction, data access requests, frontend pages, visual redesign.
- **Maxime Lenfant**: website and first detection model, GitHub working rules.
- **Ayoub Mernissi**: selection and import of AsymMirai then multimodal extension, RSNA benchmark, ultrasound code with Rayan, prediction and introduction sections of the report.
- **Rayan Oulboub**: project lead; risk model and DenseNet-121 ultrasound with Ayoub, access to the CSAW data.
- **Antoine Navarro**: deep learning / machine learning state of the art, Kaggle ADMANI notebook, contribution to the website and the first detection model.

The ultrasound module was developed jointly by Rayan Oulboub and Ayoub Mernissi.

## Running the application locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `frontend/index.html` in a browser.

Before running `app.py`, the model weights must be placed where the code expects them (see "Model weights"). Without them, the corresponding modules do not load.

`app.py` starts in debug mode on `0.0.0.0` with open CORS: use it only locally, never exposed on a network.

## Results and performance

Only figures that could be checked at their source are listed here, with their origin. **Notebook**: output recorded in the original Colab notebook (outputs are not included in this repository). **Original source**: figure taken from the publication or the model card, measured by its authors and not by the team. Nothing was retrained or recomputed for this repository.

### Mammography detection (MammoScreen, measured by the model's author)

Source: the `ianpan/mammoscreen` model card on Hugging Face. Trained on CBIS-DDSM then RSNA (Kaggle challenge), 80 / 10 / 10 split repeated 3 times, evaluated on the 10% test set.

| Metric | Value |
|---|---|
| AUC, splits 1 / 2 / 3 | 0.9464 / 0.9467 / 0.9422 (mean 0.9451, standard deviation 0.002) |
| Sensitivity / specificity, threshold 0.0072 | 98.1% / 65.4% (± 7.2) |
| Sensitivity / specificity, threshold 0.0127 | 94.3% / 78.7% (± 0.9) |
| Sensitivity / specificity, threshold 0.0184 | 90.5% / 84.8% (± 2.7) |

The model card does not give the size of the test set. These values describe the published model, not a measurement made by the team.

### Ultrasound detection (DenseNet-121, BreastMNIST)

Source: notebook `notebooks/echographie_breastmnist_densenet121.py`. Test set of 156 images (42 benign, 114 malignant), 7x TTA, threshold set by hand to 0.735 in the code.

| Recorded run | Accuracy | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Cell 15 | 87.82% | 0.8782 | 90.60% | 92.98% | 91.77% |
| Cell 21 (another run) | 89.10% | 0.9229 | 90.08% | 95.61% | 92.77% |

Cell 15 detail: benign class precision 0.79 and recall 0.74 (42 images), malignant class precision 0.91 and recall 0.93 (114 images). The model therefore detects malignant cases better than benign ones. The two runs give different values for the same notebook, and the test set is small: read this as an order of magnitude (accuracy around 88-89%), not as an established performance.

### Risk prediction at 1-5 years (AsymMirai, measured by its authors)

Source: Donnelly et al., "AsymMirai: Interpretable Mammography-based Deep Learning Model for 1-5-year Breast Cancer Risk Prediction", Radiology 2024, 310(3):e232780. Evaluated on the EMBED dataset (16,314 patients for the 1-year horizon).

| Horizon | AsymMirai AUC (95% CI) | Mirai AUC, for comparison |
|---|---|---|
| 1 year | 0.79 (0.73-0.85) | 0.84 |
| 3 years | 0.68 (0.65-0.71) | 0.72 |
| 5 years | 0.66 (0.63-0.69) | 0.71 |

Our extension (fusion with age, BMI, family history and BI-RADS density) has no measured performance: the team did not get access to the longitudinal datasets (CSAW, EMBED), and the XGBoost meta-model was never trained. The internal page `test_coherence.html` only checks that the score increases with age, BMI, density and family history; it is not a performance measurement.

### RSNA benchmark (notebook `detection_rsna_v1.py`)

Source: outputs of the original notebook. Rows 2 to 5 use a balanced set of 1,000 patients (500 cancer, 500 healthy), 80 / 20 split: 200 test patients.

| Model | AUC | Cell |
|---|---|---|
| Random Forest, age and implant (balanced set) | 0.6082 | 3 |
| ResNet50 on imaging | 0.4735 | 4 |
| EfficientNet on imaging | 0.4101 | 4 |
| Late fusion (meta-model) | 0.5594 | 6 |
| XGBoost / Random Forest / SVM, age and implant (initial clinical benchmark) | 0.6350 / 0.6647 / 0.3873 | 2 |

The imaging models tested here are at or below 0.5: they show no usable signal under this protocol, and fusion brings nothing over the clinical model alone.

### Tabular data (Wisconsin, notebook `cnn_cbis_ddsm_et_random_forest_wisconsin.py`)

Random Forest (100 trees) on 569 samples (357 benign, 212 malignant), 80 / 20 split (455 / 114): accuracy 97.37%, AUC 0.9929 (cell 14). This dataset contains cell measurements, not images: it says nothing about performance on mammograms. The CNN on CBIS-DDSM in the same notebook was interrupted and produced no metrics.

## Provenance and licenses

- Application code: collective work of the SeinTinelle team. Contributions of each member are detailed in `WHO_DID_WHAT.md`.
- AsymMirai: Donnelly, Moffett, Barnett, Trivedi, Schwartz, Lo, Rudin, Radiology 2024; code under the MIT license (see `Mirai/LICENSE`). Original Mirai: Yala et al. (2021).
- MammoScreen: third-party model called via Hugging Face, Apache-2.0 license according to its model card. mammo-crop: third-party model by the same author, Apache-2.0 license according to its model card (the card does not state the license of the RSNA training data).
- BreastMNIST: Yang et al. (2023), MedMNIST v2. The MedMNIST README states the code is under Apache-2.0 and the datasets under CC BY 4.0 (except DermaMNIST); it also asks to cite the papers of the source datasets.
- No patient data in this repository.

## Model weights

These files are deliberately not in the repository. Status according to the official pages:

| File expected by the code | Status | Why it is not included |
|---|---|---|
| `Mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p` (Mirai, 50 MB) | Conflicting rights | The `yala/Mirai` README says the trained model and the code are under the MIT license and points to a Docker image to get the weights. The team could not confirm that the weights themselves may be redistributed, so they are not included: download them from the source. |
| `Mirai/snapshots/trained_asymmirai.pt` (AsymMirai, 45 MB) | License not specified | The `jdonnelly36/AsymMirai` repository does not contain this file: its README points to a download link and only says the model is "open to access and use" with credit to the group. No license is given for the weights, and nothing is said about the effect of the EMBED dataset access conditions. |
| `breast_cancer_densenet121_final.keras` (ultrasound, 53 MB) | Team work | Model trained by the team on BreastMNIST. No license has been set, and the file is the joint work of several members, so it is not published. |

The MammoScreen and mammo-crop weights are downloaded automatically from Hugging Face at startup (`snapshot_download` in `app.py`).

To run the application: get the AsymMirai weights (and Mirai if needed) through the links in the README of their original repositories, place them in `Mirai/snapshots/`, and put the `.keras` file at the root of the repository.
