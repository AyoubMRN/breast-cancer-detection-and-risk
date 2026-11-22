# Who did what (SeinTinelle project)

E3 project (ESIEE Paris, 2025-2026): a website to raise awareness about breast cancer, with a detection module (mammography, ultrasound) and a long-term risk prediction module. Team of six students, supervised by Mrs. Imen Kachouri.

This repository contains the application code (`app.py`, `frontend/`, `Mirai/`) and the training notebooks (`notebooks/`).

## Sources

1. The "Work split" section of the team report "SeinTinelle: Project report". It was drawn up from the weekly progress reports and the contributions to the GitHub repository and the website.
2. The "Team" section of the team README (names and programs).

The roles below follow these two sources.

## Work split

| Member | Program (README) | Role |
|---|---|---|
| Lilou Parrin | Data Science & AI | Website: "Consulter un médecin" (find a doctor), "À propos de nous" (about us) and "FAQ" pages, redesign of the home and detection pages. Technical integration of the three models (mammography, ultrasound, future risk) with the Flask backend and display of the results. Ethics section of the report, project poster, contribution to the logo. |
| Nathan Gautier | Data Science & AI | State of the art on prediction (ML / deep learning comparison), data access requests for mammography datasets (OMI-DB, ADMANI, PLCO). Frontend: "Restez Informées" (stay informed) page, first version of "À propos", prevention popup, visual redesign. Video (shooting plan, coordination), report outline, poster mock-up. |
| Maxime Lenfant | Artificial Intelligence and Cybersecurity | Creation of the website and implementation of the first detection model (mammography). Shared GitHub repository, contribution rules, Git guide. Explanation of the AUC-ROC and watch on comparable projects. |
| Ayoub Mernissi | Artificial Intelligence and Cybersecurity | Future risk prediction module: targeted state of the art, selection and import of AsymMirai, multimodal extension (age, BMI, family history, BI-RADS density), RSNA benchmark. Ultrasound module code in collaboration with Rayan. Report (prediction section, introduction, cover), video script, logo and communication material. |
| Rayan Oulboub (project lead) | Artificial Intelligence and Cybersecurity | In a pair with Ayoub: 1-5 year risk model and ultrasound model (DenseNet-121 on BreastMNIST, Focal Loss, TTA, oversampling, Grad-CAM). CSAW access requests. Report (introduction, ultrasound, prediction, bibliography, appendices), logo and marketing, video (script, voice-over). |
| Antoine Navarro | Computer Science | Deep learning / machine learning state of the art for breast cancer diagnosis, references (federated learning, late fusion), Kaggle ADMANI notebook. Contribution to the creation of the website and the implementation of the first detection model. |

## Where each person's code is

| Folder / file | Content | Origin |
|---|---|---|
| `app.py`, `frontend/`, `requirements.txt` | Flask application and website | Team |
| `Mirai/` | AsymMirai fork (MIT license, see `Mirai/LICENSE`) | Donnelly et al., https://github.com/jdonnelly36/AsymMirai |
| `notebooks/asymmirai_multimodal.py` | Imaging + clinical ensemble model, Gradio interface | Ayoub Mernissi |
| `notebooks/detection_rsna_v1.py` | RSNA benchmark v1 | Ayoub Mernissi |
| `notebooks/cnn_cbis_ddsm_et_random_forest_wisconsin.py` | CBIS-DDSM CNN (interrupted) and Wisconsin Random Forest | Ayoub Mernissi |
| `notebooks/echographie_breastmnist_densenet121.py` | DenseNet-121 ultrasound on BreastMNIST | Rayan Oulboub; module developed jointly with Ayoub Mernissi |
| `notebooks/detection_1.py` | MammoScreen detection (Hugging Face) | Lilou Parrin |
| `breast_cancer_densenet121_final.keras` | Ultrasound model weights (not included in this repository) | Team |

The notebooks are "# %%" scripts converted from the Colab notebooks, without outputs or images.

## License

The repository has no license at the root: all rights reserved by default. Third-party components keep their own licenses (see the README).
