# Script conversion of the cells of: MIRAI_multimodal_FINAL_commented.ipynb (personal Drive)
# Outputs and images removed.

# %% [markdown] cellule 1
# # 🩺 AsymMirai Multimodal — Notebook Complet
# ### Breast cancer risk prediction: Vision + Clinical Data
# 
# **Full pipeline in 6 steps:**
# 1. 📦 Installing the dependencies
# 2. 🧠 Loading the AsymMirai model
# 3. 🔬 Ensemble Model (imaging + clinical fusion)
# 4. 🖥️ Multimodal Gradio interface with heatmaps
# 5. 📊 Batch Testing on a cohort of patients
# 6. 📈 Survival metrics (time-dependent AUC-ROC + C-Index)
# 
# > ⚠️ **Research use only.** Does not replace the opinion of a qualified radiologist.
# >
# > **Prerequisite:** Place `asymmirai_weights.pth` in `/content/snapshots/` before cell 2.
# 
# > 💡 **Credit and model source:**
# > This project builds on the pre-trained deep learning model **AsymMirai**, developed by Jon Donnelly as part of his research.
# > Link to the official repository: [GitHub - jdonnelly36/AsymMirai](https://github.com/jdonnelly36/AsymMirai)

# %% [markdown] cellule 2
# ## 📦 Cell 1 — Installing the dependencies

# %% cellule 3
# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — INSTALLING THE DEPENDENCIES
# Installs all the libraries needed for the 4 blocks of the project.
# Estimated duration: ~2-3 minutes on Colab.
# ═══════════════════════════════════════════════════════════════════════════

# gradio      → interface web interactive (upload images + sliders)
# huggingface_hub → access to pre-trained models on the HF Hub
# lifelines    → survival metrics (C-Index, Kaplan-Meier)
# xgboost      → imaging + clinical fusion meta-model
# scikit-survival → AUC-ROC temporelle (cumulative_dynamic_auc)
# tqdm         → progress bar for long loops
!pip install --upgrade gradio huggingface_hub lifelines xgboost scikit-survival tqdm -q

# Checking whether scikit-survival is available (optional but recommended)
try:
    import sksurv
    print('✅ scikit-survival available → rigorous time-dependent AUC enabled')
except ImportError:
    print('ℹ️  scikit-survival missing → falling back to naive AUC (sklearn)')

print('\n✅ Installation finished — go to the next cell')

# %% [markdown] cellule 4
# ## 🔧 Cellule 2 — Imports globaux & clonage AsymMirai

# %% cellule 5
# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — GLOBAL IMPORTS AND CLONING THE ASYMMIRAI REPOSITORY
# All the libraries used in the following cells are imported
# here once to avoid any redundancy.
# ═══════════════════════════════════════════════════════════════════════════

# ── Stdlib Python ─────────────────────────────────────────────────────────
# os/sys : path handling and Python PATH
# json/pickle : serialization of the results and of the XGBoost meta-model
# warnings : suppression of minor non-blocking warnings
# random : reproducibility of the synthetic cohort
import os, sys, json, pickle, warnings, random
from pathlib import Path
from datetime import datetime


# ── Scientific computing ───────────────────────────────────────────────────
# numpy  : matrix operations on the heatmaps and risk vectors
# pandas : handling of clinical data and CSV/JSON export
import numpy as np
import pandas as pd

# ── Deep Learning ──────────────────────────────────────────────────────────
# torch           : GPU inference of the AsymMirai model
# F               : softmax applied to the output logits
# torchvision     : normalization pipeline for the mammograms
# PIL.Image       : loading and manipulating images
# tqdm            : visual progress during batch testing
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from tqdm import tqdm


# ── Visualisation ──────────────────────────────────────────────────────────
# matplotlib (Agg backend) : generating charts without screen display
# cm : 'jet' colormap used for the asymmetry heatmaps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Statistical evaluation ─────────────────────────────────────────────────
# roc_curve / auc    : ROC curves for the 3 time horizons
# brier_score_loss   : calibration of the predicted probabilities
# roc_auc_score      : naive AUC (fallback if scikit-survival is missing)
from sklearn.metrics import roc_curve, auc as sklearn_auc, brier_score_loss, roc_auc_score

warnings.filterwarnings('ignore')

# ── Cloning the AsymMirai repository ──
!rm -rf /content/Mirai
!git clone https://github.com/jdonnelly36/AsymMirai.git /content/Mirai -q

sys.path.append(os.path.abspath('/content/Mirai'))
sys.path.append(os.path.abspath('/content/Mirai/asymmetry_model'))

# ── GPU/CPU device detection ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Device configured : {device}')
print('✅ Imports finished — go to the next cell')

# %% [markdown] cellule 6
# ## 🧠 Cell 3 — Loading the AsymMirai model
# > **Prerequisite:** The file `asymmirai_weights.pth` must be present in `/content/snapshots/`.

# %% cellule 7
# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — LOADING THE ASYMMIRAI MODEL
# Loads the model weights from disk and prepares inference.
# The model is loaded ONLY ONCE and reused in all the cells.
# ═══════════════════════════════════════════════════════════════════════════

# get_asymmetries is an internal module of the AsymMirai repository (cloned in cell 2).
# It exposes the get_asymmetries() function which computes the asymmetry maps
# between the left and right breast from the 4 mammographic views.

from get_asymmetries import get_asymmetries

# ── Loading function ───────────────────────────────────────────────────────
# We use torch.load() rather than model.load_state_dict() because the
# .pth file contains the complete model object (LocalizedDifModel), not only
# the weights. weights_only=False is required for this serialization format.
# model.eval() disables Dropout/BatchNorm for inference (not training).
def load_asymmirai(weights_path: str = '/content/snapshots/asymmirai_weights.pth'):
    """
    Charge le modèle AsymMirai depuis les poids pré-entraînés.

    Paramètres
    ----------
    weights_path : chemin vers le fichier .pth des poids

    Retourne
    --------
    torch.nn.Module en mode eval, ou None en cas d'erreur
    """
    print(f'⏳ Loading the weights from : {weights_path}')
    if not os.path.exists(weights_path):
        print(f'❌ File not found : {weights_path}')
        print('   → Place asymmirai_weights.pth in /content/snapshots/')
        return None
    try:
        model = torch.load(weights_path, map_location=device, weights_only=False)
        model.eval()
        print('✅ AsymMirai model loaded successfully')
        return model
    except Exception as e:
        print(f'❌ Loading error : {e}')
        return None

# Single loading — global variable reused in all the cells
asym_model = load_asymmirai()

# %% [markdown] cellule 8
# ## 🔬 Cellule 4 — Bloc 1 : Ensemble Model Multimodal
# **Fusion tardive (Late Fusion)** : vecteur de risque AsymMirai [5 ans] + 4 variables cliniques → XGBoost.
# 
# | Variable | Encodage |
# |---|---|
# | Age | Min-max normalized [20, 80] |
# | BMI | Min-max normalized [15, 45] |
# | Family history | Binary 0/1 |
# | BI-RADS density | Ordinal A=0, B=0.33, C=0.67, D=1 |

# %% cellule 9
# ═══════════════════════════════════════════════════════════════════════════
# CELLULE 4 — BLOC 1 : ENSEMBLE MULTIMODAL (version heatmap-based)
# ═══════════════════════════════════════════════════════════════════════════

# ── BI-RADS density encoding ───────────────────────────────────────────────
# The ACR classification (A, B, C, D) is ordinal; it is mapped to [0,3]
# before min-max normalization in encode_clinical_features().

BIRADS_DENSITY_MAP = {'A': 0.0, 'B': 1.0, 'C': 2.0, 'D': 3.0}


# ── encode_clinical_features ──────────────────────────────────────────────
# Normalizes the 4 clinical variables to [0,1] to make them homogeneous
# with the AsymMirai risk vector before concatenation in XGBoost.
# Age    : min-max [30, 80]
# IMC    : min-max [15, 45]
# ATCD   : binaire 0/1
# Density: ordinal divided by 3 → [0, 0.33, 0.67, 1.0]
def encode_clinical_features(age, imc, antecedents_familiaux, densite_birads):
    age_norm    = np.clip((age - 30) / (80 - 30), 0.0, 1.0)
    imc_norm    = np.clip((imc - 15) / (45 - 15), 0.0, 1.0)
    atcd_enc    = 1.0 if antecedents_familiaux else 0.0
    densite_enc = BIRADS_DENSITY_MAP.get(str(densite_birads).upper(), 1.0) / 3.0
    return np.array([age_norm, imc_norm, atcd_enc, densite_enc], dtype=np.float32)



# ── preprocess_rsna ────────────────────────────────────────────────────────
# Preprocessing pipeline compliant with the RSNA Mammography standard:
#  1. Optional horizontal flip (right views brought to the same orientation)
#  2. Conversion en niveaux de gris (L)
#  3. Percentile 1-99 normalization to get rid of extreme artifacts
#  4. Redimensionnement 256×256 + normalisation ImageNet (μ=0.5, σ=0.5)
#  5. unsqueeze(0): adds the batch dimension expected by the model
def preprocess_rsna(img, flip=False):
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_gray = img.convert('L')
    arr = np.array(img_gray, dtype=np.float32)
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    if p99 > p1:
        arr = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
    else:
        arr = arr / 255.0
    arr_uint8 = (arr * 255).astype(np.uint8)
    img_rgb = Image.fromarray(arr_uint8).convert('RGB')
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return t(img_rgb).unsqueeze(0)



# ── extract_heatmap_score ──────────────────────────────────────────────────
# Converts the raw 8×8 heatmap of the model into a scalar score [0, 1].
# Strategy: we measure the 'contrast' (max – mean) / mean, which is high
# if the model detected a localized abnormal area, low if the heatmap
# is uniform. A soft sigmoid maps the contrast into [0, 1].
# The coordinates of the maximum (row, col) are used to locate the asymmetry.
def extract_heatmap_score(heatmap_tensor):
    """
    Score d'asymétrie depuis la heatmap 8x8.
    On utilise l'écart-type et le max BRUT (avant normalisation)
    pour garder une vraie variance entre patientes.
    """
    h = heatmap_tensor.cpu().numpy().squeeze().astype(np.float32)

    # Raw values — no normalization here
    h_max    = float(h.max())
    h_mean   = float(h.mean())
    h_std    = float(h.std())

    # Score = relative intensity of the peak compared to the mean
    # If everything is flat → h_max ≈ h_mean → low score
    # If strong peak     → h_max >> h_mean → high score
    if h_mean > 0:
        contrast = (h_max - h_mean) / (h_mean + 1e-6)
    else:
        contrast = 0.0

    # Normalize the contrast into [0, 1] with a soft sigmoid
    # contrast ~ 0.5 → score ~ 0.3 (low)
    # contrast ~ 2.0 → score ~ 0.6 (moderate)
    # contrast ~ 5.0 → score ~ 0.9 (high)
    score = float(1 / (1 + np.exp(-0.8 * (contrast - 2.5))))

    # Coordinates of the maximum
    idx      = np.unravel_index(np.argmax(h), h.shape)
    row, col = int(idx[0]), int(idx[1])

    return score, row, col



# ── get_asymmirai_risk_vector ──────────────────────────────────────────────
# Orchestrates the full AsymMirai inference on the 4 mammographic views:
#  1. Preprocessing of each view (RCC and RMLO flipped to normalize the side)
#  2. Inference without gradient (torch.no_grad) → output logits + heatmaps
#  3. Computation of the global asymmetry score (mean of CC + MLO)
#  4. Building the 5-year risk vector by progressive scaling:
#     an1 = asym×0.50, an2 = ×0.65, ..., an5 = ×1.00
#  5. Remapping heatmap 8×8 → 256×256 image space for display
def get_asymmirai_risk_vector(model, img_lcc, img_rcc, img_lmlo, img_rmlo, dev):

    t_lcc  = preprocess_rsna(img_lcc,  flip=False).to(dev)
    t_rcc  = preprocess_rsna(img_rcc,  flip=True).to(dev)
    t_lmlo = preprocess_rsna(img_lmlo, flip=False).to(dev)
    t_rmlo = preprocess_rsna(img_rmlo, flip=True).to(dev)

    with torch.no_grad():
        result = model(t_lcc, t_rcc, t_lmlo, t_rmlo)

    if result is None or not isinstance(result, (tuple, list)) or len(result) < 2:
        raise ValueError(f"Sortie inattendue : {type(result)}")

    output, other = result

    score_cc,  row_cc,  col_cc  = extract_heatmap_score(other[0]['heatmap'])
    score_mlo, row_mlo, col_mlo = extract_heatmap_score(other[1]['heatmap'])

    # Global score = mean of the two views
    asym_score = (score_cc + score_mlo) / 2.0

    # Log of the scores for debugging
    print(f"  score_cc={score_cc:.3f}  score_mlo={score_mlo:.3f}  asym_score={asym_score:.3f}")

    base = asym_score
    risk_vector = np.array([
        base * 0.50,
        base * 0.65,
        base * 0.78,
        base * 0.88,
        base * 1.00,
    ], dtype=np.float32)
    risk_vector = np.clip(risk_vector, 0.0, 1.0)

    # Coordinates in image space (heatmap 8x8 → 256x256 space)
    cc_x  = float(col_cc  * 256 / 8 + 16)
    cc_y  = float(row_cc  * 256 / 8 + 16)
    mlo_x = float(col_mlo * 256 / 8 + 16)
    mlo_y = float(row_mlo * 256 / 8 + 16)

    asym_meta = {
        'cc_x':  cc_x,  'cc_y':  cc_y,
        'mlo_x': mlo_x, 'mlo_y': mlo_y,
        'heatmap_cc':  other[0]['heatmap'].cpu().numpy().squeeze(),
        'heatmap_mlo': other[1]['heatmap'].cpu().numpy().squeeze(),
    }
    return risk_vector, asym_meta



# ── _heuristic_clinical_score ─────────────────────────────────────────────
# Fallback clinical score when XGBoost is not trained yet.
# Weights taken from the literature on breast cancer risk factors:
# breast density (40%) and family history (35%) dominate.
def _heuristic_clinical_score(clinical_vector):
    age_n, imc_n, atcd, densite = clinical_vector
    return 0.15 * age_n + 0.10 * imc_n + 0.35 * atcd + 0.40 * densite



# ── EnsembleModel ──────────────────────────────────────────────────────────
# Classe de fusion tardive (Late Fusion) combinant :
#   - The AsymMirai risk vector (5 values, horizons 1→5 years)
#   - The 4 encoded clinical features
# → 9-dimensional vector passed to an XGBClassifier.
#
# __init__   : loads the serialized meta-model if available
# train()    : trains XGBoost on labeled data (event_5y)
# predict_risk() : returns the final score + the categorical risk level
#   XGBoost mode  : predict_proba()[1] (probability of the positive class)
#   Heuristic mode : weighted average score_img (50%) + score_cli (50%)
class EnsembleModel:

    def __init__(self, model_path=None):
        self.xgb_model  = None
        self.is_trained = False
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.xgb_model = pickle.load(f)
            self.is_trained = True
            print(f'XGBoost charge depuis {model_path}')

    def train(self, X, y_event, save_path='/content/ensemble_meta_model.pkl'):
        from xgboost import XGBClassifier
        self.xgb_model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='auc', random_state=42
        )
        self.xgb_model.fit(X, y_event)
        self.is_trained = True
        with open(save_path, 'wb') as f:
            pickle.dump(self.xgb_model, f)
        print(f'XGBoost saved : {save_path}')

    def predict_risk(self, risk_vector, clinical_vector):
        mirai_feats = risk_vector[:5] if len(risk_vector) >= 5 else \
                      np.pad(risk_vector, (0, 5 - len(risk_vector)))
        X = np.concatenate([mirai_feats, clinical_vector]).reshape(1, -1)

        if self.is_trained and self.xgb_model is not None:
            score_final = float(self.xgb_model.predict_proba(X)[0, 1])
            score_cli   = float(_heuristic_clinical_score(clinical_vector))
        else:
            score_img = float(mirai_feats[4])
            age_n, imc_n, atcd, densite = clinical_vector
            score_cli = (
                0.35 * atcd    +
                0.30 * densite +
                0.20 * age_n   +
                0.15 * imc_n
            )
            score_final = 0.50 * score_img + 0.50 * score_cli

        if   score_final < 0.10: niveau = 'Faible (< 10%)'
        elif score_final < 0.20: niveau = 'Modere (10-20%)'
        elif score_final < 0.35: niveau = 'Eleve (20-35%)'
        else:                    niveau = 'Tres eleve (> 35%)'

        return {
            'score_final':    round(score_final, 4),
            'score_imagerie': round(float(np.mean(mirai_feats)), 4),
            'score_clinique': round(float(score_cli), 4),
            'niveau_risque':  niveau,
            'contributions':  {
                'imagerie': '50%',
                'clinique': '50%',
            },
        }


META_MODEL_PATH = '/content/ensemble_meta_model.pkl'
ensemble = EnsembleModel(
    model_path=META_MODEL_PATH if os.path.exists(META_MODEL_PATH) else None
)
print('Bloc 1 pret (mode', 'XGBoost' if ensemble.is_trained else 'heuristique fallback', ')')

# %% cellule 10
# ═══════════════════════════════════════════════════════════════════════════
# MINIMAL DIAGNOSTIC CELL — Quick check of the model output
#
# Goal: make sure asym_model(t, t, t, t) does return a tuple
#         (output_tensor, list_of_dicts_with_heatmap).
# We use 4 identical images (black dummy 256×256) because the model expects
# exactly 4 input tensors (L-CC, R-CC, L-MLO, R-MLO).
# The output is a tensor [1, 2] (logits for 2 classes); 'other' is a
# list of 2 dicts each containing an 8×8 heatmap and the argmins.
# Run only once to validate the architecture before the interface.
# ═══════════════════════════════════════════════════════════════════════════

# Diagnostic cell — run ONCE before the interface
dummy = Image.new('RGB', (256, 256))
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
t = transform(dummy).unsqueeze(0).to(device)

with torch.no_grad():
    result = asym_model(t, t, t, t)

output, other = result

print("=== output ===")
print("  shape :", output.shape)
print("  valeurs :", F.softmax(output, dim=1))

print("\n=== other ===")
print("  type :", type(other))
print("  contenu :", other)

if isinstance(other, (list, tuple)):
    for i, o in enumerate(other):
        print(f"\n  other[{i}] type : {type(o)}")
        if isinstance(o, dict):
            for k, v in o.items():
                print(f"    '{k}' → type={type(v)}, value={v}")

# %% [markdown] cellule 11
# ## 🖥️ Cellule 5 — Bloc 2 : Interface Gradio Multimodale
# Launches the interactive interface. Use the **public `.gradio.live` link** generated in the output.
# 
# **Features:**
# - Upload of the 4 mammographic views (L-CC, R-CC, L-MLO, R-MLO)
# - Entry of the clinical data (age, BMI, history, BI-RADS density)
# - Combined risk report + overlaid asymmetry heatmaps

# %% cellule 12
# ═══════════════════════════════════════════════
# CELLULE DIAGNOSTIC COMPLET
# ═══════════════════════════════════════════════
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# GOAL: validate that the model produces varied scores on different inputs
# (otherwise the model would be degenerate and always return the
# same value regardless of the images).
#
# preprocess_rsna is redefined here locally (deliberate duplication) to
# avoid any dependency on cell 4 when run in isolation.

def preprocess_rsna(img, flip=False):
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_gray = img.convert('L')
    arr = np.array(img_gray, dtype=np.float32)
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    if p99 > p1:
        arr = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
    else:
        arr = arr / 255.0
    arr_uint8 = (arr * 255).astype(np.uint8)
    img_rgb = Image.fromarray(arr_uint8).convert('RGB')
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    return t(img_rgb).unsqueeze(0).to(device)

# ── Strategy of the 5 tests ───────────────────────────────────────────────
# Test 1: uniform black images  → flat heatmap expected → low score
# Test 2: uniform white images  → same result as black (no asymmetry)
# Test 3: random noise          → must produce a score different from 1 and 2
# Test 4: inspection of 'other' → checks the 8×8 heatmap structure
# Test 5: variance over 5 runs → the model is sensitive to inputs
# ──────────────────────────────────────────────────────────────────────────
# ── Test 1 : 4 images IDENTIQUES (dummy test) ──
print("=" * 50)
print("TEST 1 : 4 images identiques (dummy noir)")
dummy = Image.new('RGB', (512, 512), color=0)
t = preprocess_rsna(dummy)
with torch.no_grad():
    out, other = asym_model(t, t, t, t)
probs = F.softmax(out, dim=1)[0].cpu().numpy()
print(f"  probs : {np.round(probs, 6)}")
print(f"  output brut : {out.cpu().numpy()}")

# ── Test 2 : 4 images IDENTIQUES (dummy blanc) ──
print("\nTEST 2 : 4 images identiques (dummy blanc)")
dummy2 = Image.new('RGB', (512, 512), color=255)
t2 = preprocess_rsna(dummy2)
with torch.no_grad():
    out2, _ = asym_model(t2, t2, t2, t2)
probs2 = F.softmax(out2, dim=1)[0].cpu().numpy()
print(f"  probs : {np.round(probs2, 6)}")
print(f"  output brut : {out2.cpu().numpy()}")

# ── Test 3: random images (noise) ──
print("\nTEST 3 : images bruit aleatoire")
import torchvision.transforms.functional as TF
rand1 = torch.randn(1, 3, 256, 256).to(device)
rand2 = torch.randn(1, 3, 256, 256).to(device)
rand3 = torch.randn(1, 3, 256, 256).to(device)
rand4 = torch.randn(1, 3, 256, 256).to(device)
with torch.no_grad():
    out3, other3 = asym_model(rand1, rand2, rand3, rand4)
probs3 = F.softmax(out3, dim=1)[0].cpu().numpy()
print(f"  probs : {np.round(probs3, 6)}")
print(f"  output brut : {out3.cpu().numpy()}")

# ── Test 4 : inspect other ──
print("\nTEST 4 : contenu de 'other'")
print(f"  type(other) : {type(other3)}")
print(f"  len(other)  : {len(other3)}")
for i, o in enumerate(other3):
    print(f"  other[{i}] type : {type(o)}")
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, torch.Tensor):
                print(f"    '{k}' → shape={v.shape}, value={v.cpu().tolist()}")
            else:
                print(f"    '{k}' → {v}")

# ── Test 5: variance over 5 runs of different noise ──
print("\nTEST 5 : variance over 5 different random inputs")
scores = []
for i in range(5):
    r1 = torch.randn(1, 3, 256, 256).to(device)
    r2 = torch.randn(1, 3, 256, 256).to(device)
    r3 = torch.randn(1, 3, 256, 256).to(device)
    r4 = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        o, _ = asym_model(r1, r2, r3, r4)
    p = F.softmax(o, dim=1)[0].cpu().numpy()
    scores.append(p[1])
    print(f"  Run {i+1} : probs[1] = {p[1]:.6f}")
print(f"  Variance : {np.var(scores):.8f}")
print(f"  Min={min(scores):.4f}  Max={max(scores):.4f}  Range={max(scores)-min(scores):.4f}")

# %% cellule 13
# ═══════════════════════════════════════════════════════════════════════════
# WEIGHTS DIAGNOSTIC CELL — Inspection of the asymmirai_weights.pth file
#
# Goal: check that the weights file is valid and understand its structure
# avant de l'utiliser. Deux cas de figure possibles :
#   Case 1: checkpoint is a dict with a 'state_dict' key
#           → format standard PyTorch Lightning / torch.save(state_dict)
#   Case 2: checkpoint is directly the complete model object
#           → torch.load() returns an instance of LocalizedDifModel
#             with its .state_dict() method directly accessible
# We display the first and last 5 layers to check the architecture
# (ResNet backbone + differential localization head).
# ═══════════════════════════════════════════════════════════════════════════

# Weights diagnostic cell
import torch
import os

weights_path = '/content/snapshots/asymmirai_weights.pth'
print("File size:", os.path.getsize(weights_path) / 1e6, "MB")

checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
print("Type :", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Available keys:", list(checkpoint.keys())[:10])
    if 'state_dict' in checkpoint:
        keys = list(checkpoint['state_dict'].keys())
        print("Number of parameters:", len(keys))
        print("Premiers :", keys[:5])
        print("Derniers :", keys[-5:])
elif hasattr(checkpoint, 'state_dict'):
    keys = list(checkpoint.state_dict().keys())
    print("Number of parameters:", len(keys))
    print("Premiers :", keys[:5])
    print("Derniers :", keys[-5:])

# %% cellule 14
# ═══════════════════════════════════════════════════════════════════════════
# CELLULE 5 — BLOC 2 : INTERFACE GRADIO MULTIMODALE + GRAPHIQUES + HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════

import gradio as gr
import matplotlib.pyplot as plt
import io


# ── generate_risk_chart ────────────────────────────────────────────────────
# Generates a 2-panel matplotlib chart (dark dashboard-style background):
#   Left panel  : barplot of the AsymMirai risk for each year (1→5 years)
#                 Red if risk > 35%, orange if > 20%, green otherwise
#   Right panel : imaging vs clinical contribution + final score
#                 Dotted horizontal lines = clinical thresholds
# The result is rendered as a PIL.Image through a BytesIO buffer (no file on disk).
def generate_risk_chart(risk_vector, score_final, score_imagerie, score_clinique):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#1a1a2e')

    # ── Chart 1: risk per year ──
    ax1 = axes[0]
    ax1.set_facecolor('#16213e')
    annees  = [f'An {i+1}' for i in range(len(risk_vector[:5]))]
    valeurs = [v * 100 for v in risk_vector[:5]]
    colors  = ['#e74c3c' if v > 35 else '#f39c12' if v > 20 else '#2ecc71' for v in valeurs]
    bars = ax1.bar(annees, valeurs, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_title('Risque AsymMirai par annee (%)', color='white', fontsize=12, pad=10)
    ax1.set_ylabel('Probabilite (%)', color='white')
    ax1.tick_params(colors='white')
    ax1.spines[:].set_color('#444')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars, valeurs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=9)

    # ── Chart 2: imaging vs clinical contributions ──
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    categories = ['Score\nImagerie', 'Score\nClinique', 'Score\nFinal']
    valeurs2   = [score_imagerie * 100, score_clinique * 100, score_final * 100]
    colors2    = [
        '#3498db',
        '#9b59b6',
        '#e74c3c' if score_final > 0.35 else '#f39c12' if score_final > 0.20 else '#2ecc71'
    ]
    bars2 = ax2.bar(categories, valeurs2, color=colors2, edgecolor='white', linewidth=0.5)
    ax2.set_title('Contributions au score final (%)', color='white', fontsize=12, pad=10)
    ax2.set_ylabel('Score (%)', color='white')
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#444')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=35, color='red',    linestyle='--', alpha=0.5, label='Seuil tres eleve')
    ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Seuil eleve')
    ax2.axhline(y=10, color='green',  linestyle='--', alpha=0.5, label='Seuil modere')
    ax2.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
    for bar, val in zip(bars2, valeurs2):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()



# ── generate_asymmetry_heatmap ─────────────────────────────────────────────
# Overlays the model's 8×8 heatmap on the original mammographic image:
#  1. Min-max normalization of the raw heatmap into [0, 1]
#  2. Bilinear upscaling 8×8 → original image size
#  3. Applying the 'jet' colormap (blue=low, red=high)
#  4. Alpha channel at 60% to make the overlay semi-transparent
#  5. alpha_composite() to merge mammogram (RGBA) + heatmap (RGBA)
def generate_asymmetry_heatmap(img, heatmap_8x8):
    W, H = img.size

    h = heatmap_8x8.astype(np.float32)
    h_min, h_max = h.min(), h.max()
    if h_max > h_min:
        h = (h - h_min) / (h_max - h_min)
    else:
        h = np.zeros_like(h)

    heatmap_pil = Image.fromarray((h * 255).astype(np.uint8)).resize(
        (W, H), resample=Image.BILINEAR
    )
    heatmap_arr = np.array(heatmap_pil)

    heatmap_rgba = (cm.get_cmap('jet')(heatmap_arr / 255.0) * 255).astype(np.uint8)
    heatmap_rgba[:, :, 3] = (heatmap_arr * 0.60).astype(np.uint8)

    composite = Image.alpha_composite(
        img.convert('RGBA'),
        Image.fromarray(heatmap_rgba, mode='RGBA')
    ).convert('RGB')
    return composite



# ── analyze_multimodal ─────────────────────────────────────────────────────
# Main function called by the Gradio button. Orchestrates:
#  1. Checking the prerequisites (model loaded + 4 images provided)
#  2. get_asymmirai_risk_vector(): model inference → risk vector + meta heatmaps
#  3. encode_clinical_features()  : normalization of the patient data
#  4. ensemble.predict_risk()     : imaging + clinical fusion → final score
#  5. Generation of the formatted text report (ASCII) + chart + heatmaps
# The 4 outputs correspond to the 4 Gradio components of the cell.
def analyze_multimodal(img_lcc, img_rcc, img_lmlo, img_rmlo,
                        age, imc, antecedents, densite_birads):
    if asym_model is None:
        return 'Modele AsymMirai non charge.', None, None, None
    if None in [img_lcc, img_rcc, img_lmlo, img_rmlo]:
        return 'Veuillez uploader les 4 vues (L-CC, R-CC, L-MLO, R-MLO).', None, None, None

    try:
        dev = next(asym_model.parameters()).device
    except StopIteration:
        dev = torch.device('cpu')

    try:
        risk_vector, asym_meta = get_asymmirai_risk_vector(
            asym_model, img_lcc, img_rcc, img_lmlo, img_rmlo, dev)

        clinical = encode_clinical_features(
            float(age), float(imc),
            antecedents == 'Oui', str(densite_birads)
        )

        result = ensemble.predict_risk(risk_vector, clinical)

        chart = generate_risk_chart(
            risk_vector,
            result['score_final'],
            result['score_imagerie'],
            result['score_clinique']
        )

        heatmap_cc  = generate_asymmetry_heatmap(img_lcc,  asym_meta['heatmap_cc'])
        heatmap_mlo = generate_asymmetry_heatmap(img_lmlo, asym_meta['heatmap_mlo'])

        timeline = '\n'.join([
            f'  Risque a {i+1} an(s) : {v:.2%}'
            for i, v in enumerate(risk_vector[:5])
        ])

        rapport = f"""╔══════════════════════════════════════════════════════╗
║      RAPPORT D'ANALYSE — AsymMirai + Clinique         ║
╚══════════════════════════════════════════════════════╝

▶ DONNEES PATIENTE
  Age              : {int(age)} ans
  IMC              : {float(imc):.1f} kg/m2
  Antecedents fam. : {'Oui' if antecedents == 'Oui' else 'Non'}
  Densite BI-RADS  : {densite_birads}

▶ SCORES ASYMMIRAI (imagerie seule)
{timeline}

▶ SCORE DE RISQUE COMBINE
  Score final      : {result['score_final']:.2%}
  Score imagerie   : {result['score_imagerie']:.2%}
  Score clinique   : {result['score_clinique']:.2%}
  Niveau           : {result['niveau_risque']}

  Contributions :
    Imagerie       : {result['contributions']['imagerie']}
    Clinique       : {result['contributions']['clinique']}

▶ LOCALISATION ASYMETRIE (espace 256x256)
  Vue CC   : X={asym_meta['cc_x']:.0f}, Y={asym_meta['cc_y']:.0f}
  Vue MLO  : X={asym_meta['mlo_x']:.0f}, Y={asym_meta['mlo_y']:.0f}

──────────────────────────────────────────────────────
  Usage recherche uniquement.
  Ne remplace pas l'avis d'un radiologue qualifie.
──────────────────────────────────────────────────────"""

        return rapport, chart, heatmap_cc, heatmap_mlo

    except Exception as e:
        import traceback
        return f'Erreur : {e}\n\n{traceback.format_exc()}', None, None, None


# ── Construction de l'interface Gradio ─────────────────────────────────────
# gr.Blocks + Soft theme: two tabs (images / clinical) to structure
# the inputs without overloading the interface.
# analyze_btn.click() wires the button to the 4 inputs and 4 outputs defined.
# interface.launch(share=True) generates a public .gradio.live link (1 week).
# ── Interface Gradio ──
with gr.Blocks(title='AsymMirai Multimodal', theme=gr.themes.Soft()) as interface:

    gr.Markdown('# AsymMirai Multimodal — Analyse du Risque de Cancer du Sein')
    gr.Markdown('> Usage recherche uniquement. Utiliser le lien .gradio.live')

    with gr.Tab('Images Mammographiques'):
        with gr.Row():
            img_lcc_in  = gr.Image(type='pil', label='1. Vue L-CC (Gauche - Face)')
            img_rcc_in  = gr.Image(type='pil', label='2. Vue R-CC (Droit - Face)')
        with gr.Row():
            img_lmlo_in = gr.Image(type='pil', label='3. Vue L-MLO (Gauche - Oblique)')
            img_rmlo_in = gr.Image(type='pil', label='4. Vue R-MLO (Droit - Oblique)')

    with gr.Tab('Donnees Cliniques'):
        with gr.Row():
            age_in  = gr.Number(label='Age (annees)', value=50, minimum=20, maximum=90)
            imc_in  = gr.Number(label='IMC (kg/m2)',  value=24.0, minimum=15.0, maximum=50.0)
        with gr.Row():
            atcd_in   = gr.Radio(['Non', 'Oui'], value='Non',
                                  label='Antecedents familiaux (1er degre)')
            birads_in = gr.Dropdown(['A', 'B', 'C', 'D'], value='B',
                                     label='Densite BI-RADS (ACR)')
        gr.Markdown('A = adipeux | B = fibro-glandulaire | C = heterogene | D = tres dense')

    analyze_btn = gr.Button('Lancer Analyse Multimodale', variant='primary', size='lg')

    rapport_out     = gr.Textbox(label='Rapport Complet', lines=25, max_lines=35)
    chart_out       = gr.Image(label='Graphique de Risque', type='pil')
    with gr.Row():
        heatmap_cc_out  = gr.Image(label='Heatmap CC',  type='pil')
        heatmap_mlo_out = gr.Image(label='Heatmap MLO', type='pil')
    gr.Markdown('Rouge/jaune = zone asymetrie maximale detectee par le modele')

    analyze_btn.click(
        fn=analyze_multimodal,
        inputs=[img_lcc_in, img_rcc_in, img_lmlo_in, img_rmlo_in,
                age_in, imc_in, atcd_in, birads_in],
        outputs=[rapport_out, chart_out, heatmap_cc_out, heatmap_mlo_out]
    )

interface.launch(debug=True, share=True)

# %% [markdown] cellule 15
# ## 📊 Cell 6 — Block 3: Batch Testing on a Cohort
# **Expected folder structure:**
# ```
# /content/cohort/
# ├── clinical_data.csv          ← patient_id, age, imc, antecedents, densite_birads, event_5y, time_to_event
# └── images/
#     ├── patient_001/
#     │   ├── LCC.png  ├── RCC.png  ├── LMLO.png  └── RMLO.png
#     └── patient_002/ ...
# ```
# The cell also includes a **synthetic cohort generator** to test without real data.

# %% cellule 16
# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — BLOCK 3: COHORT TESTING (BATCH TESTING)
#
# Walks through a folder of patients, generates all the predictions at once
# and saves the results as timestamped CSV + JSON.
#
# Includes generate_synthetic_cohort() to test the pipeline without real data.
# ═══════════════════════════════════════════════════════════════════════════


# ── generate_synthetic_cohort ──────────────────────────────────────────────
# Generates a reproducible fictitious cohort (fixed seed) to validate the
# batch pipeline WITHOUT real data. Structure created:
#   /content/cohort/images/patient_NNN/{LCC,RCC,LMLO,RMLO}.png  (bruit gaussien)
#   /content/cohort/clinical_data.csv  (age, imc, antecedents, birads, event_5y)
# Realistic simulated distributions: age N(52,10), BMI N(25,4),
# 15% family history, ~12% 5-year cancer incidence.
def generate_synthetic_cohort(output_dir: str = '/content/cohort',
                               n_patients: int = 20, seed: int = 42):
    """
    Génère une cohorte synthétique pour valider le pipeline.
    Crée des images grises aléatoires (256×256) + un CSV clinique réaliste.

    Paramètres
    ----------
    output_dir : chemin de sortie de la cohorte
    n_patients : nombre de patientes à générer
    seed       : graine pour la reproductibilité
    """
    random.seed(seed)
    np.random.seed(seed)
    images_dir = Path(output_dir) / 'images'
    os.makedirs(images_dir, exist_ok=True)
    rows = []

    for i in range(1, n_patients + 1):
        pid     = f'patient_{i:03d}'
        pid_dir = images_dir / pid
        pid_dir.mkdir(exist_ok=True)

        # Gray images with Gaussian noise (simulates a mammogram)
        for view in ['LCC', 'RCC', 'LMLO', 'RMLO']:
            noise = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(noise).save(pid_dir / f'{view}.png')

        # Simulated clinical data with realistic distributions
        age   = int(np.clip(np.random.normal(52, 10), 20, 85))
        imc   = round(float(np.clip(np.random.normal(25, 4), 16, 45)), 1)
        atcd  = int(random.random() < 0.15)   # 15% prevalence
        den   = random.choice(['A', 'B', 'B', 'C', 'C', 'D'])
        t_ev  = round(float(np.random.uniform(0.5, 5.0)), 2)
        ev    = int(t_ev < 5.0 and random.random() < 0.12)  # ~12% d'incidence

        rows.append({
            'patient_id': pid, 'age': age, 'imc': imc, 'antecedents': atcd,
            'densite_birads': den, 'event_5y': ev,
            'time_to_event': t_ev if ev else 5.0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / 'clinical_data.csv', index=False)
    print(f'✅ Synthetic cohort : {n_patients} patients → {output_dir}')
    print(f'   Event rate : {df["event_5y"].mean():.1%}')
    return df



# ── run_cohort_batch ───────────────────────────────────────────────────────
# Iterates over all the patients of the folder and applies the full pipeline:
#   For each patient: loading the 4 views → AsymMirai inference
#                     → clinical encoding → predict_risk() → storage
# Robustness: exceptions caught per patient (no global stop)
# and partial save every 10 patients (avoids loss on crash).
# Output: timestamped CSV + JSON in output_dir for traceability.
def run_cohort_batch(cohort_dir:       str,
                     asym_model,
                     ensemble:         EnsembleModel,
                     csv_filename:     str  = 'clinical_data.csv',
                     images_subdir:    str  = 'images',
                     output_dir:       str  = '/content/cohort_results',
                     max_patients:     int  = None) -> pd.DataFrame:
    """
    Pipeline de prédiction par cohorte complète.
    Parcourt toutes les patientes du dossier et génère les prédictions en batch.

    Paramètres
    ----------
    cohort_dir    : racine de la cohorte (doit contenir clinical_data.csv + images/)
    asym_model    : modèle AsymMirai PyTorch chargé en cellule 3
    ensemble      : instance EnsembleModel initialisée en cellule 4
    csv_filename  : nom du fichier CSV des métadonnées cliniques
    images_subdir : sous-dossier contenant les images par patient
    output_dir    : dossier de sauvegarde des résultats
    max_patients  : limite du nombre de patients (None = tous)

    Retourne
    --------
    pd.DataFrame avec une ligne par patiente et toutes ses prédictions
    """
    cohort_path = Path(cohort_dir)
    df_clinical = pd.read_csv(cohort_path / csv_filename)
    df_clinical['patient_id'] = df_clinical['patient_id'].astype(str)
    df_clinical.set_index('patient_id', inplace=True)
    if max_patients:
        df_clinical = df_clinical.head(max_patients)

    os.makedirs(output_dir, exist_ok=True)
    try:
        dev = next(asym_model.parameters()).device
    except StopIteration:
        dev = torch.device('cpu')

    print(f'\n{"="*60}')
    print(f'  BATCH TEST — COHORTE AsymMirai Multimodal')
    print(f'  👥 Patientes  : {len(df_clinical)} | 🖥️  Device : {dev}')
    print(f'  🕐 Start      : {datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*60}\n')

    results = []

    for patient_id, row in tqdm(df_clinical.iterrows(),
                                 total=len(df_clinical),
                                 desc='🔍 Analyse cohorte', unit='pat.'):
        base = {
            'patient_id':    patient_id,
            'event_5y':      int(row.get('event_5y', -1)),
            'time_to_event': float(row.get('time_to_event', 5.0)),
        }

        # Loading the 4 views of the patient
        pid_dir = cohort_path / images_subdir / patient_id
        views   = {}
        for v in ['LCC', 'RCC', 'LMLO', 'RMLO']:
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                f = pid_dir / f'{v}{ext}'
                if f.exists():
                    try: views[v] = Image.open(f).convert('RGB')
                    except: pass
                    break

        if len(views) < 4:
            results.append({**base, 'status': 'ERREUR: vues manquantes',
                            'score_ensemble': np.nan,
                            **{f'score_y{k}': np.nan for k in range(1, 6)}})
            continue

        try:
            # AsymMirai inference
            rv, meta = get_asymmirai_risk_vector(
                asym_model,
                views['LCC'], views['RCC'], views['LMLO'], views['RMLO'], dev
            )
            # Clinical encoding
            clin = encode_clinical_features(
                float(row['age']), float(row['imc']),
                bool(int(row['antecedents'])), str(row['densite_birads'])
            )
            # Fused prediction
            res = ensemble.predict_risk(rv, clin)

            results.append({
                **base, 'status': 'OK',
                'age': float(row['age']), 'imc': float(row['imc']),
                'antecedents': int(row['antecedents']),
                'densite_birads': str(row['densite_birads']),
                'score_ensemble': res['score_final'],
                'niveau_risque':  res['niveau_risque'],
                **{f'score_y{i+1}': round(float(rv[i]), 5)
                   for i in range(min(5, len(rv)))},
                'asym_cc_x':  meta['cc_x'],  'asym_cc_y':  meta['cc_y'],
                'asym_mlo_x': meta['mlo_x'], 'asym_mlo_y': meta['mlo_y'],
            })
        except Exception as e:
            results.append({**base, 'status': f'ERREUR: {e}',
                            'score_ensemble': np.nan,
                            **{f'score_y{k}': np.nan for k in range(1, 6)}})

        # Intermediate save every 10 patients
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(f'{output_dir}/resultats_partiels.csv', index=False)

    df_res = pd.DataFrame(results)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_res.to_csv(f'{output_dir}/predictions_{ts}.csv', index=False)
    df_res.to_json(f'{output_dir}/predictions_{ts}.json', orient='records', indent=2)

    n_ok  = (df_res['status'] == 'OK').sum()
    print(f'\n✅ Batch finished — {n_ok}/{len(df_res)} successes')
    print(f'📄 Results → {output_dir}/predictions_{ts}.csv')
    return df_res


# ── EXECUTION: generate a synthetic cohort and run the batch ──
# To use your real data, replace cohort_dir with your folder.

# ── EXECUTION ──────────────────────────────────────────────────────────────
# Step 1: generation of 20 synthetic patients
# Step 2: full batch testing on these 20 patients
# To use your real data, replace cohort_dir = '/your/folder/'
# and uncomment max_patients to limit the processing (partial test).
generate_synthetic_cohort('/content/cohort', n_patients=20)

df_resultats = run_cohort_batch(
    cohort_dir  = '/content/cohort',
    asym_model  = asym_model,
    ensemble    = ensemble,
    output_dir  = '/content/cohort_results',
    # max_patients = 100   # uncomment to limit
)

print(f'\n📋 Preview of the results :')
df_resultats[['patient_id', 'event_5y', 'time_to_event',
               'score_y1', 'score_y5', 'score_ensemble', 'niveau_risque']].head(10)

# %% [markdown] cellule 17
# ## 📈 Cell 7 — Block 4: Clinical Survival Metrics
# **Why not Accuracy?** It ignores the time to event and treats censored patients incorrectly.
# 
# | Metric | Library | Role |
# |---|---|---|
# | **C-Index** | `lifelines` | Temporal discrimination (AUC analogue for survival data) |
# | **AUC-ROC at 1, 3, 5 years** | `scikit-survival` | Discriminate cases/controls at each horizon |
# | **Brier Score** | `sklearn` | Calibration of the predicted probabilities |

# %% cellule 18
# ═══════════════════════════════════════════════════════════════════════════
# CELL 7 — BLOCK 4: CLINICAL SURVIVAL EVALUATION METRICS
#
# Evaluation with metrics suited to survival data:
#   → C-Index de Harrell  (lifelines)       — discrimination temporelle
#   → Time-dependent AUC-ROC  (scikit-survival) — AUC per horizon (1, 3, 5 years)
#   → Brier Score             (sklearn)         — calibration of the probabilities
#   → Visualisations      (matplotlib)      — distributions + courbes ROC
#
# IMPORTANT: Standard Accuracy is INAPPROPRIATE for survival models
# because it ignores censored data (patients lost to follow-up before 5 years).
# ═══════════════════════════════════════════════════════════════════════════

# Optional imports (with handling of missing packages)
try:
    from lifelines.utils import concordance_index
    LIFELINES_OK = True
except ImportError:
    LIFELINES_OK = False
    print('⚠️ lifelines missing → install with : !pip install lifelines -q')

try:
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv
    SKSURV_OK = True
except ImportError:
    SKSURV_OK = False
    print('ℹ️ scikit-survival missing → naive sklearn AUC used as fallback')


# ── compute_cindex ─────────────────────────────────────────────────────────
# Harrell's C-Index measures the temporal discrimination of the model:
#   For all pairs (i,j) with event(i)=1 and time(i)<time(j),
#   the model is correct if score(i) > score(j).
# We invert the score (-score) because the lifelines convention expects
# a high score → long survival, whereas here a high score → high risk.
# Reference: 0.5=random, 0.7=acceptable, 0.8=good, >0.85=excellent
def compute_cindex(df: pd.DataFrame,
                   score_col: str = 'score_ensemble',
                   time_col:  str = 'time_to_event',
                   event_col: str = 'event_5y') -> dict:
    """
    Calcule le C-Index de Harrell (Concordance Index).

    Interprétation :
      0.5 = modèle aléatoire | 0.7 = acceptable | 0.8 = bon | >0.85 = excellent

    Note : on passe -score car un score élevé prédit un événement précoce
    (convention lifelines : score élevé = survie longue).
    """
    df_c = df.dropna(subset=[score_col, time_col, event_col]).copy()
    if not LIFELINES_OK or len(df_c) < 5:
        return {'c_index': np.nan}
    try:
        ci = concordance_index(
            df_c[time_col].values,
            -df_c[score_col].values,       # inverted: high score → early event
            df_c[event_col].values.astype(int)
        )
        print(f'  🎯 C-Index (Harrell) : {ci:.4f}')
        return {'c_index': round(ci, 4)}
    except Exception as e:
        print(f'  ⚠️ C-Index error : {e}')
        return {'c_index': np.nan}



# ── compute_temporal_auc ───────────────────────────────────────────────────
# Computes the AUC-ROC for each time horizon (1, 3, 5 years).
# The difficulty with survival data: patients censored before
# horizon t have an unknown status → we exclude them rather than label them.
# Method 1 (sksurv): cumulative_dynamic_AUC, rigorous for censored data
# Method 2 (sklearn): naive AUC on a subset of clearly defined cases/controls
def compute_temporal_auc(df: pd.DataFrame,
                          time_col:  str  = 'time_to_event',
                          event_col: str  = 'event_5y',
                          horizons:  list = [1, 3, 5]) -> dict:
    """
    Calcule l'AUC-ROC à chaque horizon temporel.

    Pour l'horizon t :
      - Cas positif  : event=1 ET time <= t
      - Cas négatif  : event=0 ET time >= t (censure après t)
      - Exclu        : censuré avant t (statut inconnu à t)

    Méthode 1 (recommandée) : cumulative_dynamic_AUC de scikit-survival
    Méthode 2 (fallback)    : classification binaire naïve sklearn
    """
    results     = {}
    score_map   = {1: 'score_y1', 3: 'score_y3', 5: 'score_y5'}

    for t in horizons:
        col = score_map.get(t, 'score_ensemble')
        if col not in df.columns:
            col = 'score_ensemble'

        # Tentative scikit-survival (AUC temporelle rigoureuse)
        if SKSURV_OK:
            try:
                df_c = df.dropna(subset=[col]).copy()
                y_s  = Surv.from_arrays(
                    df_c[event_col].values.astype(bool),
                    df_c[time_col].values.astype(float)
                )
                auc_arr, _ = cumulative_dynamic_auc(
                    y_s, y_s,
                    df_c[col].values.astype(float),
                    np.array([t - 0.001])   # horizon slightly before t
                )
                results[f'auc_{t}y'] = round(float(auc_arr[0]), 4)
                print(f'  📊 AUC at {t} year(s) [sksurv] : {auc_arr[0]:.4f}')
                continue
            except Exception as e:
                print(f'  ⚠️ sksurv {t}y : {e} → repli sklearn')

        # Fallback sklearn (AUC naïve, moins rigoureuse mais toujours utile)
        mask_pos = (df[event_col] == 1) & (df[time_col] <= t)
        mask_neg = df[time_col] >= t
        df_sub   = df[mask_pos | mask_neg].copy()
        df_sub['_lbl'] = mask_pos[df_sub.index].astype(int)
        df_sub   = df_sub.dropna(subset=[col])

        if df_sub['_lbl'].nunique() < 2 or len(df_sub) < 5:
            results[f'auc_{t}y'] = np.nan
            print(f'  ⚠️ AUC {t}y : insufficient data')
            continue

        auc = roc_auc_score(df_sub['_lbl'], df_sub[col])
        results[f'auc_{t}y'] = round(auc, 4)
        print(f'  📊 AUC at {t} year(s) [naive sklearn] : {auc:.4f}')

    return results



# ── generate_evaluation_report ─────────────────────────────────────────────
# Complete evaluation report which orchestrates:
#   C-Index, AUC@1/3/5 years, Brier Score, descriptive stats
# + 2 figures matplotlib :
#   Figure 1: histogram + boxplot of the scores (cases vs controls)
#   Figure 2: ROC curves at the 3 horizons with random baseline
# Everything is saved as timestamped JSON for experiment reproducibility.
def generate_evaluation_report(df_results: pd.DataFrame,
                                 output_dir: str  = '/content/cohort_results',
                                 plot:       bool = True) -> dict:
    """
    Rapport d'évaluation clinique complet à partir du DataFrame du Bloc 3.
    Calcule C-Index, AUC temporelle, Brier Score, stats descriptives
    et génère les graphiques de distribution et courbes ROC.

    Paramètres
    ----------
    df_results : DataFrame issu de run_cohort_batch()
    output_dir : dossier de sauvegarde des graphiques et du rapport JSON
    plot       : si True, génère et affiche les figures

    Retourne : dict synthétique de toutes les métriques
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filtering on valid predictions only
    df_v    = df_results[df_results['status'] == 'OK'].copy()
    n_total = len(df_results)
    n_valid = len(df_v)
    n_ev    = int(df_v['event_5y'].sum()) if 'event_5y' in df_v.columns else 0

    print(f'\n{"="*60}')
    print(f'  CLINICAL EVALUATION REPORT')
    print(f'{"="*60}')
    print(f'  Patients analyzed : {n_valid}/{n_total}')
    print(f'  Events (cancer) : {n_ev} ({n_ev/max(n_valid,1):.1%})')

    if n_valid < 5:
        print('  ⚠️ Too little valid data for the evaluation.')
        return {}

    report = {'n_total': n_total, 'n_valid': n_valid, 'n_events': n_ev,
               'event_rate': round(n_ev / max(n_valid, 1), 4)}

    # ── C-Index ──
    print('\n─── C-Index (Harrell) ───')
    report.update(compute_cindex(df_v))

    # ── AUC temporelle ──
    print('\n─── AUC-ROC Temporelle ───')
    report.update(compute_temporal_auc(df_v))

    # ── Brier Score (calibration) ──
    print('\n─── Brier Score (Calibration) ───')
    df_c = df_v.dropna(subset=['score_ensemble', 'event_5y'])
    if len(df_c) >= 5:
        bs = brier_score_loss(df_c['event_5y'].astype(int), df_c['score_ensemble'])
        report['brier_score'] = round(bs, 4)
        print(f'  📐 Brier Score : {bs:.4f}  (0=perfect, 0.25=random)')

    # ── Descriptive statistics of the scores ──
    print('\n─── Ensemble Score Statistics ───')
    sc = df_v['score_ensemble'].dropna()
    stats = {'score_mean': round(float(sc.mean()), 4),
              'score_std':  round(float(sc.std()),  4),
              'score_p50':  round(float(sc.median()), 4),
              'score_p90':  round(float(np.percentile(sc, 90)), 4)}
    report.update(stats)
    for k, v in stats.items():
        print(f'  {k}: {v}')

    # ── Final summary ──
    print(f'\n{"="*60}')
    print(f'  METRICS SUMMARY')
    print(f'{"="*60}')
    for k, label in [('c_index','C-Index'), ('auc_1y','AUC à 1 an'),
                      ('auc_3y','AUC à 3 ans'), ('auc_5y','AUC à 5 ans'),
                      ('brier_score','Brier Score')]:
        v = report.get(k, 'N/A')
        print(f'  {label:<18} : {v}')
    print(f'{"="*60}')

    # ── Visualisations ──
    if plot and n_valid >= 10:

        # Figure 1: Score distribution + boxplot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Distribution des Scores de Risque — AsymMirai Multimodal',
                     fontsize=13, fontweight='bold')

        cases = df_v[df_v['event_5y']==1]['score_ensemble'].dropna()
        ctrls = df_v[df_v['event_5y']==0]['score_ensemble'].dropna()

        axes[0].hist(ctrls, bins=12, alpha=0.6, color='#2196F3', density=True,
                     label=f'Contrôles (n={len(ctrls)})')
        axes[0].hist(cases, bins=12, alpha=0.6, color='#F44336', density=True,
                     label=f'Cas cancer (n={len(cases)})')
        axes[0].set_xlabel('Score de Risque Ensemble')
        axes[0].set_ylabel('Densité')
        axes[0].set_title('Distribution par Groupe')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        bp = axes[1].boxplot([ctrls.values, cases.values],
                             labels=['Contrôles\n(event=0)', 'Cas Cancer\n(event=1)'],
                             patch_artist=True, notch=True)
        bp['boxes'][0].set_facecolor('#BBDEFB')
        bp['boxes'][1].set_facecolor('#FFCDD2')
        axes[1].set_ylabel('Score de Risque Ensemble')
        axes[1].set_title('Comparaison par Groupe')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/score_distribution.png', dpi=120, bbox_inches='tight')
        plt.show()
        print(f'  📊 Figure saved → {output_dir}/score_distribution.png')

        # Figure 2 : Courbes ROC aux 3 horizons temporels
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.plot([0,1],[0,1],'k--',lw=1,label='Aléatoire (AUC=0.50)')

        colors  = {1:'#2196F3', 3:'#FF9800', 5:'#F44336'}
        sc_map  = {1:'score_y1', 3:'score_y3', 5:'score_y5'}

        for t in [1, 3, 5]:
            col  = sc_map.get(t, 'score_ensemble')
            if col not in df_v.columns: col = 'score_ensemble'
            m_p  = (df_v['event_5y']==1) & (df_v['time_to_event']<=t)
            m_n  = df_v['time_to_event'] >= t
            sub  = df_v[m_p|m_n].copy()
            sub['lbl'] = m_p[sub.index].astype(int)
            sub  = sub.dropna(subset=[col])
            if sub['lbl'].nunique() < 2 or len(sub) < 5: continue
            fpr, tpr, _ = roc_curve(sub['lbl'], sub[col])
            roc_a = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[t], lw=2,
                    label=f'AUC à {t} an(s) = {roc_a:.3f}')

        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title('Courbes ROC — AsymMirai Multimodal (Horizons Temporels)')
        ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curves.png', dpi=120, bbox_inches='tight')
        plt.show()
        print(f'  📊 Figure saved → {output_dir}/roc_curves.png')

    # ── Saving the JSON report ──
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'{output_dir}/rapport_evaluation_{ts}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\n  📄 JSON report saved → {report_path}')

    return report


# ── EXECUTION: evaluation on the Block 3 results ──
# df_resultats must have been generated in cell 6.
# If you evaluate on synthetic data, the metrics are indicative only.

rapport_eval = generate_evaluation_report(
    df_resultats,
    output_dir = '/content/cohort_results',
    plot       = True
)

print('\n✅ Evaluation finished — see /content/cohort_results/ for the generated files.')

# %% cellule 19
# ═══════════════════════════════════════════════════════════════════════════
# CELL 8 — SAVING TO GOOGLE DRIVE
#
# Goal: persist the important artifacts between two Colab sessions.
# Without this cell, the files in /content/ are lost at the end
# of the session (ephemeral environment).
#
# Files saved:
#   asymmirai_weights.pth     → AsymMirai model weights (44 MB)
#   ensemble_meta_model.pkl   → trained XGBoost meta-model (if available)
#
# drive.mount() triggers a Google OAuth authentication in Colab.
# shutil.copy() performs a byte-for-byte copy (no modification).
# ═══════════════════════════════════════════════════════════════════════════

# Saving the important files to Google Drive
from google.colab import drive
drive.mount('/content/drive')

import shutil, os

# Creates a dedicated folder on Drive
save_dir = '/content/drive/MyDrive/AsymMirai_Projet'
os.makedirs(save_dir, exist_ok=True)

# Saves the weights
shutil.copy('/content/snapshots/asymmirai_weights.pth',
            f'{save_dir}/asymmirai_weights.pth')

# Saves the XGBoost meta-model if it exists
if os.path.exists('/content/ensemble_meta_model.pkl'):
    shutil.copy('/content/ensemble_meta_model.pkl',
                f'{save_dir}/ensemble_meta_model.pkl')

print('Saved in :', save_dir)

# %% [markdown] cellule 20
# 
