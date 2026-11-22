# Script conversion of the cells of: modele détection 1.ipynb (Drive, POC/DEEP LEARNING/DÉTECTION/Détection 1)
# Outputs and images removed.

# %% [markdown] cellule 1
# # Breast cancer detection — MammoScreen
# 
# Mammography analysis with the models `ianpan/mammo-crop` and `ianpan/mammoscreen`.

# %% cellule 2
!pip install dicomsdl -q
!pip install transformers==4.41.0 -q
!pip install pylibjpeg pylibjpeg-libjpeg gdcm -q


# %% cellule 3
import cv2
import torch
import dicomsdl
import numpy as np
from transformers import AutoModel

# %% [markdown] cellule 4
# ## 1. Choose GPU or CPU

# %% cellule 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation : {device}")

# %% [markdown] cellule 6
# ## 2. Load the models from HuggingFace

# %% cellule 7
print("Downloading the models... (first time only)")

crop_model = AutoModel.from_pretrained("ianpan/mammo-crop", trust_remote_code=True)
crop_model = crop_model.eval().to(device)

model = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True)
model = model.eval().to(device)

# %% [markdown] cellule 8
# ## 3. Fonctions utilitaires

# %% cellule 9
def crop_mammo(img, crop_model, device):
    img_shape = torch.tensor([img.shape[:2]]).to(device)
    x = crop_model.preprocess(img)
    x = torch.from_numpy(x).expand(1, 1, -1, -1).float().to(device)
    with torch.inference_mode():
        coords = crop_model(x, img_shape)
    coords = coords[0].cpu().numpy()
    x, y, w, h = coords
    return img[y: y + h, x: x + w]

# %% cellule 10
def load_dicom(path):
    dcm_file = dicomsdl.open(str(path))
    data = dcm_file.pixelData()
    data = (data - data.min()) / (data.max() - data.min())
    if dcm_file.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = 1 - data
    return (data * 255).astype(np.uint8)

# %% [markdown] cellule 11
# ## 4. Main prediction

# %% cellule 12
def predict(cc_path=None, mlo_path=None):
    """
    cc_path  : path to the craniocaudal view image (CC)
    mlo_path : path to the oblique view image (MLO)
    """
    if not cc_path or not mlo_path:
        print(" Both images (CC and MLO) are required!")
        return
    if not cc_path.lower().endswith(".dcm"):
        print(" CC: only DICOM files (.dcm) are accepted!")
        return
    if not mlo_path.lower().endswith(".dcm"):
        print(" MLO: only DICOM files (.dcm) are accepted!")
        return
    print("🔍 Detection in progress...")
    images = {}

     # Load CC + MLO with load_dicom (pydicom)
    images["cc"]  = load_dicom(cc_path)
    images["mlo"] = load_dicom(mlo_path)

    # Crop with the pydicom version (the last one loaded)
    images["cc"]  = crop_mammo(images["cc"],  crop_model, device)
    images["mlo"] = crop_mammo(images["mlo"], crop_model, device)

    # Prediction
    with torch.inference_mode():
        output = model(images, device=device)

    cancer_score  = output['cancer'].item()

    print(f"\n=== Results ===")
    print(f"Score : {cancer_score:.4f}")
    print(f"({'Suspect' if cancer_score > 0.013 else 'Normal'})")

# %% [markdown] cellule 13
# ## 5. Utilisation

# %% cellule 14
# With DICOM files:
# predict("vue_cc.dcm", "vue_mlo.dcm")

# With already converted PNGs:
# predict("vue_cc.png", "vue_mlo.png")
predict(
    cc_path  = "/content/00002_20990909_L_CC_1.dcm",   # vue CC
    mlo_path = "/content/00002_20990909_L_MLO_1.dcm",   # vue MLO
)
