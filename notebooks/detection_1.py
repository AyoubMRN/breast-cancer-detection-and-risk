# Conversion en script des cellules de : modele détection 1.ipynb (Drive, POC/DEEP LEARNING/DÉTECTION/Détection 1)
# Sorties et images retirées.

# %% [markdown] cellule 1
# # Détection de cancer du sein — MammoScreen
# 
# Analyse mammographique avec les modèles `ianpan/mammo-crop` et `ianpan/mammoscreen`.

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
# ## 1. Choisir GPU ou CPU

# %% cellule 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation : {device}")

# %% [markdown] cellule 6
# ## 2. Charger les modèles depuis HuggingFace

# %% cellule 7
print("Téléchargement des modèles... (première fois seulement)")

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
# ## 4. Prédiction principale

# %% cellule 12
def predict(cc_path=None, mlo_path=None):
    """
    cc_path  : chemin vers l'image vue de face (CC)
    mlo_path : chemin vers l'image vue oblique (MLO)
    """
    if not cc_path or not mlo_path:
        print(" Les deux images (CC et MLO) sont obligatoires !")
        return
    if not cc_path.lower().endswith(".dcm"):
        print(" CC : seuls les fichiers DICOM (.dcm) sont acceptés !")
        return
    if not mlo_path.lower().endswith(".dcm"):
        print(" MLO : seuls les fichiers DICOM (.dcm) sont acceptés !")
        return
    print("🔍 Détection en cours...")
    images = {}

     # Charger CC + MLO avec load_dicom (pydicom)
    images["cc"]  = load_dicom(cc_path)
    images["mlo"] = load_dicom(mlo_path)

    # Crop avec la version pydicom (la dernière chargée)
    images["cc"]  = crop_mammo(images["cc"],  crop_model, device)
    images["mlo"] = crop_mammo(images["mlo"], crop_model, device)

    # Prédiction
    with torch.inference_mode():
        output = model(images, device=device)

    cancer_score  = output['cancer'].item()

    print(f"\n=== Résultats ===")
    print(f"Score : {cancer_score:.4f}")
    print(f"({'Suspect' if cancer_score > 0.013 else 'Normal'})")

# %% [markdown] cellule 13
# ## 5. Utilisation

# %% cellule 14
# Avec des fichiers DICOM :
# predict("vue_cc.dcm", "vue_mlo.dcm")

# Avec des PNG déjà convertis :
# predict("vue_cc.png", "vue_mlo.png")
predict(
    cc_path  = "/content/00002_20990909_L_CC_1.dcm",   # vue CC
    mlo_path = "/content/00002_20990909_L_MLO_1.dcm",   # vue MLO
)
