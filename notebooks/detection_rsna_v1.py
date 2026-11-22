# Script conversion of the cells of the Colab notebook detection-rsnav1.ipynb
# Outputs and images removed.

# %% cellule 1
import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras import layers, models

# Automatic path scanner
CHEMIN_CSV = ""
DOSSIER_IMAGES = '/kaggle/input/datasets/theoviel/rsna-breast-cancer-256-pngs/'

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'train.csv':
            CHEMIN_CSV = os.path.join(dirname, filename)
            print(f"✅ File found: {CHEMIN_CSV}")

df = pd.read_csv(CHEMIN_CSV).sample(n=2000, random_state=42).reset_index(drop=True)
df['age'] = df['age'].fillna(df['age'].mean())

# %% cellule 2
print("\n--- 1. CLINICAL BENCHMARK ---")
X_clinique = df[['age', 'implant']].values
y = df['cancer'].values

# Split
X_clin_train, X_clin_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_clinique, y, df.index, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_clin_train = scaler.fit_transform(X_clin_train)
X_clin_test = scaler.transform(X_clin_test)

# Comparatif
modeles_cliniques = {
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss'),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

scores_cliniques = {}
for nom, m in modeles_cliniques.items():
    m.fit(X_clin_train, y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_clin_test)[:, 1])
    scores_cliniques[nom] = auc
    print(f"-> {nom} AUC : {auc:.4f}")

# Keep the best one for the fusion (XGBoost)
modele_clinique_best = modeles_cliniques["XGBoost"]

# %% cellule 3
#Cellule 2 bis

print("\n--- 1. CLINICAL MODULE (FORCED 50/50 SAMPLING) ---")

# 1. Loading
df_full = pd.read_csv(CHEMIN_CSV)

# FORCE THE BALANCE: 500 sick and 500 healthy
# This ensures the model sees as many cancers as healthy cases
n_samples = 500

df_positifs = df_full[df_full['cancer'] == 1].sample(n=min(n_samples, len(df_full[df_full['cancer'] == 1])), random_state=42)
df_negatifs = df_full[df_full['cancer'] == 0].sample(n=len(df_positifs), random_state=42)

df = pd.concat([df_positifs, df_negatifs]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Forced balance: {len(df)} patients ({len(df_positifs)} cancers, {len(df_negatifs)} healthy)")

# 2. Preparation
df['age'] = df['age'].fillna(df['age'].mean())
X_clinique = df[['age', 'implant']].values
y = df['cancer'].values

# 3. Train/Test split
X_clin_train, X_clin_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_clinique, y, df.index, test_size=0.2, random_state=42, stratify=y
)

# 4. Training
scaler = StandardScaler()
X_clin_train = scaler.fit_transform(X_clin_train)
X_clin_test = scaler.transform(X_clin_test)

modele_clinique_best = RandomForestClassifier(n_estimators=100) # Using RF, which is more stable
modele_clinique_best.fit(X_clin_train, y_train)

auc_clinique = roc_auc_score(y_test, modele_clinique_best.predict_proba(X_clin_test)[:, 1])
print(f"✅ Clinical AUC (Random Forest) : {auc_clinique:.4f}")

# %% cellule 4
print("\n--- 2. BENCHMARK IMAGERIE (CNN) ---")

def charger_images(indices):
    imgs = []
    for i in indices:
        nom_fichier = f"{df.loc[i, 'patient_id']}_{df.loc[i, 'image_id']}.png"
        img = cv2.imread(os.path.join(DOSSIER_IMAGES, nom_fichier), cv2.IMREAD_GRAYSCALE)
        if img is None: img = np.zeros((224, 224))
        img = cv2.resize(img, (224, 224))
        imgs.append(np.stack((img,)*3, axis=-1) / 255.0)
    return np.array(imgs)

X_img_train, X_img_test = charger_images(indices_train), charger_images(indices_test)

# Handling class imbalance
weight_for_1 = (len(y_train) / np.sum(y_train == 1)) / 2
class_weights = {0: 1.0, 1: weight_for_1}

# Comparaison EfficientNet vs ResNet
for nom, arch in [("EfficientNet", EfficientNetB0), ("ResNet50", ResNet50)]:
    base = arch(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    modele = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(1, activation='sigmoid')])
    modele.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    modele.fit(X_img_train, y_train, epochs=5, batch_size=32, class_weight=class_weights, verbose=0)
    auc = roc_auc_score(y_test, modele.predict(X_img_test).flatten())
    print(f"-> {nom} AUC : {auc:.4f}")

# Keep EfficientNet for the fusion
modele_imagerie_best = modele

# %% cellule 5
print("\n--- 3. FUSION PREPARATION (WITH THE BEST MODELS) ---")

# USE THE WINNERS: Random Forest for the clinical part, EfficientNet for the image
modele_clinique_final = modeles_cliniques["Random Forest"]

prob_clin_test = modele_clinique_final.predict_proba(X_clin_test)[:, 1]
prob_img_test = modele_imagerie_best.predict(X_img_test).flatten()

# Fusion
X_fusion_test = np.column_stack((prob_clin_test, prob_img_test))
# (Note: for the fusion training, make sure the Random Forest was also trained)
# To keep it simple, the test probabilities are used for the final result

# %% cellule 6
print("\n--- 3. FUSION PREPARATION (CORRECTED) ---")

# Use the best models
modele_clinique_final = modeles_cliniques["Random Forest"]

# 1. TRAIN SET CREATION
prob_clin_train = modele_clinique_final.predict_proba(X_clin_train)[:, 1]
prob_img_train = modele_imagerie_best.predict(X_img_train).flatten()
X_fusion_train = np.column_stack((prob_clin_train, prob_img_train))

# 2. TEST SET CREATION
prob_clin_test = modele_clinique_final.predict_proba(X_clin_test)[:, 1]
prob_img_test = modele_imagerie_best.predict(X_img_test).flatten()
X_fusion_test = np.column_stack((prob_clin_test, prob_img_test))


print("\n--- 4. META-MODEL (LATE FUSION) ---")
# Training the logistic regression on the TRAIN set
meta = LogisticRegression()
meta.fit(X_fusion_train, y_train)

# Final evaluation on the TEST set
auc_fusion = roc_auc_score(y_test, meta.predict_proba(X_fusion_test)[:, 1])

print(f"🏆 AUC FINAL FUSION : {auc_fusion:.4f}")

# %% cellule 7
# --- RUN ONLY ONCE AFTER TRAINING TO SAVE ---
np.save('y_test_sauvegarde.npy', y_test)
np.save('prob_clin_test.npy', prob_clin_test)
np.save('prob_img_test.npy', prob_img_test)
np.save('indices_test.npy', indices_test)
df.to_csv('df_sauvegarde.csv', index=False)
print("✅ Data saved successfully!")
