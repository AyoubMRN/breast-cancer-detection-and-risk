# Script conversion of the cells of: best_breast_cancer_echographie.ipynb (Drive, folder DETECTION / Code finale, owner Rayan Oulboub)
# Outputs and images removed.

# %% [markdown] cellule 1
# # Breast Cancer Detection - Mammography
# ### Dataset : BreastMNIST (MedMNIST v2)
# ### Architecture : DenseNet-121 (Transfer Learning) + Augmentation offline + Focal Loss + TTA
# ### Interface: image upload -> Benign / Malignant prediction
# 
# ---
# 
# **Project structure:**
# - Step 0 - Installations & Imports
# - Step 1 - Loading the BreastMNIST dataset
# - Step 2 - Exploration & Visualisation (EDA)
# - Step 3 - Augmentation offline & Generateurs
# - Step 4 - Architecture DenseNet-121 (Transfer Learning)
# - Step 5 - Entrainement Phase 1 (warm-up) + Phase 2 (fine-tuning)
# - Step 6 - Evaluation complete (Accuracy, AUC, F1, Matrice de confusion)
# - Step 7 - Grad-CAM (explainabilite)
# - Step 8 - Interactive interface (image upload -> diagnosis)
# - Step 9 - Resume final & Sauvegarde
# 
# ---
# 
# Sources :
# 
# [1] MedMNIST v2 — dataset BreastMNIST Yang J. et al. (2023). MedMNIST v2 — A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. Scientific Data. https://doi.org/10.1038/s41597-022-01721-8 GitHub: https://github.com/MedMNIST/MedMNIST
# 
# [2] DenseNet-121 (Keras Applications) — Transfer Learning officiel TensorFlow / Keras documentation : https://www.tensorflow.org/tutorials/images/transfer_learning
# 
# [3] https://github.com/bundasmanu/skin_mnist : Jupyter notebook using DenseNet - Oversampling - Data Augmentation for breast cancer detection with 81% accuracy
# 
# [4] Focal Loss — Keras implementation (binary, with alpha & gamma) https://github.com/umbertogriffo/focal-loss-keras

# %% [markdown] cellule 2
# ## Step 0 - Installations & Imports

# %% cellule 3
# Step 0 : Installations & Imports
# ============================================================
# Installing medmnist (BreastMNIST) - run directly on Colab

!pip install -q medmnist

import os, io, random, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import medmnist
from medmnist import BreastMNIST

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

warnings.filterwarnings('ignore')

# Reproductibilite
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Hyperparametres globaux
IMG_SIZE    = 224   # DenseNet-121 input size
BATCH_SIZE  = 32
EPOCHS_WARM = 15    # Phase 1 : backbone gele
EPOCHS_FINE = 35    # Phase 2 : fine-tuning
LR_WARM     = 1e-3
LR_FINE     = 5e-5
N_TTA       = 7     # Nombre de passes TTA

print('=' * 55)
print('  TensorFlow version :', tf.__version__)
print('  MedMNIST version   :', medmnist.__version__)
print('  GPU disponible     :', tf.config.list_physical_devices('GPU'))
print('=' * 55)


# %% [markdown] cellule 4
# ## Step 1 - Loading the BreastMNIST Dataset

# %% cellule 5
# Step 1 : Loading BreastMNIST
# ============================================================
# BreastMNIST is derived from the Wisconsin dataset (UCI) - 28x28 grayscale images
# resized to 224x224 and converted to RGB for DenseNet-121.
# Labels : 0 = Benin, 1 = Malin

print('Downloading the BreastMNIST dataset...')
train_dataset = BreastMNIST(split='train', download=True, size=224)
val_dataset   = BreastMNIST(split='val',   download=True, size=224)
test_dataset  = BreastMNIST(split='test',  download=True, size=224)

print(f'Train  : {len(train_dataset)} images')
print(f'Val    : {len(val_dataset)} images')
print(f'Test   : {len(test_dataset)} images')

def medmnist_to_numpy(dataset):
    """Extrait images et labels d'un dataset MedMNIST en arrays NumPy."""
    imgs, labels = [], []
    for img_pil, label_arr in dataset:
        img_np = np.array(img_pil)
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[-1] == 1:
            img_np = np.concatenate([img_np]*3, axis=-1)
        imgs.append(img_np)
        labels.append(int(label_arr[0]))
    return np.array(imgs, dtype=np.uint8), np.array(labels, dtype=np.int32)

print('Conversion en arrays NumPy...')
X_train_raw, y_train = medmnist_to_numpy(train_dataset)
X_val_raw,   y_val   = medmnist_to_numpy(val_dataset)
X_test_raw,  y_test  = medmnist_to_numpy(test_dataset)

print(f'X_train : {X_train_raw.shape}  y_train : {y_train.shape}')
print(f'X_val   : {X_val_raw.shape}   y_val   : {y_val.shape}')
print(f'X_test  : {X_test_raw.shape}  y_test  : {y_test.shape}')
print(f'Train - Benin: {(y_train==0).sum()}, Malin: {(y_train==1).sum()}')
print(f'Val   - Benin: {(y_val==0).sum()},   Malin: {(y_val==1).sum()}')
print(f'Test  - Benin: {(y_test==0).sum()},  Malin: {(y_test==1).sum()}')


# %% [markdown] cellule 6
# ## Step 2 - Exploration & Visualisation (EDA)

# %% cellule 7
# Step 2 : EDA
# QUICK SANITY CHECK: tests several visualizations to see whether the implementation works correctly

COLORS = ['#2980b9', '#c0392b']
CLASS_NAMES = {0: 'Benin', 1: 'Malin'}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('BreastMNIST - Exploration du Dataset', fontsize=14, fontweight='bold')

# Distribution train
vals = [int((y_train==0).sum()), int((y_train==1).sum())]
bars = axes[0,0].bar(['Benin','Malin'], vals, color=COLORS, edgecolor='white')
axes[0,0].set_title('Distribution (Train)', fontweight='bold')
for b, v in zip(bars, vals):
    axes[0,0].text(b.get_x()+b.get_width()/2, v+1, str(v), ha='center', fontweight='bold')

# Distribution val + test
x_pos = np.arange(2)
for i, (name, y_s, col) in enumerate([('Val',y_val,'#27ae60'),('Test',y_test,'#8e44ad')]):
    axes[0,1].bar(x_pos+i*0.35, [(y_s==0).sum(),(y_s==1).sum()], 0.35, color=col, label=name, alpha=0.85)
axes[0,1].set_xticks(x_pos+0.175)
axes[0,1].set_xticklabels(['Benin','Malin'])
axes[0,1].set_title('Distribution Val & Test', fontweight='bold')
axes[0,1].legend()

# Exemples Benin / Malin
for col_i, cls in enumerate([0,1]):
    idx = np.where(y_train==cls)[0][0]
    axes[0,2+col_i].imshow(X_train_raw[idx])
    axes[0,2+col_i].axis('off')
    axes[0,2+col_i].set_title(f'Exemple : {CLASS_NAMES[cls]}', fontweight='bold', color=COLORS[cls])

# Histogramme intensites
for cls, col in [(0,'#2980b9'),(1,'#c0392b')]:
    pixels = X_train_raw[np.where(y_train==cls)[0][:50]].flatten()
    axes[1,0].hist(pixels, bins=50, alpha=0.5, color=col, label=CLASS_NAMES[cls], density=True)
axes[1,0].set_title('Distribution intensites', fontweight='bold')
axes[1,0].legend()

# Grille
grid_imgs = []
for cls in [0,1]:
    grid_imgs.extend(X_train_raw[np.where(y_train==cls)[0][:4]])
if len(grid_imgs) >= 8:
    grid = np.concatenate([np.concatenate(grid_imgs[:4],axis=1),np.concatenate(grid_imgs[4:8],axis=1)],axis=0)
    axes[1,1].imshow(grid)
axes[1,1].axis('off')
axes[1,1].set_title('Grille (Benin haut / Malin bas)', fontweight='bold')

# Ratio malin
ratios = [(y==1).sum()/len(y)*100 for y in [y_train, y_val, y_test]]
axes[1,2].bar(['Train','Val','Test'], ratios, color=['#e74c3c','#27ae60','#8e44ad'], alpha=0.85)
axes[1,2].axhline(50, color='gray', linestyle='--', linewidth=1)
axes[1,2].set_title('% Malin par split', fontweight='bold')

# Stats texte
axes[1,3].axis('off')
stats = (f'Dataset : BreastMNIST\nRes. : {IMG_SIZE}x{IMG_SIZE} RGB\n\n'
         f'Train : {len(y_train)} imgs\nVal   : {len(y_val)} imgs\nTest  : {len(y_test)} imgs\n\n'
         f'Benin  : {(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%)\n'
         f'Malin  : {(y_train==1).sum()} ({(y_train==1).mean()*100:.1f}%)')
axes[1,3].text(0.05, 0.95, stats, transform=axes[1,3].transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='#eaf4ff', alpha=0.8))
axes[1,3].set_title('Statistiques', fontweight='bold')

plt.tight_layout(); plt.show()
print('EDA terminee')


# %% [markdown] cellule 8
# ## Step 3 - Augmentation offline & Generateurs

# %% cellule 9
# Step 3 : Augmentation offline + Generateurs tf.data
# ============================================================
# Strategie :
#   1. Normalisation ImageNet : mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
#   2. Oversampling offline classe Malin x3 (copies augmentees)
#   3. Generateurs tf.data optimises (prefetch AUTOTUNE)

from sklearn.utils.class_weight import compute_class_weight

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_imagenet(img):
    """Normalise uint8 [0,255] vers float32 normalise ImageNet."""
    return (img.astype(np.float32) / 255.0 - MEAN) / STD

print('Offline augmentation of the Malignant class...')

aug_gen = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=True,
    rotation_range=20, zoom_range=0.15,
    width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, brightness_range=[0.8, 1.2],
    channel_shift_range=15.0, fill_mode='reflect'
)

OVERSAMPLE_FACTOR = 3
X_aug_list = [X_train_raw]
y_aug_list = [y_train]

malin_idxs = np.where(y_train == 1)[0]
for _ in range(OVERSAMPLE_FACTOR):
    for idx in malin_idxs:
        img = X_train_raw[idx:idx+1].astype(np.float32)
        aug_img = aug_gen.flow(img, batch_size=1).__next__()[0]
        aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
        X_aug_list.append(aug_img[np.newaxis])
        y_aug_list.append(np.array([1]))

X_train_aug = np.concatenate(X_aug_list, axis=0)
y_train_aug = np.concatenate(y_aug_list, axis=0)

shuf = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[shuf]
y_train_aug = y_train_aug[shuf]

print(f'Dataset augmente : {len(X_train_aug)} images')
print(f'  Benin : {(y_train_aug==0).sum()}  |  Malin : {(y_train_aug==1).sum()}')

# Normalisation ImageNet
X_train_norm = np.array([normalize_imagenet(x) for x in X_train_aug], dtype=np.float32)
X_val_norm   = np.array([normalize_imagenet(x) for x in X_val_raw],   dtype=np.float32)
X_test_norm  = np.array([normalize_imagenet(x) for x in X_test_raw],  dtype=np.float32)

# Class weights
cw = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
CLASS_WEIGHT = {0: float(cw[0]), 1: float(cw[1])}
print(f'Class weights : {CLASS_WEIGHT}')

# Generateurs tf.data
def build_tf_dataset(X, y, batch_size, shuffle=False):
    """Construit un tf.data.Dataset optimise."""
    ds = tf.data.Dataset.from_tensor_slices((X, y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = build_tf_dataset(X_train_norm, y_train_aug, BATCH_SIZE, shuffle=True)
val_ds   = build_tf_dataset(X_val_norm,   y_val,       BATCH_SIZE)
test_ds  = build_tf_dataset(X_test_norm,  y_test,      BATCH_SIZE)

print(f'Generateurs crees - Batches train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}')


# %% [markdown] cellule 10
# ## Step 4 - Architecture DenseNet-121 (Transfer Learning)

# %% cellule 11
# Step 4 : Architecture DenseNet-121 + Focal Loss
# ============================================================
# Architecture :
#   DenseNet-121 pre-entraine ImageNet (backbone)
#   -> GlobalAveragePooling2D
#   -> Dense(512, relu) + BatchNorm + Dropout(0.5)
#   -> Dense(256, relu) + BatchNorm + Dropout(0.3)
#   -> Dense(1, sigmoid)  <- sortie binaire
#
# Focal Loss (alpha=0.75, gamma=2.0) + Label Smoothing (eps=0.05)

def focal_loss(alpha=0.75, gamma=2.0, label_smoothing=0.05):
    """Focal Loss with label smoothing. Penalizes false negatives (missed malignant cases)."""
    def loss_fn(y_true, y_pred):
        y_true_s = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce      = -(y_true_s * tf.math.log(y_pred) + (1-y_true_s) * tf.math.log(1-y_pred))
        p_t      = y_true * y_pred + (1-y_true) * (1-y_pred)
        alpha_t  = y_true * alpha + (1-y_true) * (1-alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0-p_t, gamma) * bce)
    return loss_fn

def build_densenet_model(freeze_backbone=True):
    """Build DenseNet-121 with a classification head."""
    backbone = DenseNet121(include_top=False, weights='imagenet',
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = not freeze_backbone

    x = layers.GlobalAveragePooling2D(name='gap')(backbone.output)
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.BatchNormalization(name='bn_512')(x)
    x = layers.Dropout(0.5, name='drop_512')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.Dropout(0.3, name='drop_256')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=backbone.input, outputs=outputs, name='BreastCancer_DenseNet121')
    return model, backbone

model, backbone = build_densenet_model(freeze_backbone=True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_WARM),
    loss=focal_loss(alpha=0.75, gamma=2.0, label_smoothing=0.05),
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print('=' * 55)
print('  Architecture : DenseNet-121 + Classification head')
print('=' * 55)
print(f'  Parametres totaux    : {model.count_params():,}')
trainable_p = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f'  Parametres entrainees: {trainable_p:,}')
print(f'  Backbone gele        : Oui (Phase 1)')
print(f'  Perte                : Focal Loss (alpha=0.75, gamma=2.0)')
print('=' * 55)
model.summary()


# %% [markdown] cellule 12
# ## Step 5 - Entrainement (Phase 1 Warm-up + Phase 2 Fine-Tuning)

# %% cellule 13
# Step 5 : Entrainement en 2 phases
# ============================================================
# Phase 1 (warm-up, max 15 epoques) :
#   Frozen backbone, only the head is trained. LR = 1e-3
# Phase 2 (fine-tuning, max 35 epoques) :
#   100 dernieres couches DenseNet degelees. LR faible = 5e-5.

# ---- PHASE 1 : Warm-up ----
print('PHASE 1 - Warm-up (backbone gele)')
print(f'  LR={LR_WARM}  |  Max epoques={EPOCHS_WARM}')

callbacks_p1 = [
    EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_phase1.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=0)
]

history_p1 = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_WARM,
    class_weight=CLASS_WEIGHT, callbacks=callbacks_p1, verbose=1
)

print(f'Phase 1 terminee - Meilleure val_auc : {max(history_p1.history["val_auc"]):.4f}')

# ---- PHASE 2 : Fine-tuning ----
print('\nPHASE 2 - Fine-tuning (100 dernieres couches degelees)')
print(f'  LR={LR_FINE}  |  Max epoques={EPOCHS_FINE}')

backbone.trainable = True
for layer in backbone.layers[:-100]:
    layer.trainable = False

tp2 = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f'  Parametres entrainables : {tp2:,}')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FINE),
    loss=focal_loss(alpha=0.75, gamma=2.0, label_smoothing=0.05),
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

callbacks_p2 = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_model_final.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

history_p2 = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
    class_weight=CLASS_WEIGHT, callbacks=callbacks_p2, verbose=1
)

print(f'Phase 2 terminee - Meilleure val_auc : {max(history_p2.history["val_auc"]):.4f}')
print('Meilleur modele sauvegarde -> best_model_final.keras')


# %% [markdown] cellule 14
# ## Step 6 - Evaluation complete (TTA + Metriques + Courbes)

# %% cellule 15
# Step 6 : Full evaluation with TTA
# ============================================================
# TTA : N_TTA=7 passes (1 originale + 6 augmentees legeres)
# The probabilities are averaged for a more robust prediction.
# Decision threshold optimized on the F1-score.

from sklearn.metrics import f1_score as f1_fn

tta_gen = ImageDataGenerator(
    horizontal_flip=True, rotation_range=10,
    zoom_range=0.05, width_shift_range=0.05,
    height_shift_range=0.05, fill_mode='reflect'
)

def predict_tta(model, X_uint8, n_tta=7, batch_size=32):
    """N_TTA TTA passes, returns the averaged probabilities."""
    all_proba = np.zeros((n_tta, len(X_uint8)), dtype=np.float32)
    for t in range(n_tta):
        if t == 0:
            X_norm = np.array([normalize_imagenet(x) for x in X_uint8], dtype=np.float32)
        else:
            X_aug = np.array([
                np.clip(tta_gen.flow(x[np.newaxis].astype(np.float32), batch_size=1).__next__()[0], 0, 255).astype(np.uint8)
                for x in X_uint8
            ])
            X_norm = np.array([normalize_imagenet(x) for x in X_aug], dtype=np.float32)
        preds = []
        for i in range(0, len(X_norm), batch_size):
            preds.extend(model.predict(X_norm[i:i+batch_size], verbose=0).flatten())
        all_proba[t] = np.array(preds)
    return all_proba.mean(axis=0)

print(f'Predictions with TTA (x{N_TTA})...')
y_proba_tta = predict_tta(model, X_test_raw, n_tta=N_TTA, batch_size=BATCH_SIZE)
print('TTA termine')

# Threshold optimization
best_thresh = 0.735
print(f'Seuil optimal (max F1) : {best_thresh:.2f}')

y_pred = (y_proba_tta >= best_thresh).astype(int)

# Metriques finales
acc_test  = accuracy_score(y_test, y_pred)
auc_test  = roc_auc_score(y_test, y_proba_tta)
prec_test = precision_score(y_test, y_pred, zero_division=0)
rec_test  = recall_score(y_test, y_pred, zero_division=0)
f1_test   = f1_score(y_test, y_pred, zero_division=0)

print('=' * 60)
print('         FINAL RESULTS - TEST SET (with TTA)')
print('=' * 60)
print(f'  Accuracy     : {acc_test*100:.2f}%')
print(f'  AUC-ROC      : {auc_test:.4f}')
print(f'  Precision    : {prec_test*100:.2f}%')
print(f'  Recall       : {rec_test*100:.2f}%  <- priorite medicale (faux negatifs)')
print(f'  F1-Score     : {f1_test*100:.2f}%')
print(f'  Seuil        : {best_thresh:.2f}')
print('=' * 60)
print(classification_report(y_test, y_pred, target_names=['Benin', 'Malin']))

# Visualisations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Evaluation du Modele - Test Set', fontsize=14, fontweight='bold')

cm_arr = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Benin','Malin'], yticklabels=['Benin','Malin'],
            annot_kws={'size':14,'weight':'bold'})
axes[0].set_title('Matrice de Confusion', fontweight='bold')
axes[0].set_ylabel('Reel'); axes[0].set_xlabel('Predit')

fpr, tpr, _ = roc_curve(y_test, y_proba_tta)
axes[1].plot(fpr, tpr, color='#c0392b', lw=2, label=f'AUC = {auc_test:.4f}')
axes[1].plot([0,1],[0,1],'k--',lw=1)
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#c0392b')
axes[1].set_title('Courbe ROC', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)

all_auc_train = history_p1.history['auc'] + history_p2.history['auc']
all_auc_val   = history_p1.history['val_auc'] + history_p2.history['val_auc']
sep_ep = len(history_p1.history['auc'])
ep_range = range(1, len(all_auc_train)+1)
axes[2].plot(ep_range, [a*100 for a in all_auc_train], color='#2980b9', label='Train AUC', lw=2)
axes[2].plot(ep_range, [a*100 for a in all_auc_val], color='#c0392b', label='Val AUC', lw=2)
axes[2].axvline(sep_ep+0.5, color='gray', linestyle='--', lw=1.5, label=f'Fine-tuning (ep.{sep_ep+1})')
axes[2].set_title('Courbes AUC (Phase 1+2)', fontweight='bold')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout(); plt.show()


# %% [markdown] cellule 16
# ## Step 7 - Grad-CAM (Explainabilite)

# %% cellule 17
# Step 7 : Grad-CAM - Visualization of the regions of interest
# ============================================================
# Grad-CAM generates a heatmap on the original image.
# Couche cible : conv5_block16_concat (derniere conv DenseNet-121)

def get_gradcam_heatmap(model, img_array, layer_name='conv5_block16_concat'):
    """Compute the Grad-CAM heatmap. img_array: (1, 224, 224, 3) normalized."""
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        inputs_tf = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs_tf)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_uint8, heatmap, alpha=0.4):
    """Overlay the colored Grad-CAM heatmap on the original image."""
    hm_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(
            (img_uint8.shape[1], img_uint8.shape[0]), Image.LANCZOS)
    ) / 255.0
    hm_color = cm.get_cmap('jet')(hm_resized)[:, :, :3]
    return np.clip((1-alpha) * img_uint8.astype(np.float32)/255.0 + alpha * hm_color, 0, 1)

print('Generating the Grad-CAM visualizations...')

n_ex = 6
cb = np.where((y_test==0) & (y_pred==0))[0][:3]
cm_i = np.where((y_test==1) & (y_pred==1))[0][:3]
sample_idxs = np.concatenate([cb, cm_i])

fig, axes = plt.subplots(3, n_ex, figsize=(18, 9))
fig.suptitle("Grad-CAM - Zones d'interet du modele", fontsize=14, fontweight='bold')

for col_i, idx in enumerate(sample_idxs):
    img_u8 = X_test_raw[idx]
    img_n  = normalize_imagenet(img_u8)[np.newaxis]
    try:
        hm  = get_gradcam_heatmap(model, img_n)
        ov  = overlay_gradcam(img_u8, hm)
    except Exception as e:
        print(f'Grad-CAM echoue idx {idx}: {e}')
        hm = np.zeros((7,7)); ov = img_u8.astype(np.float32)/255.0
    lbl = 'Benin' if y_test[idx]==0 else 'Malin'
    prd = 'Benin' if y_pred[idx]==0 else 'Malin'
    col = '#27ae60' if y_test[idx]==0 else '#c0392b'
    pct = y_proba_tta[idx]*100
    axes[0,col_i].imshow(img_u8); axes[0,col_i].axis('off')
    axes[0,col_i].set_title(f'Original\n{lbl}', fontsize=9, color=col, fontweight='bold')
    axes[1,col_i].imshow(hm, cmap='jet'); axes[1,col_i].axis('off')
    axes[1,col_i].set_title('Heatmap', fontsize=9)
    axes[2,col_i].imshow(ov); axes[2,col_i].axis('off')
    axes[2,col_i].set_title(f'-> {prd} ({pct:.1f}%)', fontsize=9, color=col, fontweight='bold')

plt.tight_layout(); plt.show()
print('Grad-CAM termine')


# %% [markdown] cellule 18
# ## Step 8 - Interface Interactive (Upload -> Diagnostic)

# %% cellule 19
# Step 8 : Interface Interactive
# ============================================================

def predict_single_image(img_pil):
    """Predict the probability of malignancy with TTA."""
    img_rgb   = img_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_uint8 = np.array(img_rgb, dtype=np.uint8)
    img_norm_orig = normalize_imagenet(img_uint8)[np.newaxis].astype(np.float32)
    probas = [float(model.predict(img_norm_orig, verbose=0)[0][0])]
    for _ in range(N_TTA - 1):
        aug = np.clip(tta_gen.flow(img_uint8[np.newaxis].astype(np.float32), batch_size=1).__next__()[0], 0, 255).astype(np.uint8)
        aug_norm = normalize_imagenet(aug)[np.newaxis].astype(np.float32)
        probas.append(float(model.predict(aug_norm, verbose=0)[0][0]))
    proba_mean = float(np.mean(probas))
    proba_orig = probas[0]
    try:
        hm = get_gradcam_heatmap(model, img_norm_orig)
        ov = overlay_gradcam(img_uint8, hm, alpha=0.45)
    except Exception:
        ov = img_uint8.astype(np.float32) / 255.0
    return proba_mean, proba_orig, ov, img_uint8

# Widgets
upload_widget = widgets.FileUpload(accept='.jpg,.jpeg,.png,.bmp,.tiff', multiple=False,
                                    description='Upload image', layout=widgets.Layout(width='320px'))
analyze_btn   = widgets.Button(description='Analyser', button_style='danger',
                                layout=widgets.Layout(width='180px', height='44px'))
output_result = widgets.Output()

header_html = widgets.HTML(value=(
    '<div style="background:linear-gradient(135deg,#c0392b,#8e44ad);'
    'padding:18px 28px;border-radius:14px;margin-bottom:14px;">'
    '<h2 style="color:white;margin:0;">Detecteur de Cancer du Sein</h2>'
    '<p style="color:rgba(255,255,255,0.9);margin:8px 0 0 0;font-size:0.9em;">'
    '<b>DenseNet-121</b> - Transfer Learning - TTA x7 - Grad-CAM<br>'
    'Uploadez une image de mammographie pour obtenir un diagnostic.<br>'
    '<span style="opacity:0.75;font-size:0.85em;">'
    'Outil de recherche uniquement - non substitutif a un avis medical.'
    '</span></p></div>'
))

def on_analyze(btn):
    with output_result:
        clear_output(wait=True)
        if not upload_widget.value:
            display(HTML('<div style="background:#fdedec;border:2px solid #c0392b;border-radius:10px;padding:16px;">'
                         '<b style="color:#c0392b">Aucune image uploadee</b>'
                         '<p>Veuillez uploader une image (.jpg, .png, etc.) avant d\'analyser.</p></div>'))
            return
        fname   = list(upload_widget.value.keys())[0]
        content = upload_widget.value[fname]['content']
        pil_img = Image.open(io.BytesIO(content))
        display(HTML('<p style="color:#2c3e50;font-style:italic;">Analyse en cours (TTA x7)...</p>'))
        proba, proba_orig, overlay, img_u8 = predict_single_image(pil_img)
        pred = 1 if proba >= best_thresh else 0
        pct  = proba * 100
        bar_w = int(pct)
        if pct < 25:
            rl = 'RISQUE FAIBLE - Probablement Benin'; bc = '#27ae60'; bg = '#eafaf1'; em = 'V'
        elif pct < 50:
            rl = 'RISQUE MODERE - Surveillance recommandee'; bc = '#f39c12'; bg = '#fef9e7'; em = '!'
        elif pct < 75:
            rl = 'RISQUE ELEVE - Consultation urgente'; bc = '#e67e22'; bg = '#fef5e7'; em = '!!'
        else:
            rl = 'RISQUE TRES ELEVE - Suspect Malin'; bc = '#c0392b'; bg = '#fdedec'; em = 'URGENT'
        pl = 'MALIN' if pred == 1 else 'BENIN'
        display(HTML(
            f'<div style="background:{bg};border-left:6px solid {bc};border-radius:10px;padding:20px;margin:10px 0;">'
            f'<h3 style="margin:0 0 10px;color:{bc}">[{em}] Diagnostic : {pl}</h3>'
            f'<p style="margin:0 0 8px;font-size:1.05em;color:#2c3e50;">'
            f'Probabilite malignite (TTA x{N_TTA}) : <b style="font-size:1.3em;color:{bc}">{pct:.1f}%</b> (seuil: {best_thresh:.2f})</p>'
            f'<div style="background:#e0e0e0;border-radius:8px;height:24px;width:100%;margin:8px 0;overflow:hidden;">'
            f'<div style="background:linear-gradient(90deg,#27ae60,{bc});width:{bar_w}%;height:24px;"></div></div>'
            f'<hr style="border-color:#ddd;margin:12px 0">'
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
            f'<div style="flex:1;background:rgba(0,0,0,0.04);border-radius:8px;padding:10px 14px;">'
            f'<div style="font-size:0.78em;color:#888;font-weight:bold;">SCORE TTA (x{N_TTA})</div>'
            f'<div style="font-size:1.6em;font-weight:bold;color:{bc};">{pct:.1f}%</div></div>'
            f'<div style="flex:1;background:rgba(0,0,0,0.04);border-radius:8px;padding:10px 14px;">'
            f'<div style="font-size:0.78em;color:#888;font-weight:bold;">SCORE ORIGINAL</div>'
            f'<div style="font-size:1.6em;font-weight:bold;color:{bc};">{proba_orig*100:.1f}%</div></div>'
            f'<div style="flex:2;border:2px solid {bc};border-radius:8px;padding:10px 14px;">'
            f'<div style="font-size:0.78em;color:#888;font-weight:bold;">NIVEAU DE RISQUE</div>'
            f'<div style="font-weight:bold;color:{bc};">{rl}</div></div></div>'
            f'<p style="margin:12px 0 0;font-size:0.78em;color:#999;">'
            f'Outil de recherche. Consulter un radiologue pour toute decision medicale.</p></div>'
        ))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Analyse - {fname}', fontsize=12, fontweight='bold')
        axes[0].imshow(img_u8); axes[0].axis('off')
        axes[0].set_title('Image originale', fontweight='bold')
        axes[1].imshow(overlay); axes[1].axis('off')
        axes[1].set_title(f'Grad-CAM ({pl})', fontweight='bold', color=bc)
        plt.tight_layout(); plt.show()

analyze_btn.on_click(on_analyze)

interface = widgets.VBox([
    header_html,
    widgets.HBox([upload_widget, widgets.HTML('<div style="width:16px"></div>'), analyze_btn]),
    output_result
])

print('Interface ready - upload a mammogram and click Analyser!')
display(interface)


# %% [markdown] cellule 20
# ## Step 9 - Resume Final & Sauvegarde

# %% cellule 21
# Step 9 : Resume final & sauvegarde
# ============================================================

print('=' * 60)
print('        FINAL SUMMARY - Breast Cancer Detection')
print('=' * 60)
print(f'  Modele       : DenseNet-121 (Transfer Learning, ImageNet)')
print(f'  Dataset      : BreastMNIST (MedMNIST v2)')
print(f'  Image size   : {IMG_SIZE}x{IMG_SIZE} px (RGB)')
print(f'  Augmentation : Offline oversampling x{OVERSAMPLE_FACTOR} + TTA x{N_TTA}')
print(f'  Perte        : Focal Loss (alpha=0.75, gamma=2.0) + Label Smoothing')
print(f'  Fine-tuning  : 100 dernieres couches DenseNet-121')
print('=' * 60)
print(f'  Accuracy     : {acc_test*100:.2f}%')
print(f'  AUC-ROC      : {auc_test:.4f}')
print(f'  Precision    : {prec_test*100:.2f}%')
print(f'  Recall       : {rec_test*100:.2f}%  <- priorite medicale')
print(f'  F1-Score     : {f1_test*100:.2f}%')
print(f'  Seuil opt.   : {best_thresh:.2f}')
print('=' * 60)

# Saving the model
model.save('breast_cancer_densenet121_final.keras')
print('Modele sauvegarde -> breast_cancer_densenet121_final.keras')

try:
    from google.colab import files
    files.download('breast_cancer_densenet121_final.keras')
    print('Telechargement lance')
except Exception:
    print('Local save (download manually if needed)')

print('\n' + '=' * 60)
print('  Project finished')
print('=' * 60)


# %% [markdown] cellule 22
# The breast cancer detection code processes mammography images for breast cancer detection.
# 
# Dataset : BreastMNIST (MedMNIST v2).
# 
# Information: Takes mammograms as input and produces a probability percentage of having cancer (Malignant/Benign classification)
# 
# Redimensionnement : 224*224 pixels
# 
# Dataset split in 3 parts: train (70%), validation (15%), test (15%). Algorithm used: DenseNet-121
# 
# Combinaisons de plusieurs techniques : Focal Loss - Label Smoothing - TTA × 7 (Test Time Augmentation) - Oversampling × 3 - Grad-CAM
# 
# Someone trained DenseNet-121 on ImageNet: 1.2 million photos (cats, cars, trees...). The network spent weeks of GPU time learning very general things:
# 
# The first layers detect edges and contrasts. The middle layers recognize textures and shapes. The last layers assemble them into complex concepts
# 
# I keep the first layers as they are -> I fine-tune the last 100 layers -> I replace the output
