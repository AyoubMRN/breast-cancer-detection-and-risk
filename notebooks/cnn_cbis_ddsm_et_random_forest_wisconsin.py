# Conversion en script des cellules de : Projet_AI_E3E.ipynb (Drive personnel)
# Sorties et images retirées.

# %% cellule 1
# --- PARTIE 1 : MODÈLE CNN (IMAGES MAMOGRAPHIQUE) ---

# %% cellule 2
# On installe la nouvelle bibliothèque vue sur ta capture d'écran
!pip install kagglehub

import cv2
import numpy as np
import kagglehub
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# Télechargement de la BDD : CBIS-DDSM: Breast Cancer Image Dataset
path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")

# %% cellule 3
import pandas as pd
import os

# On cible le fichier CSV d'entraînement pour les masses tumorales
chemin_csv_masses = os.path.join(path, "csv", "mass_case_description_train_set.csv")

# On demande à Pandas de lire ce fichier
df_diagnostic = pd.read_csv(chemin_csv_masses)

print("Voici à quoi ressemble le vrai fichier médical :")
# On affiche les 5 premières lignes du tableau
display(df_diagnostic.head())

# %% cellule 4
import os
import glob
import pandas as pd

#On liste TOUTES les images jpg du disque
toutes_les_images = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)

#On crée un dictionnaire : { 'LONG_NUMERO_UID' : 'CHEMIN_DE_L_IMAGE' }
dict_images_disque = {}
for img in toutes_les_images:
    # L'UID, c'est le nom du dossier juste avant l'image
    uid = os.path.basename(os.path.dirname(img))
    dict_images_disque[uid] = img

#La fonction Brute Force
def match_infaillible(csv_path):
    csv_path = str(csv_path) # On s'assure que c'est du texte
    # Pour chaque UID existant sur notre disque dur...
    for uid, chemin_img in dict_images_disque.items():
        # ... on regarde s'il est caché à l'intérieur du texte du CSV
        if uid in csv_path:
            return chemin_img
    return None # Si on ne trouve rien

#On applique à notre tableau
df_final = df_diagnostic.copy()
df_final['vrai_chemin'] = df_final['image file path'].apply(match_infaillible)

#On crée les labels (0 = Bénin, 1 = Malin)
df_final['label'] = df_final['pathology'].map({
    'BENIGN': 0,
    'BENIGN_WITHOUT_CALLBACK': 0,
    'MALIGNANT': 1
})

#On supprime les échecs
df_final = df_final.dropna(subset=['vrai_chemin', 'label'])

print(f"\n {len(df_final)} images liées avec succès à leur diagnostic.")
display(df_final[['pathology', 'label', 'vrai_chemin']].head())

# %% cellule 5
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("⏳ Préparation des générateurs définitifs...")

# TensorFlow préfère que les étiquettes 0 et 1 soient du texte pour cette fonction
df_final['label'] = df_final['label'].astype(str)

# On coupe le tableau en deux (80% Train, 20% Validation/Test)
# "stratify" garantit qu'il y a le même pourcentage de cancers dans l'entraînement et dans le test
train_df, val_df = train_test_split(
    df_final,
    test_size=0.2,
    random_state=42,
    stratify=df_final['label']
)

# On crée l'outil qui va redimensionner et normaliser les images à la volée (pixels / 255)
generateur = ImageDataGenerator(rescale=1./255.)

# Le générateur d'entraînement
print("\n Générateur d'entraînement (Train) :")
train_gen = generateur.flow_from_dataframe(
    dataframe=train_df,
    x_col="vrai_chemin",    # La colonne avec les adresses des images
    y_col="label",          # La colonne avec le diagnostic (0 ou 1)
    target_size=(224, 224), # La taille pour l'IA
    color_mode="grayscale", # On force le noir et blanc
    class_mode="binary",    # Car on a seulement 2 choix (Bénin/Malin)
    batch_size=32           # On donne les images par paquets de 32
)

# Le générateur de test
print("\n Générateur de validation (Test) :")
val_gen = generateur.flow_from_dataframe(
    dataframe=val_df,
    x_col="vrai_chemin",
    y_col="label",
    target_size=(224, 224),
    color_mode="grayscale",
    class_mode="binary",
    batch_size=32
)

# %% cellule 6
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print(" Construction de l'architecture du réseau de neurones...")

# On crée un modèle "Séquentiel" (une pile de couches, de haut en bas)
modele = Sequential([
    # --- BLOC 1 : Les yeux de base ---
    # 32 filtres de 3x3 pixels. On précise la taille de l'image d'entrée (224x224 en noir et blanc "1")
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2, 2),

    # --- BLOC 2 : L'analyse des textures ---
    # 64 filtres, l'IA devient plus experte
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # --- BLOC 3 : L'analyse approfondie ---
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # --- BLOC DE DÉCISION ---
    # On aplatit la carte 2D des résultats en une simple liste 1D
    Flatten(),

    # Un "gros" cerveau de 128 neurones pour réfléchir aux indices trouvés
    Dense(128, activation='relu'),

    # Le "Dropout" est une astuce vitale : on désactive 50% des neurones au hasard à chaque tour.
    # Ça force l'IA à réfléchir globalement et l'empêche d'apprendre les images par cœur !
    Dropout(0.5),

    # LA SORTIE : 1 seul neurone.
    # 'sigmoid' le force à donner un résultat entre 0 (Bénin) et 1 (Malin).
    Dense(1, activation='sigmoid')
])

# On affiche le résumé de la construction
print("\n📋 Résumé du modèle :")
modele.summary()

# %% cellule 7

import tensorflow as tf
# On configure le cerveau avant l'entraînement
modele.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)


# On lance l'apprentissage
# 'epochs=10' signifie que l'IA va revoir l'intégralité des 1054 images 10 fois de suite.
historique = modele.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# %% cellule 8
import numpy as np
from tensorflow.keras.preprocessing import image

#On récupère une image au hasard dans le dossier de test (val_df)
# On prend la première ligne pour l'exemple
test_image_path = val_df['vrai_chemin'].iloc[0]
vrai_diagnostic = val_df['pathology'].iloc[0]

#On prépare l'image pour l'IA (exactement comme pour l'entraînement)
img_test = image.load_img(test_image_path, target_size=(224, 224), color_mode="grayscale")
img_array = image.img_to_array(img_test)
img_array = img_array / 255.0  # Normalisation (0 à 1)
img_array = np.expand_dims(img_array, axis=0)  # On ajoute une dimension pour dire "c'est un paquet de 1 image"

# L'IA donne sa prédiction
prediction = modele.predict(img_array)

# On affiche le résultat
score = prediction[0][0]
verdict = "MALIN (CANCER)" if score > 0.5 else "BÉNIN"

print(f"\n--- TEST DE L'IA ---")
print(f"Vrai diagnostic (Médecin) : {vrai_diagnostic}")
print(f"Prédiction de l'IA : {verdict} (Confiance : {score*100:.2f}%)")

# Affichage de l'image pour vérifier
import matplotlib.pyplot as plt
plt.imshow(img_test, cmap='gray')
plt.title(f"Verdict IA: {verdict}")
plt.axis('off')
plt.show()

# %% cellule 9
modele.save('cnn_cbis_ddsm_ayoub.keras')

# %% cellule 10
# --- PARTIE 2 : MODÈLE TABULAIRE (DONNÉES CLINIQUES) ---

# %% cellule 11
!pip install -q kagglehub
import kagglehub
import pandas as pd
import os

print("⏳ Téléchargement de la base de données Wisconsin...")

# On télécharge le dossier contenant le CSV
dossier_tabulaire = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
chemin_csv_wisconsin = os.path.join(dossier_tabulaire, "data.csv")

# On lit le tableau avec Pandas
df_clinique = pd.read_csv(chemin_csv_wisconsin)

print("✅ Données chargées avec succès !")
display(df_clinique.head())

# %% cellule 12
print("🧹 Nettoyage des données...")

# 1. On supprime la colonne 'id' et une colonne vide qui traîne souvent sur Kaggle ('Unnamed: 32')
if 'id' in df_clinique.columns:
    df_clinique = df_clinique.drop('id', axis=1)
if 'Unnamed: 32' in df_clinique.columns:
    df_clinique = df_clinique.drop('Unnamed: 32', axis=1)

# 2. On transforme le texte en chiffres pour l'algorithme
# M = Malin (Cancer) -> 1
# B = Bénin (Sain) -> 0
df_clinique['diagnosis'] = df_clinique['diagnosis'].map({'M': 1, 'B': 0})

print("📊 Répartition des patientes dans le tableau :")
print(df_clinique['diagnosis'].value_counts())
print("(0 = Bénin, 1 = Malin)")

# %% cellule 13
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# On sépare la "Cible" (le diagnostic) du reste des données (les mesures)
y_tabulaire = df_clinique['diagnosis']
X_tabulaire = df_clinique.drop('diagnosis', axis=1)

# On sépare en 80% Apprentissage et 20% Examen final
X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X_tabulaire, y_tabulaire, test_size=0.2, random_state=42, stratify=y_tabulaire
)

# ⚠️ ÉTAPE CRUCIALE : La normalisation des données chiffrées
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_tab)
X_test_scaled = scaler.transform(X_test_tab)

print(f"👉 Entraînement : {X_train_tab.shape[0]} patientes")
print(f"👉 Test : {X_test_tab.shape[0]} patientes")

# %% cellule 14
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

print("🧠 Entraînement du modèle Machine Learning (Random Forest)...")

# Création du modèle avec 100 "arbres de décision"
modele_clinique = RandomForestClassifier(n_estimators=100, random_state=42)
modele_clinique.fit(X_train_scaled, y_train_tab)

print("✅ Entraînement terminé !")

# --- EXAMEN FINAL ---
# predict_proba() est LA fonction la plus importante pour vous :
# Elle ne dit pas juste "Malade" ou "Sain", elle sort un pourcentage (ex: 85% de risque) !
probabilites_clinique = modele_clinique.predict_proba(X_test_scaled)[:, 1]
predictions_brutes = modele_clinique.predict(X_test_scaled)

precision = accuracy_score(y_test_tab, predictions_brutes)
auc_clinique = roc_auc_score(y_test_tab, probabilites_clinique)

print(f"\n🏆 RÉSULTATS DU MODÈLE TABULAIRE :")
print(f"➡️ Précision Globale (Accuracy) : {precision * 100:.2f}%")
print(f"➡️ Score AUC (À envoyer à Maxime !) : {auc_clinique:.4f}")

# %% cellule 15
import joblib

# On sauvegarde DEUX choses (le modèle ET l'outil de mise à l'échelle)
joblib.dump(modele_clinique, 'modele_tabulaire_wisconsin.pkl')
joblib.dump(scaler, 'scaler_wisconsin.pkl')

print("💾 Modèle Tabulaire et Scaler sauvegardés avec succès !")
print("Tous les ingrédients pour la Fusion Multimodale sont prêts.")

# %% cellule 16


# %% cellule 17
import numpy as np

print("🔬 Récupération d'un dossier patiente au hasard...")

# 1. On prend le dossier clinique de la toute première patiente de notre lot de test
patiente_test = X_test_tab.iloc[[0]]
vrai_diagnostic = y_test_tab.iloc[0] # Ce que le médecin avait vraiment conclu

# 2. ⚠️ L'ÉTAPE PIÈGE : La mise à l'échelle
# On doit utiliser l'outil 'scaler' pour que l'IA comprenne les chiffres
patiente_scaled = scaler.transform(patiente_test)

# 3. L'IA analyse le dossier et donne son pourcentage
# predict_proba renvoie deux chiffres : [Proba_Bénin, Proba_Malin]
probabilites = modele_clinique.predict_proba(patiente_scaled)
risque_malignite = probabilites[0][1] * 100

print("\n" + "="*40)
print("🩺 RÉSULTAT DE LA CONSULTATION IA")
print("="*40)

# On affiche la vérité
if vrai_diagnostic == 1:
    print("Vrai diagnostic (Médecin) : MALIN (Cancer) ⚠️")
else:
    print("Vrai diagnostic (Médecin) : BÉNIN (Sain) ✅")

print("-" * 40)

# On affiche la prédiction de l'IA
print(f"L'IA estime le risque de cancer à : {risque_malignite:.1f}%")

if risque_malignite > 50:
    print("Verdict de l'IA : Le modèle classe cette tumeur comme MALIGNE.")
else:
    print("Verdict de l'IA : Le modèle classe cette tumeur comme BÉNIGNE.")
