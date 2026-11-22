# Script conversion of the cells of the Colab notebook Projet_AI_E3E.ipynb
# Outputs and images removed.

# %% cellule 1
# --- PART 1: CNN MODEL (MAMMOGRAPHY IMAGES) ---

# %% cellule 2
# Install the required library
!pip install kagglehub

import cv2
import numpy as np
import kagglehub
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# Downloading the database: CBIS-DDSM: Breast Cancer Image Dataset
path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")

# %% cellule 3
import pandas as pd
import os

# Target the training CSV file for tumor masses
chemin_csv_masses = os.path.join(path, "csv", "mass_case_description_train_set.csv")

# Ask Pandas to read this file
df_diagnostic = pd.read_csv(chemin_csv_masses)

print("Here is what the real medical file looks like:")
# Display the first 5 rows of the table
display(df_diagnostic.head())

# %% cellule 4
import os
import glob
import pandas as pd

#List ALL the jpg images on the disk
toutes_les_images = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)

#Create a dictionary: { 'LONG_UID_NUMBER' : 'IMAGE_PATH' }
dict_images_disque = {}
for img in toutes_les_images:
    # The UID is the name of the folder just before the image
    uid = os.path.basename(os.path.dirname(img))
    dict_images_disque[uid] = img

#The brute-force function
def match_infaillible(csv_path):
    csv_path = str(csv_path) # Make sure it is text
    # For each UID existing on our hard drive...
    for uid, chemin_img in dict_images_disque.items():
        # ... check whether it is hidden inside the CSV text
        if uid in csv_path:
            return chemin_img
    return None # If nothing is found

#Apply it to our table
df_final = df_diagnostic.copy()
df_final['vrai_chemin'] = df_final['image file path'].apply(match_infaillible)

#Create the labels (0 = Benign, 1 = Malignant)
df_final['label'] = df_final['pathology'].map({
    'BENIGN': 0,
    'BENIGN_WITHOUT_CALLBACK': 0,
    'MALIGNANT': 1
})

#Remove the failures
df_final = df_final.dropna(subset=['vrai_chemin', 'label'])

print(f"\n {len(df_final)} images successfully linked to their diagnosis.")
display(df_final[['pathology', 'label', 'vrai_chemin']].head())

# %% cellule 5
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("⏳ Preparing the final generators...")

# TensorFlow prefers the 0 and 1 labels to be text for this function
df_final['label'] = df_final['label'].astype(str)

# Split the table in two (80% Train, 20% Validation/Test)
# "stratify" guarantees the same percentage of cancers in training and in test
train_df, val_df = train_test_split(
    df_final,
    test_size=0.2,
    random_state=42,
    stratify=df_final['label']
)

# Create the tool that resizes and normalizes the images on the fly (pixels / 255)
generateur = ImageDataGenerator(rescale=1./255.)

# The training generator
print("\n Training generator (Train):")
train_gen = generateur.flow_from_dataframe(
    dataframe=train_df,
    x_col="vrai_chemin",    # The column with the image paths
    y_col="label",          # The column with the diagnosis (0 or 1)
    target_size=(224, 224), # The size for the AI
    color_mode="grayscale", # Force black and white
    class_mode="binary",    # Because there are only 2 choices (Benign/Malignant)
    batch_size=32           # Feed the images in batches of 32
)

# The test generator
print("\n Validation generator (Test):")
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

print(" Building the neural network architecture...")

# Create a "Sequential" model (a stack of layers, from top to bottom)
modele = Sequential([
    # --- BLOCK 1: The basic eyes ---
    # 32 filters of 3x3 pixels. Specify the input image size (224x224 in black and white "1")
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2, 2),

    # --- BLOCK 2: Texture analysis ---
    # 64 filtres, l'IA devient plus experte
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # --- BLOC 3 : L'analyse approfondie ---
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # --- DECISION BLOCK ---
    # Flatten the 2D map of results into a simple 1D list
    Flatten(),

    # A "big" brain of 128 neurons to reason about the clues found
    Dense(128, activation='relu'),

    # "Dropout" is a vital trick: 50% of the neurons are randomly turned off at each step.
    # It forces the AI to reason globally and prevents it from memorizing the images!
    Dropout(0.5),

    # THE OUTPUT: a single neuron.
    # 'sigmoid' forces it to output a value between 0 (Benign) and 1 (Malignant).
    Dense(1, activation='sigmoid')
])

# Display the build summary
print("\n📋 Model summary:")
modele.summary()

# %% cellule 7

import tensorflow as tf
# Configure the brain before training
modele.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)


# On lance l'apprentissage
# 'epochs=10' means the AI will go through all 1054 images 10 times in a row.
historique = modele.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# %% cellule 8
import numpy as np
from tensorflow.keras.preprocessing import image

#Pick a random image from the test set (val_df)
# Take the first row as an example
test_image_path = val_df['vrai_chemin'].iloc[0]
vrai_diagnostic = val_df['pathology'].iloc[0]

#Prepare the image for the AI (exactly as for training)
img_test = image.load_img(test_image_path, target_size=(224, 224), color_mode="grayscale")
img_array = image.img_to_array(img_test)
img_array = img_array / 255.0  # Normalization (0 to 1)
img_array = np.expand_dims(img_array, axis=0)  # Add a dimension to say "this is a batch of 1 image"

# The AI gives its prediction
prediction = modele.predict(img_array)

# Display the result
score = prediction[0][0]
verdict = "MALIGNANT (CANCER)" if score > 0.5 else "BENIGN"

print(f"\n--- AI TEST ---")
print(f"True diagnosis (Doctor): {vrai_diagnostic}")
print(f"AI prediction: {verdict} (Confidence: {score*100:.2f}%)")

# Display the image to check
import matplotlib.pyplot as plt
plt.imshow(img_test, cmap='gray')
plt.title(f"Verdict IA: {verdict}")
plt.axis('off')
plt.show()

# %% cellule 9
modele.save('cnn_cbis_ddsm_ayoub.keras')

# %% cellule 10
# --- PART 2: TABULAR MODEL (CLINICAL DATA) ---

# %% cellule 11
!pip install -q kagglehub
import kagglehub
import pandas as pd
import os

print("⏳ Downloading the Wisconsin database...")

# Download the folder containing the CSV
dossier_tabulaire = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
chemin_csv_wisconsin = os.path.join(dossier_tabulaire, "data.csv")

# Read the table with Pandas
df_clinique = pd.read_csv(chemin_csv_wisconsin)

print("✅ Data loaded successfully!")
display(df_clinique.head())

# %% cellule 12
print("🧹 Cleaning the data...")

# 1. Drop the 'id' column and an empty column often found on Kaggle ('Unnamed: 32')
if 'id' in df_clinique.columns:
    df_clinique = df_clinique.drop('id', axis=1)
if 'Unnamed: 32' in df_clinique.columns:
    df_clinique = df_clinique.drop('Unnamed: 32', axis=1)

# 2. Convert the text to numbers for the algorithm
# M = Malin (Cancer) -> 1
# B = Benign (Healthy) -> 0
df_clinique['diagnosis'] = df_clinique['diagnosis'].map({'M': 1, 'B': 0})

print("📊 Distribution of the patients in the table:")
print(df_clinique['diagnosis'].value_counts())
print("(0 = Benign, 1 = Malignant)")

# %% cellule 13
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate the "Target" (the diagnosis) from the rest of the data (the measurements)
y_tabulaire = df_clinique['diagnosis']
X_tabulaire = df_clinique.drop('diagnosis', axis=1)

# Split into 80% Training and 20% Final exam
X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X_tabulaire, y_tabulaire, test_size=0.2, random_state=42, stratify=y_tabulaire
)

# ⚠️ CRUCIAL STEP: Normalization of the numerical data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_tab)
X_test_scaled = scaler.transform(X_test_tab)

print(f"👉 Training: {X_train_tab.shape[0]} patients")
print(f"👉 Test : {X_test_tab.shape[0]} patientes")

# %% cellule 14
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

print("🧠 Training the Machine Learning model (Random Forest)...")

# Create the model with 100 "decision trees"
modele_clinique = RandomForestClassifier(n_estimators=100, random_state=42)
modele_clinique.fit(X_train_scaled, y_train_tab)

print("✅ Training finished!")

# --- EXAMEN FINAL ---
# predict_proba() is THE most important function here:
# It does not just say "Sick" or "Healthy", it outputs a percentage (e.g. 85% risk)!
probabilites_clinique = modele_clinique.predict_proba(X_test_scaled)[:, 1]
predictions_brutes = modele_clinique.predict(X_test_scaled)

precision = accuracy_score(y_test_tab, predictions_brutes)
auc_clinique = roc_auc_score(y_test_tab, probabilites_clinique)

print(f"\n🏆 TABULAR MODEL RESULTS:")
print(f"➡️ Overall accuracy: {precision * 100:.2f}%")
print(f"➡️ AUC score: {auc_clinique:.4f}")

# %% cellule 15
import joblib

# Save TWO things (the model AND the scaling tool)
joblib.dump(modele_clinique, 'modele_tabulaire_wisconsin.pkl')
joblib.dump(scaler, 'scaler_wisconsin.pkl')

print("💾 Tabular model and scaler saved successfully!")
print("All the ingredients for the Multimodal Fusion are ready.")

# %% cellule 16


# %% cellule 17
import numpy as np

print("🔬 Retrieving a random patient record...")

# 1. Take the clinical record of the very first patient in our test batch
patiente_test = X_test_tab.iloc[[0]]
vrai_diagnostic = y_test_tab.iloc[0] # What the doctor actually concluded

# 2. ⚠️ THE TRAP STEP: Scaling
# The 'scaler' tool must be used so the AI understands the numbers
patiente_scaled = scaler.transform(patiente_test)

# 3. The AI analyzes the record and gives its percentage
# predict_proba returns two numbers: [Proba_Benign, Proba_Malignant]
probabilites = modele_clinique.predict_proba(patiente_scaled)
risque_malignite = probabilites[0][1] * 100

print("\n" + "="*40)
print("🩺 AI CONSULTATION RESULT")
print("="*40)

# Display the ground truth
if vrai_diagnostic == 1:
    print("True diagnosis (Doctor): MALIGNANT (Cancer) ⚠️")
else:
    print("True diagnosis (Doctor): BENIGN (Healthy) ✅")

print("-" * 40)

# Display the AI prediction
print(f"The AI estimates the cancer risk at: {risque_malignite:.1f}%")

if risque_malignite > 50:
    print("AI verdict: the model classifies this tumor as MALIGNANT.")
else:
    print("AI verdict: the model classifies this tumor as BENIGN.")
