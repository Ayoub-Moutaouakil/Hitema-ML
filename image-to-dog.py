import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

# Fonction pour prétraiter l'image
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Charger le modèle pré-entraîné
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Titre de l'application
st.title("Détection de la race de chien")

# Téléchargement de l'image
st.write("Téléchargez une image de chien pour prédire sa race :")
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

# Affichage de l'image et prédiction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Classification de l'image
    if st.button("Prédire la race"):
        # Prétraitement de l'image
        image = image.resize((224, 224))  # Redimensionner l'image selon les besoins du modèle
        image = np.array(image)  # Convertir l'image en tableau numpy
        image = preprocess_image(image)  # Prétraiter l'image pour l'entrée du modèle

        # Effectuer la prédiction en utilisant le modèle
        image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
        preds = model.predict(image)
        predicted_class = np.argmax(preds)

        # Charger les étiquettes de classes (par exemple, pour ImageNet)
        labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                              'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        with open(labels_path) as f:
            labels = f.readlines()

        # Afficher le résultat de la prédiction
        st.success(f"La race du chien est prédite comme : {labels[predicted_class]}")