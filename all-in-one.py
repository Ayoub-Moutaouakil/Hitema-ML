import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForCTC

# Chargement du modèle et du tokenizer pour l'analyse de sentiment
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Chargement du modèle pré-entraîné Wav2Vec2 pour la détection de commandes vocales
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
voice_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Chargement du modèle pré-entraîné MobileNetV2 pour la détection de race de chien
dog_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Fonction pour prédire le sentiment
def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = sentiment_model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    return predicted_class

# Fonction pour détecter les commandes vocales
def detect_commands(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file, normalize=True)
    
    # Rééchantillonnage de l'audio à 16 kHz si nécessaire
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Vérifier et ajuster les dimensions de l'audio
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)  # Squeeze si l'audio est en format batched
    elif waveform.dim() == 3:
        waveform = waveform.squeeze(1)  # Squeeze si l'audio est en format channel first
    
    # Obtenir une estimation de la longueur maximale attendue par le modèle
    max_waveform_length = processor.feature_extractor.feature_size
    
    # Vérifier si la longueur de l'audio est compatible avec le modèle
    if waveform.size(0) > max_waveform_length:
        st.warning(f"La longueur de l'audio est supérieure à {max_waveform_length}, il peut y avoir une troncature.")

    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = voice_model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

# Fonction pour prétraiter l'image pour la détection de race de chien
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Interface Streamlit
st.title("Analyse multimodale")

# Analyse de sentiment
st.subheader("Analyse de sentiment de critiques de films")
user_input = st.text_input("Entrez votre critique de film :")
if st.button("Analyser le sentiment"):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)
        if sentiment == 1:
            st.error("La critique est négative!")
        else:
            st.success("La critique est positive!")
    else:
        st.warning("Veuillez entrer une critique de film.")

# Détection de commandes vocales
st.subheader("Détection de commandes vocales")
uploaded_file_voice = st.file_uploader("Uploader un fichier audio...", type=["wav", "mp3"])
if uploaded_file_voice is not None:
    st.audio(uploaded_file_voice, format="audio/wav")
    prediction_voice = detect_commands(uploaded_file_voice)
    st.write("Commande détectée:", prediction_voice)

# Détection de la race de chien
st.subheader("Détection de la race de chien")
uploaded_file_dog = st.file_uploader("Uploader une image de chien...", type=["jpg", "jpeg", "png"])
if uploaded_file_dog is not None:
    image = Image.open(uploaded_file_dog)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    if st.button("Prédire la race"):
        image = image.resize((224, 224))
        image = np.array(image)
        image = preprocess_image(image)
        image = np.expand_dims(image, axis=0)
        preds = dog_model.predict(image)
        predicted_class = np.argmax(preds)
        labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                              'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        with open(labels_path) as f:
            labels = f.readlines()
        st.success(f"La race du chien est prédite comme : {labels[predicted_class]}")
