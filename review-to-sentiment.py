import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Chargement du modèle et du tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Fonction pour prédire la sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    return predicted_class

# Interface utilisateur avec Streamlit
st.title("Analyse de sentiment de critiques de films")

# Zone de texte pour saisir la critique
user_input = st.text_input("Entrez votre critique de film :")

# Bouton pour lancer la prédiction
if st.button("Analyser"):
    if user_input.strip() != "":
        # Prédiction du sentiment
        sentiment = predict_sentiment(user_input)
        if sentiment == 1:
            st.error("La critique est négative!")
        else:
            st.success("La critique est positive!")
    else:
        st.warning("Veuillez entrer une critique de film.")