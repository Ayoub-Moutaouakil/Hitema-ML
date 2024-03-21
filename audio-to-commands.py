import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Chargement du modèle pré-entraîné Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

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
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

# Interface Streamlit
st.title("Détection de commandes vocales")

uploaded_file = st.file_uploader("Uploader un fichier audio...", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    prediction = detect_commands(uploaded_file)
    st.write("Commande détectée:", prediction)
