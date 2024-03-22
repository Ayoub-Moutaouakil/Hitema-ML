---
title: Review To Sentiment
emoji: 🌖
colorFrom: yellow
colorTo: purple
sdk: streamlit
sdk_version: 1.32.2
app_file: app.py
pinned: false
---

Analyse Multimodale

Ce projet consiste en une application Streamlit pour l'analyse multimodale, qui comprend l'analyse de sentiment de critiques de films, la détection de commandes vocales à partir de fichiers audio et la prédiction de la race d'un chien à partir d'une image.

Instructions d'installation

Assurez-vous d'avoir Python installé sur votre système. Si ce n'est pas le cas, vous pouvez le télécharger et l'installer à partir de python.org.

Clonez ce dépôt GitHub sur votre machine locale en utilisant la commande suivante :

git clone https://github.com/votre_utilisateur/analyse-multimodale.git

Accédez au répertoire du projet et installez les dépendances en exécutant la commande suivante :

pip install -r requirements.txt

Instructions d'utilisation

Une fois les dépendances installées, lancez l'application Streamlit en exécutant la commande suivante :

streamlit run app.py

L'application devrait s'ouvrir dans votre navigateur par défaut.

Vous pouvez maintenant utiliser les fonctionnalités suivantes :

Analyse de sentiment de critiques de films : Entrez une critique de film dans la zone de texte et cliquez sur le bouton "Analyser le sentiment" pour obtenir une analyse du sentiment (positif ou négatif).

Détection de commandes vocales : Uploadez un fichier audio au format WAV ou MP3, puis cliquez sur "Uploader un fichier audio...". L'audio sera lu et une commande vocale détectée sera affichée.

Détection de la race de chien : Uploadez une image de chien au format JPG, JPEG ou PNG, puis cliquez sur "Uploader une image de chien...". L'image sera affichée, et en cliquant sur "Prédire la race", la race du chien sera prédite et affichée.

Remarques

Assurez-vous que les fichiers audio uploadés pour la détection de commandes vocales sont clairs et de bonne qualité pour de meilleurs résultats.

L'analyse de sentiment utilise un modèle BERT pré-entraîné, la détection de commandes vocales utilise un modèle Wav2Vec2 pré-entraîné, et la détection de race de chien utilise un modèle MobileNetV2 pré-entraîné.

Ce projet est à titre éducatif et démonstratif, et peut être étendu ou adapté selon les besoins spécifiques.
