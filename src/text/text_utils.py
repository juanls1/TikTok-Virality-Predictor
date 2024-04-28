import os
import json

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import matplotlib
from langdetect import detect
from nltk.probability import FreqDist
matplotlib.use('TkAgg') 

# Función para cargar el texto de las transcripciones de los vídeos
def load_transcriptions(folder_path):
    transcriptions = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            video_id = filename.split('_')[0]  # Extraer el ID del vídeo
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                transcription = file.read()
                transcriptions[video_id] = transcription
    return transcriptions

# Función para cargar la información del archivo JSON
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        return "en"  # Set a default language in case of error

# Function to clean the text
def clean_text(text):
    language = detect_language(text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation and special characters
    tokens = [word for word in tokens if word.isalnum()]
    # Identify language to remove stopwords
    stop_words = set(stopwords.words(language))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join processed tokens into clean text
    cleaned_text = ' '.join(tokens)
    return cleaned_text


def plot_freq_dist(words, num_words = 20):
    fdist = FreqDist(words)
    print(fdist)
    fdist.plot(num_words,cumulative=False)