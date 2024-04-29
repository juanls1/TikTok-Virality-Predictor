import os
import json
import sys
from pathlib import Path

# Get the absolute path of the folder containing the module
root_dir = Path.cwd().resolve().parent.parent

# Add the folder path to sys.path
sys.path.append(str(root_dir))

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib 
import matplotlib.pyplot as plt
from langdetect import detect
from nltk.probability import FreqDist
matplotlib.use('TkAgg') 
from collections import Counter
from wordcloud import WordCloud
from nltk import FreqDist

from config.variables import language_map

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

# Function to detect the language of the text
def detect_language(text):
    try:
        language_code = detect(text)
        language = language_map.get(language_code, 'english')
        return language
    except:
        return "english"  # Set English as default language in case of error

# Function to clean the text
def clean_text(text):
    language = detect_language(text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove words that appear after a "#"
    cleaned_tokens = []
    skip_next = False
    for i, token in enumerate(tokens):
        if token == '#':
            skip_next = True
        elif skip_next:
            skip_next = False
        else:
            cleaned_tokens.append(token)
    tokens = cleaned_tokens
    # Remove punctuation and special characters
    tokens = [word for word in tokens if word.isalnum()]
    # Identify language to remove stopwords
    stop_words = set(stopwords.words(language))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join processed tokens into clean text
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Function to plot frequency distribution
def plot_freq_dist(words, num_words = 20):
    fdist = FreqDist(words)
    print(fdist)
    fdist.plot(num_words,cumulative=False)

# Function to perform exploratory analysis of the text
def analyze_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Count the most common words
    word_freq = Counter(tokens)
    common_words = word_freq.most_common(10)
    print("Most common words:", common_words)
    # Create a WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()