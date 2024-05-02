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
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def get_bag_of_words(sentences):
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(sentences)
    print(bag_of_words.todense())
    print(vectorizer.vocabulary_)

def perform_topic_modeling(video_data):
    topics_per_video = {}
    for video_id, info in video_data.items():
        cleaned_data = []
        if info['clean_transcription']:
            cleaned_data.append(info['clean_transcription'])
        if info['clean_text']:
            cleaned_data.append(info['clean_text'])
        cleaned_data.extend(info['hashtags'])
        
        # Perform topic modeling if there is cleaned data available
        if cleaned_data:
            topics_per_video[video_id] = perform_topic_modeling_single(cleaned_data)
        else:
            topics_per_video[video_id] = "No cleaned data available"
    return topics_per_video

def perform_topic_modeling_single(cleaned_data):
    dictionary = corpora.Dictionary([cleaned_data])
    corpus = [dictionary.doc2bow(doc) for doc in [cleaned_data]]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=50)
    return lda_model.print_topics(num_topics=3, num_words=2)


def analyze_sentiment_text(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment

def analyze_sentiment_vader(text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_score = sentiment_analyzer.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_score['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment

def classify_sentiment(video_data):
    for video_id, info in video_data.items():
        text = info.get('clean_text', '')  # Use clean text if available
        if not text:
            text = info.get('text', '')  # Otherwise, use raw text
        transcript = info.get('clean_transcription', '')  # Use clean transcript if available
        if not transcript:
            transcript = info.get('transcription', '')  # Otherwise, use raw transcript

        # Initialize variables for sentiment analysis results
        text_sentiment_blob = None
        text_sentiment_vader = None
        transcript_sentiment_blob = None
        transcript_sentiment_vader = None
        
        # Analyze sentiment for text using TextBlob
        if text:
            text_sentiment_blob = analyze_sentiment_text(text)
            text_sentiment_vader = analyze_sentiment_vader(text)
        
        # Analyze sentiment for transcript using VADER
        if transcript:
            transcript_sentiment_blob = analyze_sentiment_text(transcript)
            transcript_sentiment_vader = analyze_sentiment_vader(transcript)
        
        # Combine sentiments from TextBlob and VADER
        if text_sentiment_blob and text_sentiment_vader:
            if text_sentiment_blob == text_sentiment_vader:
                # If both sentiments agree, use the agreed sentiment
                video_data[video_id]['text_sentiment'] = text_sentiment_blob
            else:
                # If sentiments disagree, take the average sentiment score
                text_polarity = TextBlob(text).sentiment.polarity
                text_compound = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
                avg_sentiment = (text_polarity + text_compound) / 2
                if avg_sentiment > 0:
                    video_data[video_id]['text_sentiment'] = "Positive"
                elif avg_sentiment < 0:
                    video_data[video_id]['text_sentiment'] = "Negative"
                else:
                    video_data[video_id]['text_sentiment'] = "Neutral"
        elif text_sentiment_blob:
            video_data[video_id]['text_sentiment'] = text_sentiment_blob
        elif text_sentiment_vader:
            video_data[video_id]['text_sentiment'] = text_sentiment_vader
        else:
            # If both text is empty, mark sentiment as Unknown
            video_data[video_id]['text_sentiment'] = "Unknown"

        # Combine sentiments from TextBlob and VADER
        if transcript_sentiment_blob and transcript_sentiment_vader:
            if transcript_sentiment_blob == transcript_sentiment_vader:
                # If both sentiments agree, use the agreed sentiment
                video_data[video_id]['transcript_sentiment'] = transcript_sentiment_blob
            else:
                # If sentiments disagree, take the average sentiment score
                transcript_polarity = TextBlob(transcript).sentiment.polarity
                transcript_compound = SentimentIntensityAnalyzer().polarity_scores(transcript)['compound']
                avg_sentiment = (transcript_polarity + transcript_compound) / 2
                if avg_sentiment > 0:
                    video_data[video_id]['transcript_sentiment'] = "Positive"
                elif avg_sentiment < 0:
                    video_data[video_id]['transcript_sentiment'] = "Negative"
                else:
                    video_data[video_id]['transcript_sentiment'] = "Neutral"
        elif transcript_sentiment_blob:
            video_data[video_id]['transcript_sentiment'] = transcript_sentiment_blob
        elif transcript_sentiment_vader:
            video_data[video_id]['transcript_sentiment'] = transcript_sentiment_vader
        else:
            # If both transcript is empty, mark sentiment as Unknown
            video_data[video_id]['transcript_sentiment'] = "Unknown"


def count_sentiments(data):

    for analyzer in ['blob', 'vader', 'both']:
        # Initialize counters for each sentiment category
        sentiment_counts_text = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Unknown': 0}
        sentiment_counts_transcription = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Unknown': 0}
        
        # Count sentiments for text and transcription
        for info in data.values():

            if analyzer == 'both':
                text_sentiment = info.get('text_sentiment', '')
                if text_sentiment in sentiment_counts_text:
                    sentiment_counts_text[text_sentiment] += 1
                
                transcription_sentiment = info.get('transcript_sentiment', '')
                if transcription_sentiment in sentiment_counts_transcription:
                    sentiment_counts_transcription[transcription_sentiment] += 1
                
            else:
                text = info.get('clean_text', '')  # Use clean text if available
                if not text:
                    text = info.get('text', '')  # Otherwise, use raw text
                transcript = info.get('clean_transcription', '')  # Use clean transcript if available
                if not transcript:
                    transcript = info.get('transcription', '')  # Otherwise, use raw transcript

                # Analyze sentiment based on the selected analyzer
                if analyzer == 'blob':
                    sentiment_text = analyze_sentiment_text(text)
                    if sentiment_text in sentiment_counts_text:
                        sentiment_counts_text[sentiment_text] += 1
                    
                    sentiment_transcript = analyze_sentiment_text(transcript)
                    if sentiment_transcript in sentiment_counts_transcription:
                        sentiment_counts_transcription[sentiment_transcript] += 1
                        
                elif analyzer == 'vader':
                    sentiment_text = analyze_sentiment_vader(text)
                    if sentiment_text in sentiment_counts_text:
                        sentiment_counts_text[sentiment_text] += 1
                    
                    sentiment_transcript = analyze_sentiment_vader(transcript)
                    if sentiment_transcript in sentiment_counts_transcription:
                        sentiment_counts_transcription[sentiment_transcript] += 1
        
        print(f"Sentiment Counts for Text and Transcriptions ({analyzer.upper()}):\n")
        print("{:<10} {:<10} {:<10}".format('Sentiment', 'Text', 'Transcription'))
        for sentiment in sentiment_counts_text:
            text_count = sentiment_counts_text[sentiment]
            transcription_count = sentiment_counts_transcription[sentiment]
            print("{:<10} {:<10} {:<10}".format(sentiment, text_count, transcription_count))
        print('\n')

# Metrics
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mse, mae

# Concatenate the embeddings of each observation into a single vector
def concatenate_embeddings(row, df):
    return np.concatenate([np.array(row[col]) for col in df.columns])