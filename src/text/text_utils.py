import os
import json

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