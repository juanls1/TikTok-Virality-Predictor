from moviepy.editor import VideoFileClip
import tempfile
import io
import os
import pandas as pd
import torch
import torch.nn as nn
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import load_model
import cv2

import sys
from pathlib import Path

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path.cwd().resolve().parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from config.variables import model_paths
from src.text.text_utils import CustomDataset
from src.audio.audio_utils import load_audio_features

def extract_audio(video_bytes):
    if video_bytes:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name
        
        # Extract audio
        video_clip = VideoFileClip(temp_file_path)
        audio_clip = video_clip.audio
        audio_tempfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_clip.write_audiofile(audio_tempfile.name)
        audio_clip.close()  # Close the audio clip
        
        return audio_tempfile.name
    

def clean_hashtags(hashtags):
    if hashtags:
        # Clean the hashtags removing also the # symbol
        clean_hashtags = [hashtag.replace("#", "").strip(" ") for hashtag in hashtags.split(",")]
        return clean_hashtags
    else:
        return []
    
    
def create_text_prediction(text):
    
    text_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
    config = AutoConfig.from_pretrained("xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    text_model.config.num_labels = 1  
    text_model.lm_head.decoder = nn.Linear(text_model.config.hidden_size, 1)  
    
    text_model.load_state_dict(torch.load(os.path.join(root_dir, model_paths["text_model"])))
    
    df = pd.DataFrame({'text': [text], 'virality': [0]})

    dataset = CustomDataset(df['text'], df['virality'], tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Set the model to evaluation mode
    text_model.eval()

    # Evaluation loop
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.mean(dim=1).squeeze(-1)
            
    return float(logits[0])


def create_audio_prediction(audio_wav):
    model = load_model(os.path.join(root_dir, model_paths["audio_model"]))
    
    features = load_audio_features(audio_wav)
    
    tensor_input = torch.tensor([features]).unsqueeze(2)
    
    prediction = model.predict(tensor_input)
    
    return float(prediction[0])
    
    
def extract_frames(video_bytes, n_frames=8):
    if video_bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()  # Asegurar que todos los bytes están escritos
            temp_file_path = temp_file.name
        
        cap = cv2.VideoCapture(temp_file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = [round(i) for i in np.linspace(0, frame_count - 1, n_frames)]
        
        frames = []  # Lista para almacenar los frames extraídos
        success_count = 0
        for frame_id in frame_ids:
            frame, success, id = try_capture_frame(cap, frame_id)
            if success:
                frame = cv2.resize(frame, (224, 224))  # Redimensionar a 224x224 para el modelo CNN
                frames.append(frame)  # Agregar el frame redimensionado a la lista
                success_count += 1
        
        cap.release()
        os.unlink(temp_file_path)  # Eliminar el archivo temporal
        
        return frames

def try_capture_frame(cap, frame_id):
    while frame_id >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            return frame, True, frame_id
        print(f"Error capturing frame: {frame_id}, trying previous frame")
        frame_id -= 1
    print("No valid frames available to capture.")
    return None, False, -1