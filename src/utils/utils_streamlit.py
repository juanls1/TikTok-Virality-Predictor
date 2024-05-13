from moviepy.editor import VideoFileClip
import tempfile
import io
import speech_recognition as sr

import sys
from pathlib import Path

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path.cwd().resolve().parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))



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
        
        # Read the audio file and return the data
        with open(audio_tempfile.name, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        return audio_data, audio_tempfile.name
    

def clean_hashtags(hashtags):
    if hashtags:
        # Clean the hashtags removing also the # symbol
        clean_hashtags = [hashtag.replace("#", "").strip(" ") for hashtag in hashtags.split(",")]
        return clean_hashtags
    else:
        return []
