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

from src.audio.audio_utils import detect_language

def extract_audio(video_bytes):

    if video_bytes:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name
        
        # Extract audio
        video_clip = VideoFileClip(temp_file_path)
        audio_clip = video_clip.audio
        video_clip.close()  # Close the video clip
        return audio_clip
    


def transcribe_audio(audio_data):
    recognizer = sr.Recognizer()
    with io.BytesIO(audio_data) as source:
        source.seek(0)  # Reset the pointer to the beginning of the file-like object
        audio_data = recognizer.record(source)  # Grabamos el audio del archivo
        try:
            language = detect_language(audio_data)  # Make sure to define this function
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            print("No se pudo entender el audio")
            return ""
        except sr.RequestError as e:
            print(f"Error en la solicitud a Google Speech Recognition API: {e}")
            return ""