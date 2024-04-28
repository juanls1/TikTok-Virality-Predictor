import sys
from pathlib import Path
import os

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path.cwd().resolve().parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

import subprocess
from langdetect import detect
import speech_recognition as sr

from config.variables import ffmpeg_path

def convert_to_wav(mp3_file, output_folder):
    wav_file = os.path.splitext(os.path.basename(mp3_file))[0] + '.wav'
    wav_file_path = os.path.join(output_folder, wav_file)
    subprocess.run([os.path.join(root_dir, ffmpeg_path), '-i', mp3_file, '-ac', '1', '-ar', '16000', wav_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_file_path

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)  # Grabamos el audio del archivo
        try:
            language = detect_language(audio_file)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            print("No se pudo entender el audio")
            return ""
        except sr.RequestError as e:
            print(f"Error en la solicitud a Google Speech Recognition API: {e}")
            return ""
        

def detect_language(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            language = detect(recognizer.recognize_google(audio_data, show_all=True))
            return language
        except:
            return "en-EN"  # Establece un idioma predeterminado en caso de error