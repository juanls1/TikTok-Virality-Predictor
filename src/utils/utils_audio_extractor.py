import os
from moviepy.editor import VideoFileClip
import sys
from pathlib import Path

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

def extract_audios():

    # Crear la carpeta "audios" dentro de la carpeta comprimida si no existe
    audios_dir = os.path.join(root_dir, "data\\inputs\\audios")
    if not os.path.exists(audios_dir):
        os.makedirs(audios_dir)

    # Iterar sobre los archivos de video y extraer los audios
    for video_file in os.listdir(os.path.join(root_dir, "data\\inputs\\Raw\\videos")):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(root_dir, "data\\inputs\\Raw\\videos", video_file)
            audio_path = os.path.join(audios_dir, video_file.replace(".mp4", "_audio.mp3"))
            print(f"Extrayendo audio de {video_file}")
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path)
            audio_clip.close()
            video_clip.close()

# Llamar a la función para extraer los audios
extract_audios()