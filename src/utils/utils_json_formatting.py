import json
import pandas as pd
import sys
from pathlib import Path
import os

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

def extract_tiktok_info():
    # Cargar el JSON de los TikToks
    with open(os.path.join(root_dir,"data\\inputs\\Raw\\trending.json"), "r", encoding="utf-8") as json_file:
        tiktok_data = json.load(json_file)

    # Crear un diccionario para almacenar la información de los TikToks
    tiktok_info = {}

    # Iterar sobre los datos de TikToks y agregar la información relevante al diccionario
    for tiktok in tiktok_data["collector"]:
        tiktok_id = tiktok["id"]
        tiktok_info[tiktok_id] = {}

        # Obtener métricas
        metrics = {
            "likes": tiktok["diggCount"],
            "shares": tiktok["shareCount"],
            "comments": tiktok["commentCount"],
            "views": tiktok["playCount"],
            "saved": "Not supported"  # Esta información no está disponible en el JSON proporcionado
        }
        tiktok_info[tiktok_id]["metrics"] = metrics

        # Obtener información del autor
        author_info = {
            "id": tiktok["authorMeta"]["id"],
            "username": tiktok["authorMeta"]["nickName"],
            "verified": tiktok["authorMeta"]["verified"]
        }
        tiktok_info[tiktok_id]["author"] = author_info

        # Obtener información general del TikTok
        general_info = {
            "text": tiktok["text"],
            "hashtags": [hashtag["name"] for hashtag in tiktok["hashtags"]],
            "music_id": tiktok["musicMeta"]["musicId"]
        }
        tiktok_info[tiktok_id]["general"] = general_info

        # Obtener información de la música
        music_info = {
            "music_name": tiktok["musicMeta"]["musicName"],
            "music_author": tiktok["musicMeta"]["musicAuthor"],
            "music_original": tiktok["musicMeta"]["musicOriginal"]
        }
        tiktok_info[tiktok_id]["music"] = music_info

        # Agregar la URL del video
        tiktok_info[tiktok_id]["video_url"] = tiktok["videoUrl"]

    # Guardar la lista de información de TikToks en un archivo JSON
    with open(os.path.join(root_dir,"data\\inputs\\tiktok_info.json"), "w") as json_output:
        json.dump(tiktok_info, json_output, indent=4)

# Llamar a la función para extraer la información de los TikToks
extract_tiktok_info()