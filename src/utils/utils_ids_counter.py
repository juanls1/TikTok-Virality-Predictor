import json

import sys
from pathlib import Path
import os

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

def contar_ids_tiktok(filename):
    # Lista para almacenar los IDs únicos
    ids = []

    # Cargar el archivo JSON
    with open(filename, "r") as json_file:
        tiktok_info = json.load(json_file)

    # Iterar sobre cada TikTok y obtener su ID
    for tiktok_id in tiktok_info:
        ids.append(tiktok_id)

    # Contar los IDs únicos
    ids_unicos = len(set(ids))

    return ids_unicos

# Nombre del archivo JSON
filename = os.path.join(root_dir,"data\\inputs\\tiktok_info.json")

# Llamar a la función para contar los IDs únicos
cantidad_ids = contar_ids_tiktok(filename)
print("Número de IDs únicos en tiktok_info.json:", cantidad_ids)