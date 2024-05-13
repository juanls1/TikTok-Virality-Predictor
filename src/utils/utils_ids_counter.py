import json

import sys
from pathlib import Path
import os

# Get the absolute path of the folder containing the module
root_dir = Path(__file__).resolve().parent.parent.parent

# Add the folder path to sys.path
sys.path.append(str(root_dir))

def contar_ids_tiktok(filename):
    # List to store unique IDs
    ids = []

    # Load the JSON file
    with open(filename, "r") as json_file:
        tiktok_info = json.load(json_file)

    # Iterate over each TikTok and get its ID
    for tiktok_id in tiktok_info:
        ids.append(tiktok_id)

    # Count unique IDs
    unique_ids_count = len(set(ids))

    return unique_ids_count

# JSON file name
filename = os.path.join(root_dir,"data\\inputs\\tiktok_info.json")

# Call the function to count unique IDs
unique_ids_count = contar_ids_tiktok(filename)
print("Number of unique IDs in tiktok_info.json:", unique_ids_count)
