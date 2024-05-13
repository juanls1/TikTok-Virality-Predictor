import json
import pandas as pd
import sys
from pathlib import Path
import os

# Get the absolute path of the folder containing the module
root_dir = Path(__file__).resolve().parent.parent.parent

# Add the folder path to sys.path
sys.path.append(str(root_dir))

def extract_tiktok_info():
    # Load the TikTok JSON data
    with open(os.path.join(root_dir,"data\\inputs\\Raw\\trending.json"), "r", encoding="utf-8") as json_file:
        tiktok_data = json.load(json_file)

    # Create a dictionary to store TikTok information
    tiktok_info = {}

    # Iterate over TikTok data and add relevant information to the dictionary
    for tiktok in tiktok_data["collector"]:
        tiktok_id = tiktok["id"]
        tiktok_info[tiktok_id] = {}

        # Get metrics
        metrics = {
            "likes": tiktok["diggCount"],
            "shares": tiktok["shareCount"],
            "comments": tiktok["commentCount"],
            "views": tiktok["playCount"],
            "saved": "Not supported"  # This information is not available in the provided JSON
        }
        tiktok_info[tiktok_id]["metrics"] = metrics

        # Get author information
        author_info = {
            "id": tiktok["authorMeta"]["id"],
            "username": tiktok["authorMeta"]["nickName"],
            "verified": tiktok["authorMeta"]["verified"]
        }
        tiktok_info[tiktok_id]["author"] = author_info

        # Get general TikTok information
        general_info = {
            "text": tiktok["text"],
            "hashtags": [hashtag["name"] for hashtag in tiktok["hashtags"]],
            "music_id": tiktok["musicMeta"]["musicId"]
        }
        tiktok_info[tiktok_id]["general"] = general_info

        # Get music information
        music_info = {
            "music_name": tiktok["musicMeta"]["musicName"],
            "music_author": tiktok["musicMeta"]["musicAuthor"],
            "music_original": tiktok["musicMeta"]["musicOriginal"]
        }
        tiktok_info[tiktok_id]["music"] = music_info

        # Add the video URL
        tiktok_info[tiktok_id]["video_url"] = tiktok["videoUrl"]

    # Save the list of TikTok information to a JSON file
    with open(os.path.join(root_dir,"data\\inputs\\tiktok_info.json"), "w") as json_output:
        json.dump(tiktok_info, json_output, indent=4)

# Call the function to extract TikTok information
extract_tiktok_info()
