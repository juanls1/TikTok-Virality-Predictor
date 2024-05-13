from TikTokApi import TikTokApi
import asyncio
import os
import json
from moviepy.editor import VideoFileClip
import argparse
import time

parser = argparse.ArgumentParser(description="Download videos from TikTok and extract audio.")
parser.add_argument("-videos", help="Download videos", action="store_true")
parser.add_argument("-audio", help="Extract audio from downloaded videos", action="store_true")
args = parser.parse_args()


ms_token = os.environ.get(
    "ms_token", None
)  

import asyncio

async def main():
    if args.videos:
        if not os.path.exists("videos"):
            os.makedirs("videos")
        if not os.path.exists("info.json"):
            with open("info.json", "w") as out:
                out.write("{}")
        
        with open("info.json", "r") as file:
            downloaded_videos = json.load(file)
        
        async with TikTokApi() as api:
            await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False)
            
            download_tasks = []  # List to store download tasks
            async for video in api.trending.videos(count=15):
                if video.id not in downloaded_videos:
                    # Schedule video download concurrently
                    task = asyncio.create_task(download_video(video))
                    download_tasks.append(task)
                else:
                    print(f"Video {video.id} already downloaded. Skipping...")
            
            # Wait for all download tasks to complete
            await asyncio.gather(*download_tasks)

    if args.audio:
        extract_audio()


def extract_audio():
    if not os.path.exists("audios"):
        os.makedirs("audios")
    for video_file in os.listdir("videos"):
        if video_file.endswith(".mp4"):
            video_path = os.path.join("videos", video_file)
            audio_path = os.path.join("audios", video_file.replace(".mp4", "_audio.mp3"))
            print(f"Extracting audio from {video_file}")
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path)
            audio_clip.close()
            video_clip.close()


async def download_video(video):
    path = "videos/"
    print(f"Downloading {video.id}.mp4")

    try:
        video_bytes = await video.bytes()
        with open(f"{path}{video.id}.mp4", "wb") as out:
            out.write(video_bytes)

        print(f"Downloaded {video.id}.mp4")

        with open("info.json", "r") as file:
            data = json.load(file)

        data[video.id] = {}

        data[video.id]["metrics"] = {
            "likes": video.stats['diggCount'],
            "shares": video.stats['shareCount'],
            "comments": video.stats['commentCount'],
            "views": video.stats['playCount'],
            "saved": video.stats['collectCount']
        }
        data[video.id]["author"] = {
            "id": video.author.user_id,
            "username": video.author.username,
            "verified": video.author.as_dict['verified']
        }

        # Convert hashtags into a list of dictionaries
        hashtags_list = []
        for hashtag in video.hashtags:
            hashtags_list.append({"id": hashtag.id, "name": hashtag.name})

        data[video.id]["general"] = {
            "text": video.as_dict['desc'],
            "hashtags": hashtags_list,  # Here hashtags are included as a list of dictionaries
            "music_id": video.sound.id
        }

        with open("info.json", "w") as out:
            json.dump(data, out, indent=4)

    except Exception as e:
        print(f"Error downloading {video.id}.mp4: {e}")
        

    
if __name__ == "__main__":
    asyncio.run(main())
