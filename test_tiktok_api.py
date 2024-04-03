from TikTokApi import TikTokApi
import asyncio
import os
import json

ms_token = os.environ.get(
    "ms_token", None
)  

async def main():
    if not os.path.exists("videos"):
        os.makedirs("videos")
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False) #headless=False is the unique change. It is needed to function properly
        videos = []
        async for video in api.trending.videos(count=5):
            videos.append(video)
            await download_video(video)


async def download_video(video):
    path = f"videos/"
    print(f"Downloading {video.id}.mp4")

    video_bytes = await video.bytes()
    with open(f"{path}{video.id}.mp4", "wb") as out:
        out.write(video_bytes)

    print(f"Downloaded {video.id}.mp4")

    if not os.path.exists(f"info.json"):
        with open(f"info.json", "w") as out:
            out.write("{}")

    with open(f"info.json", "r") as file:
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
    data[video.id]["general"] = {
        "text": video.as_dict['desc'],
        "hashtags": video.hashtags,
        "music_id": video.sound.id
    }

    with open(f"info.json", "w") as out:
        json.dump(data, out, indent=4)

    
if __name__ == "__main__":
    asyncio.run(main())