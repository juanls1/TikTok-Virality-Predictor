from TikTokApi import TikTokApi
import os

ms_token = os.environ.get(
    "ms_token", None
)  # set your own ms_token, think it might need to have visited a profile


async def fetch_videos():
    api = TikTokApi()
    async with api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
        videos = []
        async for video in api.trending.videos(count=3):
            videos.append(video)
            print("hey")
        return videos


async def debug_fetch_videos():
    videos = await fetch_videos()
    # Aquí puedes poner puntos de interrupción o utilizar print() para inspeccionar los vídeos
    for video in videos:
        print(video)

import asyncio

asyncio.run(debug_fetch_videos())

print("done")