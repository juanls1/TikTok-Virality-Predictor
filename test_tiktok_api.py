from TikTokApi import TikTokApi
import asyncio
import os

ms_token = os.environ.get(
    "ms_token", None
)  # set your own ms_token, think it might need to have visited a profile

async def foo():
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False) #headless=False is the unique change. It is needed to function properly
        videos = []
        async for video in api.trending.videos(count=2):
            print(video.id)
            videos.append(video)
            video_bytes = await video.bytes()

            with open(f"{video.id}.mp4", "wb") as out:
                out.write(video_bytes)

asyncio.run(foo())

print("done")