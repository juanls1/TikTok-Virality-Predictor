from TikTokApi import TikTokApi
import os
import pytest

ms_token = os.environ.get("ms_token", None)


@pytest.mark.asyncio
async def test_user_info():
    api = TikTokApi()
    async with api:
        await api.create_sessions(num_sessions=1, sleep_after=3)
        count = 0
        async for video in api.trending.videos(count=3):
            count += 1

        assert count >= 3


print("done")