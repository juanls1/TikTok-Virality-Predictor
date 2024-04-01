from TikTokApi import TikTokApi
import os
import pytest

ms_token = "XXEM7EhAUnpm4JaBYMqFdO0yJ2I-Z6TXaDI1_WivMRgVstsOMvfepC5atfqiBq0mYsjF_G-s4L2GKT6EC1t5UC08d5CkaJPZ8QYPME_pqxvDaNcFLnQFbNk4m1MB6enReFevg76FmXOoOlZ3bw=="


@pytest.mark.asyncio
async def test_user_info():
    api = TikTokApi()
    async with api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
        count = 0
        async for video in api.trending.videos(count=3):
            count += 1

        assert count >= 3


print("done")