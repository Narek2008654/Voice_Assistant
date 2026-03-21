"""
Generate a LiveKit room token for testing the STT agent.
Prints the token and a ready-to-use join URL.

Usage:
    python generate_token.py
"""

from dotenv import load_dotenv
import os

load_dotenv(".env.local")

from livekit.api import AccessToken, VideoGrants

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

ROOM_NAME = "test-room"
USER_IDENTITY = "test-user"
USER_NAME = "Test User"


def generate_token() -> str:
    token = AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
    token.with_identity(USER_IDENTITY).with_name(USER_NAME).with_grants(
        VideoGrants(room_join=True, room=ROOM_NAME)
    )
    return token.to_jwt()


if __name__ == "__main__":
    jwt = generate_token()
    ws_url = LIVEKIT_URL

    print("\n" + "=" * 60)
    print("  LIVEKIT CONNECTION DETAILS")
    print("=" * 60)
    print(f"  Room:     {ROOM_NAME}")
    print(f"  Identity: {USER_IDENTITY}")
    print(f"  URL:      {ws_url}")
    print(f"\n  Token:\n  {jwt}")
    print(f"\n  Meet URL (open in browser):")
    print(f"  https://meet.livekit.io/custom?liveKitUrl={ws_url}&token={jwt}")
    print("=" * 60 + "\n")
