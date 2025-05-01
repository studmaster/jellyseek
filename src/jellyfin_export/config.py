from dotenv import load_dotenv
import os

load_dotenv()

JELLYFIN_URL = os.getenv("JELLYFIN_SERVER_URL")
JELLYFIN_API_KEY = os.getenv("JELLYFIN_SERVER_API_KEY")

if not JELLYFIN_URL or not JELLYFIN_API_KEY:
    raise ValueError("JELLYFIN_SERVER_URL and JELLYFIN_SERVER_API_KEY must be set in the environment variables.")  
