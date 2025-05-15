from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

JELLYFIN_URL = os.getenv("JELLYFIN_SERVER_URL")
JELLYFIN_API_KEY = os.getenv("JELLYFIN_SERVER_API_KEY")

# Get default data path
DEFAULT_DATA_PATH = os.path.join(os.path.expanduser("~"), ".local", "share", "jellyseek", "data")
# Get configured path and expand user directory if needed
raw_path = os.getenv("JELLYFIN_DATA_PATH", DEFAULT_DATA_PATH)
JELLYFIN_DATA_PATH = os.path.expanduser(raw_path)

if not JELLYFIN_URL or not JELLYFIN_API_KEY:
    raise ValueError("JELLYFIN_SERVER_URL and JELLYFIN_SERVER_API_KEY must be set in the environment variables.")

# Create data directory if it doesn't exist
try:
    os.makedirs(JELLYFIN_DATA_PATH, exist_ok=True)
except PermissionError:
    print(f"Warning: Cannot create directory at {JELLYFIN_DATA_PATH}. Check permissions.")
    JELLYFIN_DATA_PATH = DEFAULT_DATA_PATH
    os.makedirs(JELLYFIN_DATA_PATH, exist_ok=True)
