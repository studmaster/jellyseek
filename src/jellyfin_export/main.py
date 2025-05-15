from config import JELLYFIN_API_KEY, JELLYFIN_URL, JELLYFIN_DATA_PATH
import requests
import json
from pathlib import Path

def fetch_items():
    headers = {
        "X-Emby-Token": JELLYFIN_API_KEY
    }
    response = requests.get(f"{JELLYFIN_URL}/Items", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch items: {response.status_code} - {response.text}")
        return None

def save_items(items):
    """Save items to JSON file"""
    output_file = Path(JELLYFIN_DATA_PATH) / 'jellyfin_items.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Data saved to: {output_file}")

def main():
    print(f"API Key: {JELLYFIN_API_KEY}")
    print(f"URL: {JELLYFIN_URL}")
    print(f"Data Path: {JELLYFIN_DATA_PATH}")
    
    items = fetch_items()
    if items:
        save_items(items)
        print(f"Fetched and saved {len(items.get('Items', []))} items")

if __name__ == "__main__":
    main()