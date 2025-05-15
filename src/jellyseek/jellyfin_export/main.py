from jellyseek.jellyfin_export.config import JELLYFIN_API_KEY, JELLYFIN_URL, JELLYFIN_DATA_PATH
import requests
import json
from pathlib import Path

def get_movies_folder_id():
    """Get the ID of the Movies folder"""
    headers = {
        "X-Emby-Token": JELLYFIN_API_KEY
    }
    
    response = requests.get(f"{JELLYFIN_URL}/Items", headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch items: {response.status_code} - {response.text}")
        return None
        
    data = response.json()
    for item in data.get('Items', []):
        if (item.get('Name') == 'Movies' and 
            item.get('Type') == 'Folder' and 
            item.get('IsFolder')):
            return item.get('Id')
    
    return None

def fetch_items():
    """Fetch all movie items from Jellyfin under the Movies folder"""
    movies_folder_id = get_movies_folder_id()
    if not movies_folder_id:
        print("Could not find Movies folder!")
        return None
        
    print(f"Found Movies folder with ID: {movies_folder_id}")
    
    headers = {
        "X-Emby-Token": JELLYFIN_API_KEY
    }
    
    params = {
        "ParentId": movies_folder_id,
        "Recursive": "true",
        "IncludeItemTypes": "Movie",
        "Fields": "Path,Overview,PremiereDate,CriticRating,CommunityRating,OfficialRating,Tags,Genres,Actors",
        "EnableImages": "false"
    }
    
    response = requests.get(
        f"{JELLYFIN_URL}/Items", 
        headers=headers,
        params=params
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch movies: {response.status_code} - {response.text}")
        return None

def save_items(items):
    """Save items to JSON file"""
    output_file = Path(JELLYFIN_DATA_PATH) / 'jellyfin_items.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Data saved to: {output_file}")

def main():
    print(f"Connecting to Jellyfin server at: {JELLYFIN_URL}")
    
    items = fetch_items()
    if items:
        save_items(items)
        print(f"Fetched and saved {len(items.get('Items', []))} movies")

if __name__ == "__main__":
    main()