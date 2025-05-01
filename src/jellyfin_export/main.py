from config import JELLYFIN_API_KEY, JELLYFIN_URL
import requests

def main():
    print(f"API Key: {JELLYFIN_API_KEY}")
    print(f"URL: {JELLYFIN_URL}")

if __name__ == "__main__":
    main()

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

    items = fetch_items()
    if items:
        print("Fetched items:", items)