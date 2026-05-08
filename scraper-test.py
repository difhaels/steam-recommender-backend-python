from dotenv import load_dotenv
import os

load_dotenv()

import requests
import json
import time

# --- KONFIGURASI ---
API_KEY = os.getenv("API_KEY")  # Ganti dengan Steam API Key kamu
SEED_ID = '76561198418191535'   # ID awal untuk memulai snowball sampling
TARGET_COUNT = 10               # Jumlah user valid yang ingin diambil

def get_owned_games(steam_id):
    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={API_KEY}&steamid={steam_id}&include_appinfo=1&format=json"
    try:
        r = requests.get(url, timeout=10)
        return r.json().get('response', {}).get('games', [])
    except:
        return None

def get_friends(steam_id):
    url = f"http://api.steampowered.com/ISteamUser/GetFriendList/v1/?key={API_KEY}&steamid={steam_id}&relationship=friend"
    try:
        r = requests.get(url, timeout=10)
        friends = r.json().get('friendslist', {}).get('friends', [])
        return [f['steamid'] for f in friends]
    except:
        return []

def run_scraper():
    valid_users = []
    queue = [SEED_ID]
    visited = set()
    
    print(f"Memulai Snowball Sampling untuk {TARGET_COUNT} user...")

    while len(valid_users) < TARGET_COUNT and queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        
        visited.add(current_id)
        print(f"Mengecek User: {current_id}...", end=" ")

        # 1. Ambil game user
        games = get_owned_games(current_id)
        if games is None:
            print("Skips (Private/Error)")
            continue

        # 2. Filter: Hanya game dengan playtime > 0
        playtime_data = [
            {
                "app_id": g['appid'],
                "name": g['name'],
                "playtime": g['playtime_forever']
            }
            for g in games if g['playtime_forever'] > 0
        ]

        # 3. Validasi Kriteria: Minimal 10 game
        if len(playtime_data) >= 10:
            user_entry = {
                "steam_id": current_id,
                "total_valid_games": len(playtime_data),
                "library": playtime_data
            }
            valid_users.append(user_entry)
            print(f"BERHASIL! ({len(valid_users)}/{TARGET_COUNT})")
            
            # Tambahkan teman user ini ke antrean (Snowball)
            friends = get_friends(current_id)
            queue.extend(friends)
        else:
            print(f"Skips (Hanya punya {len(playtime_data)} game aktif)")

        time.sleep(1) # Jeda sopan agar tidak di-ban Steam

    # Simpan hasil akhir
    output = {
        "metadata": {
            "source": "Steam Web API",
            "method": "Snowball Sampling",
            "criteria": "min_10_games_with_playtime",
            "total_collected": len(valid_users)
        },
        "data": valid_users
    }

    with open('dataset_sample.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print("\nFile 'dataset_sample.json' telah berhasil dibuat!")

if __name__ == "__main__":
    run_scraper()