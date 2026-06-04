import os
import time
import json
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# 1. Inisialisasi dan Konfigurasi
load_dotenv()
API_KEY = os.getenv("API_KEY")
MAX_PLAYTIME = 120000  # Batasan max 120.000 menit dari draf Bab III

app = FastAPI(
    title="Steam Recommender System API",
    description="Backend FastAPI untuk Skripsi Sistem Rekomendasi Game Steam menggunakan KNN"
)

# Aktifkan CORS agar Next.js (Frontend) bisa nembak API ini tanpa diblokir browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Di produksi, ganti dengan URL Next.js kamu (misal: http://localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables untuk menampung data di memori server agar loading instan
train_matrix = None
game_mapping = {}

@app.on_event("startup")
def load_data_to_memory():
    """Fungsi yang otomatis berjalan saat server FastAPI dinyalakan"""
    global train_matrix, game_mapping
    print("[STARTUP] Memuat dataset ke memori server...")
    
    try:
        # Muat matriks utama pelatihan (492 user x 5227 game)
        train_matrix = pd.read_csv('user_item_matrix_train.csv', index_col='steam_id')
        train_matrix.columns = train_matrix.columns.astype(int) # Kembalikan App ID ke Integer
        print(f"[SUKSES] Matriks Pelatihan Berhasil Dimuat: {train_matrix.shape[0]} User, {train_matrix.shape[1]} Game.")
        
        # Simulasi/Load kamus nama game agar outputnya keluar judul game, bukan cuma ID angka
        # Jika kamu punya file mapping/dataset game mentah, bisa di-load di sini.
        # Sebagai fallback, kita buat dictionary kosong atau dummy
        if os.path.exists('dataset_from_500.json'):
            with open('dataset_from_500.json', 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                for user in raw_data.get('data', []):
                    for game in user.get('library', []):
                        game_mapping[int(game['app_id'])] = game['name']
            print(f"[SUKSES] Berhasil memuat {len(game_mapping)} nama judul game ke kamus sistem.")
    except Exception as e:
        print(f"[ERROR STARTUP] Gagal memuat file csv/json: {str(e)}")

# 2. Helper Functions untuk Fetch & Preprocess Data Real-time (dari scraper.py)
def fetch_steam_user_games(steam_id: str):
    """Nembak Steam Web API secara live untuk mengambil library game milik user baru"""
    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={API_KEY}&steamid={steam_id}&include_appinfo=1&format=json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get('response', {}).get('games', [])
        return None
    except:
        return None

def process_new_user_vector(games_list):
    """Mengubah JSON dari Steam API menjadi vektor 1D yang selaras dengan kolom train_matrix"""
    # Buat Series kosong dengan struktur kolom yang SAMA PERSIS dengan matriks pelatihan
    user_vector = pd.Series(0.0, index=train_matrix.columns)
    
    valid_game_count = 0
    for g in games_list:
        app_id = int(g['appid'])
        playtime = g['playtime_forever']
        
        # Filter 1: Game harus terdaftar di dalam kolom data latih kita
        if app_id in user_vector.index and playtime > 0:
            # Filter 2: Capping Outlier max 120.000 menit
            if playtime > MAX_PLAYTIME:
                playtime = MAX_PLAYTIME
                
            # Filter 3: Normalisasi Min-Max (Min=0, Max=120.000)
            playtime_norm = playtime / MAX_PLAYTIME
            
            user_vector[app_id] = playtime_norm
            valid_game_count += 1
            
            # Update game mapping secara dinamis jika ada nama baru
            if app_id not in game_mapping:
                game_mapping[app_id] = g.get('name', f"Unknown Game ({app_id})")
                
    return user_vector, valid_game_count

# 3. Model Request Pydantic untuk Skema Input API
class RecommendationRequest(BaseModel):
    steam_id: str
    metric: str = "Cosine"  # Pilihan: "Cosine" atau "Euclidean"
    k_neighbors: int = 20
    n_recommendations: int = 10

# 4. Core Endpoint API Utama
@app.post("/api/recommend")
def get_recommendations(req: RecommendationRequest):
    global train_matrix, game_mapping
    
    target_user = req.steam_id
    metric = req.metric
    k = req.k_neighbors
    n = req.n_recommendations
    
    # Validasi apakah input berupa string angka
    if not target_user.isdigit():
        raise HTTPException(status_code=400, detail="Steam ID harus berupa barisan angka valid.")
    
    target_user_numeric = int(target_user) if target_user.isdigit() else target_user
    
    # TAHAP 1: Ambil Vektor Inti Pengguna Target
    is_existing_user = target_user_numeric in train_matrix.index or target_user in train_matrix.index
    
    if is_existing_user:
        # Jika user lama, langsung ambil barisnya dari train_matrix
        idx = target_user_numeric if target_user_numeric in train_matrix.index else target_user
        target_vector = train_matrix.loc[idx]
        current_train_data = train_matrix.drop(index=idx)  # Buang diri sendiri agar tidak bias
    else:
        # Jika user baru, tembak Steam API secara live
        print(f"[LIVE FETCH] Menarik data untuk user baru: {target_user}")
        raw_games = fetch_steam_user_games(target_user)
        
        if not raw_games:
            raise HTTPException(
                status_code=404, 
                detail="Steam ID tidak ditemukan atau akun di-setting PRIVATE. Pastikan inventory/game details Anda PUBLIC."
            )
            
        target_vector, valid_count = process_new_user_vector(raw_games)
        
        if valid_count < 10:
            raise HTTPException(
                status_code=400, 
                detail=f"User baru hanya memiliki {valid_count} game yang valid di database. Syarat minimal adalah 10 game aktif."
            )
        current_train_data = train_matrix

    # TAHAP 2: Hitung Matrix Similarity (Cosine vs Euclidean Transformed)
    target_matrix_1d = target_vector.values.reshape(1, -1)
    
    if metric.lower() == 'cosine':
        # Hitung kemiripan sudut Cosine
        sim_scores = cosine_similarity(target_matrix_1d, current_train_data.values)[0]
    elif metric.lower() == 'euclidean':
        # Hitung jarak Euclidean lalu transformasikan menjadi Similarity: 1 / (1 + d)
        dist_scores = euclidean_distances(target_matrix_1d, current_train_data.values)[0]
        sim_scores = 1 / (1 + dist_scores)
    else:
        raise HTTPException(status_code=400, detail="Metrik tidak valid. Pilih 'Cosine' atau 'Euclidean'.")
        
    user_similarities = pd.Series(sim_scores, index=current_train_data.index)
    
    # TAHAP 3: Cari K-Nearest Neighbors
    nearest_neighbors = user_similarities.sort_values(ascending=False).head(k)
    neighbors_data = current_train_data.loc[nearest_neighbors.index]
    
    # TAHAP 4: Hitung Skor Prediksi dengan Dot Product (Weighted Sum)
    predicted_scores = nearest_neighbors.values.dot(neighbors_data.values)
    predicted_scores_series = pd.Series(predicted_scores, index=current_train_data.columns)
    
    # TAHAP 5: Filter Game yang Sudah Dimainkan
    games_not_played = target_vector[target_vector == 0].index
    valid_recommendations = predicted_scores_series.loc[games_not_played]
    
    # TAHAP 6: Ambil Top-N Teratas
    top_n_recommendations = valid_recommendations.sort_values(ascending=False).head(n)
    recommended_ids = top_n_recommendations.index.tolist()
    
    # TAHAP 7: Kembalikan respon berformat JSON yang kaya informasi untuk Frontend
    results = []
    for i, app_id in enumerate(recommended_ids, 1):
        results.append({
            "rank": i,
            "app_id": app_id,
            "game_name": game_mapping.get(app_id, f"Unknown Game ID: {app_id}"),
            "predicted_score": round(float(top_n_recommendations[app_id]), 4)
        })
        
    return {
        "status": "success",
        "user_type": "Existing Dataset User" if is_existing_user else "New Live Steam User",
        "metric_used": metric,
        "k_neighbors": k,
        "n_displayed": n,
        "recommendations": results
    }

# Endpoint cek kesehatan server
@app.get("/")
def read_root():
    return {"message": "Steam Recommender Backend is Running Smoothly!"}