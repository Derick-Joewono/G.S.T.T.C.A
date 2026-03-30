import os
import numpy as np
import shutil
from tqdm import tqdm

def compute_patch_stats(cache_dir):
    """Menghitung jumlah pixel per kelas untuk setiap file .npz"""
    # Ambil list ID dari file ids.npy hasil preprocess_to_npy
    ids = np.load(os.path.join(cache_dir, "ids.npy"))
    stats = []

    print(f"🔍 Menganalisis statistik pixel dari {len(ids)} patches...")
    for file_id in tqdm(ids):
        # Sesuaikan ekstensi ke .npz karena hasil preprocess_to_npy adalah NPZ
        npz_path = os.path.join(cache_dir, file_id.replace(".tif", ".npz"))
        
        if not os.path.exists(npz_path):
            continue
            
        data = np.load(npz_path)
        mask = data["mask"]

        total = mask.size
        # Hitung pixel (0: No Change, 1: Deforest, 2: Forest)
        c0 = np.sum(mask == 0)
        c1 = np.sum(mask == 1)
        c2 = np.sum(mask == 2)
        change = c1 + c2

        stats.append({
            "id": file_id,
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "change_ratio": change / total if total > 0 else 0
        })
    return stats

def select_750_with_pixel_ratio(stats, target_ratio=0.1, total_patches=750):
    """Memilih 750 patch terbaik menggunakan algoritma Greedy"""
    # Sort dari yang punya rasio perubahan (change_ratio) tertinggi
    stats = sorted(stats, key=lambda x: x["change_ratio"], reverse=True)

    selected = []
    total_c0, total_c1, total_c2 = 0, 0, 0

    for s in stats:
        if len(selected) >= total_patches:
            break

        # Kalkulasi simulasi rasio jika patch ini ditambah
        new_c0 = total_c0 + s["c0"]
        new_c1 = total_c1 + s["c1"]
        new_c2 = total_c2 + s["c2"]

        total_pixels = new_c0 + new_c1 + new_c2
        new_ratio = (new_c1 + new_c2) / total_pixels

        # Syarat Greedy: Jangan melebihi target_ratio + toleransi 2%
        if new_ratio <= target_ratio + 0.02:
            selected.append(s["id"])
            total_c0, total_c1, total_c2 = new_c0, new_c1, new_c2

    print(f"\n✅ Berhasil memilih: {len(selected)} patches")
    final_ratio = (total_c1 + total_c2) / (total_c0 + total_c1 + total_c2)
    print(f"📊 Final Pixel Ratio (Change/Total): {final_ratio:.4f}")
    
    return selected