import os
import numpy as np
import shutil
from tqdm import tqdm

# ==========================================
# 1. FUNGSI STATISTIK & SELEKSI (GREEDY)
# ==========================================

def compute_patch_stats(cache_dir):
    """Menghitung statistik pixel per kelas untuk setiap file .npz"""
    ids_path = os.path.join(cache_dir, "ids.npy")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"File ids.npy tidak ditemukan di {cache_dir}!")
        
    ids = np.load(ids_path)
    stats = []

    print(f"🔍 Menganalisis statistik pixel dari {len(ids)} patches...")
    for file_id in tqdm(ids):
        npz_path = os.path.join(cache_dir, file_id.replace(".tif", ".npz"))
        
        if not os.path.exists(npz_path):
            continue
            
        with np.load(npz_path) as data:
            mask = data["mask"]
            total = mask.size
            c0 = np.sum(mask == 0)
            c1 = np.sum(mask == 1) # Deforest
            c2 = np.sum(mask == 2) # Forest
            change = c1 + c2

            stats.append({
                "id": file_id,
                "c0": c0,
                "c1": c1,
                "c2": c2,
                "change_ratio": change / total if total > 0 else 0
            })
    return stats

def select_5000_with_30_percent_ratio(stats, target_ratio=0.3, total_patches=5000):
    """Memilih 5000 patch dengan target rasio pixel 30% Change"""
    # Sort dari yang perubahan pixel-nya paling banyak
    stats = sorted(stats, key=lambda x: x["change_ratio"], reverse=True)

    selected = []
    total_c0, total_c1, total_c2 = 0, 0, 0

    print(f"🎯 Mencari {total_patches} patches untuk rasio {target_ratio*100}%...")

    for s in stats:
        if len(selected) >= total_patches:
            break

        # Simulasi rasio jika patch ini ditambah
        temp_c0 = total_c0 + s["c0"]
        temp_c1 = total_c1 + s["c1"]
        temp_c2 = total_c2 + s["c2"]

        total_pixels = temp_c0 + temp_c1 + temp_c2
        new_ratio = (temp_c1 + temp_c2) / total_pixels

        # Syarat Greedy: Karena 5000 data itu banyak, kita beri toleransi 5% (0.35)
        # agar algoritma tidak terlalu pelit dan bisa memenuhi kuota 5000 file.
        if new_ratio <= target_ratio + 0.05:
            selected.append(s)
            total_c0, total_c1, total_c2 = temp_c0, temp_c1, temp_c2

    final_ratio = (total_c1 + total_c2) / (total_c0 + total_c1 + total_c2)
    print(f"\n✅ Berhasil memilih: {len(selected)} patches")
    print(f"📊 Final Pixel Ratio (Change/Total): {final_ratio*100:.2f}%")
    
    return selected