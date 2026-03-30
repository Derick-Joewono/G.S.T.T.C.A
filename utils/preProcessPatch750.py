import os
import numpy as np
import shutil
from tqdm import tqdm

# ==========================================
# 1. FUNGSI STATISTIK & SELEKSI (GREEDY)
# ==========================================

def compute_patch_stats(cache_dir):
    """Menghitung statistik pixel per kelas untuk setiap file .npz"""
    ids = np.load(os.path.join(cache_dir, "ids.npy"))
    stats = []

    print(f"🔍 Menganalisis statistik pixel dari {len(ids)} patches...")
    for file_id in tqdm(ids):
        npz_path = os.path.join(cache_dir, file_id.replace(".tif", ".npz"))
        
        if not os.path.exists(npz_path):
            continue
            
        # Gunakan 'with' agar file langsung tertutup setelah dibaca (aman untuk memori)
        with np.load(npz_path) as data:
            mask = data["mask"]
            total = mask.size
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
    """Memilih 750 patch terbaik dengan rasio pixel target"""
    stats = sorted(stats, key=lambda x: x["change_ratio"], reverse=True)

    selected = []
    total_c0, total_c1, total_c2 = 0, 0, 0

    for s in stats:
        if len(selected) >= total_patches:
            break

        new_c0 = total_c0 + s["c0"]
        new_c1 = total_c1 + s["c1"]
        new_c2 = total_c2 + s["c2"]

        total_pixels = new_c0 + new_c1 + new_c2
        new_ratio = (new_c1 + new_c2) / total_pixels

        if new_ratio <= target_ratio + 0.02:
            selected.append(s) # Simpan full dict untuk diproses di tahap copy
            total_c0, total_c1, total_c2 = new_c0, new_c1, new_c2

    final_ratio = (total_c1 + total_c2) / (total_c0 + total_c1 + total_c2)
    print(f"\n✅ Berhasil memilih: {len(selected)} patches")
    print(f"📊 Final Pixel Ratio (Change/Total): {final_ratio:.4f}")
    
    return selected

# ==========================================
# 2. ALUR EKSEKUSI (JALANKAN DI CELL INI)
# ==========================================

# --- CONFIG PATH ---
cache_dir_asli = "data/cache_npy"
cache_dir_output = "data/cache_npy_output"
os.makedirs(cache_dir_output, exist_ok=True)

# A. Ambil Statistik (Ini proses 6 jam kamu)
stats_data = compute_patch_stats(cache_dir_asli)

# B. Seleksi ID Terbaik
selected_items = select_750_with_pixel_ratio(stats_data, target_ratio=0.1, total_patches=750)

# C. Salin File & Buat Metadata Sinkron (ids.npy & labels.npy)
print(f"\n📂 Tahap Akhir: Menyalin file & sinkronisasi metadata...")
new_ids = []
new_labels = []

for item in tqdm(selected_items):
    file_id = item["id"]
    file_npz = file_id.replace(".tif", ".npz")
    
    src = os.path.join(cache_dir_asli, file_npz)
    dst = os.path.join(cache_dir_output, file_npz)
    
    if os.path.exists(src):
        # 1. Salin file fisik .npz
        shutil.copy(src, dst)
        
        # 2. Tambahkan ke list IDs
        new_ids.append(file_id)
        
        # 3. Buat Label Biner (0: No Change, 1: Change) untuk stratify DataLoader
        # Jika c1 atau c2 > 0, maka dianggap ada perubahan
        label = 1 if (item["c1"] > 0 or item["c2"] > 0) else 0
        new_labels.append(label)

# D. Save Metadata ke Folder Output
np.save(os.path.join(cache_dir_output, "ids.npy"), np.array(new_ids))
np.save(os.path.join(cache_dir_output, "labels.npy"), np.array(new_labels))

print(f"\n✨ SEMUA SIAP!")
print(f"Dataset 750 patch emas ada di: {cache_dir_output}")