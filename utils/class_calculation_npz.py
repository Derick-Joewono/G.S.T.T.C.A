import os
import glob
import numpy as np
from tqdm import tqdm

def calculate_npz_class_distribution(cache_dir):
    # 1. Ambil semua file .npz di direktori cache
    file_list = glob.glob(os.path.join(cache_dir, "*.npz"))
    
    if not file_list:
        print(f"Folder tidak ditemukan atau tidak ada file .npz di: {cache_dir}")
        return

    # Inisialisasi hitungan (0: No Change, 1: Deforestation, 2: Forestation)
    counts = {0: 0, 1: 0, 2: 0}
    total_valid_pixels = 0
    ignore_index = -1
    ignored_pixels = 0

    print(f"Menganalisis {len(file_list)} file .npz hasil preprocess...")

    # 2. Iterasi setiap file
    for file_path in tqdm(file_list):
        try:
            # Load data npz
            data = np.load(file_path)
            mask = data["mask"] # Mengambil array mask saja
            
            # Hitung pixel per kelas
            # Kita abaikan ignore_index (-1) agar persentase kelas asli akurat
            for cls_id in [0, 1, 2]:
                count = np.sum(mask == cls_id)
                counts[cls_id] += count
                total_valid_pixels += count
            
            # Hitung juga yang di-ignore untuk info tambahan
            ignored_pixels += np.sum(mask == ignore_index)
            
        except Exception as e:
            print(f"Error membaca {file_path}: {e}")

    # 3. Tampilkan Hasil
    print("\n" + "="*45)
    print("      HASIL ANALISIS DISTRIBUSI KELAS (.NPZ)")
    print("="*45)
    
    class_names = {0: "No Change", 1: "Deforestation", 2: "Forestation"}
    
    for cls_id in [0, 1, 2]:
        pixel_count = counts[cls_id]
        percentage = (pixel_count / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
        
        print(f"Kelas {cls_id} ({class_names[cls_id]:13}):")
        print(f"  - Jumlah Pixel : {pixel_count:,}")
        print(f"  - Persentase   : {percentage:.4f} %")
    
    print("-" * 45)
    print(f"Total Valid Pixels   : {total_valid_pixels:,}")
    print(f"Total Ignored Pixels : {ignored_pixels:,} (Index -1)")
    print("="*45)
    
    # 4. Saran Class Weight (Inverse Frequency)
    print("\nSaran Class Weight (untuk Loss Function):")
    for cls_id in [0, 1, 2]:
        if counts[cls_id] > 0:
            # Rumus: Total / (Jumlah_Kelas * Frekuensi_Kelas)
            weight = total_valid_pixels / (len(counts) * counts[cls_id])
            print(f"  - Weight Kelas {cls_id}: {weight:.4f}")

# Jalankan fungsi pada folder cache_npy kamu
cache_dir = "data/cache_npy_output" 

if os.path.exists(cache_dir):
    calculate_npz_class_distribution(cache_dir)
else:
    print(f"Path cache tidak ditemukan: {cache_dir}")