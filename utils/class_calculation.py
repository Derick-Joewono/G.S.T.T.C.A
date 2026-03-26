import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_class_distribution(mask_path):
    # 1. Ambil semua file .tiff di direktori tersebut
    file_list = glob.glob(os.path.join(mask_path, "*.tif"))
    
    if not file_list:
        print("Folder tidak ditemukan atau tidak ada file .tif!")
        return

    # Inisialisasi hitungan (0: No Change, 1: Deforestation, 2: Forestation)
    counts = {0: 0, 1: 0, 2: 0}
    total_pixels = 0

    print(f"Menganalisis {len(file_list)} file mask...")

    # 2. Iterasi setiap file
    for file_name in tqdm(file_list):
        try:
            # Buka mask
            with Image.open(file_name) as img:
                mask_array = np.array(img)
                
                # Hitung nilai unik dan frekuensinya
                unique, counts_in_patch = np.unique(mask_array, return_counts=True)
                
                # Update total counts
                for val, count in zip(unique, counts_in_patch):
                    if val in counts:
                        counts[val] += count
                        total_pixels += count
        except Exception as e:
            print(f"Error membaca {file_name}: {e}")

    # 3. Hitung presentase dan tampilkan hasil
    print("\n" + "="*30)
    print("HASIL ANALISIS DISTRIBUSI KELAS")
    print("="*30)
    
    class_names = {0: "No Change", 1: "Deforestation", 2: "Forestation"}
    
    for cls_id in [0, 1, 2]:
        pixel_count = counts[cls_id]
        percentage = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        
        print(f"Kelas {cls_id} ({class_names[cls_id]}):")
        print(f"  - Jumlah Pixel : {pixel_count:,}")
        print(f"  - Persentase   : {percentage:.4f} %")
    
    print("="*30)
    
    # Menghitung Inverse Frequency Weight (untuk referensi coding)
    print("\nSaran Class Weight (Inverse Frequency):")
    for cls_id in [0, 1, 2]:
        if counts[cls_id] > 0:
            weight = total_pixels / (len(counts) * counts[cls_id])
            print(f"  - Weight Kelas {cls_id}: {weight:.4f}")

# Jalankan fungsi
# Ganti dengan path folder mask kamu
# Ganti baris terakhir pada kode sebelumnya dengan ini:
path_mask = "/Users/macbook/Documents/Binus/Research Track/Codes/data/mask"

# Tambahkan pengecekan folder untuk memastikan path benar sebelum running
if os.path.exists(path_mask):
    calculate_class_distribution(path_mask)
else:
    print(f"Path tidak ditemukan! Pastikan folder berikut ada: {path_mask}")

calculate_class_distribution(path_mask)