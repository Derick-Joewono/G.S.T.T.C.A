import os
import shutil
import numpy as np
import tifffile as tiff
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = '../data/dataSumatera' 
OUTPUT_DIR = '../data'
FOLDERS = ['t1', 't2', 'mask']
MAX_BACKGROUND = 0.4 # Target: Label 0 (Background) maksimal 40% dari gambar

for f in FOLDERS: 
    os.makedirs(os.path.join(OUTPUT_DIR, f), exist_ok=True)

def get_pixel_ratios(path):
    mask = tiff.imread(path)
    flat_mask = mask.flatten()
    
    # 1. Hitung jumlah masing-masing label (0, 1, 2)
    # Kita filter hanya yang >= 0 agar bincount tidak error
    counts = np.bincount(flat_mask[flat_mask >= 0].astype(int), minlength=3)
    
    # 2. Hitung total pixel yang VALID saja (bukan -1)
    total_valid_pixels = np.sum(flat_mask >= 0)
    
    if total_valid_pixels == 0:
        return np.array([1.0, 0.0, 0.0]) # Kasus ekstrim: gambar isi -1 semua
        
    # 3. Rasio dihitung terhadap total_valid_pixels (Sesuai maumu)
    return counts / total_valid_pixels

# # 1. List File
all_masks = [f for f in os.listdir(os.path.join(BASE_DIR, 'mask')) if f.endswith('.tif')]
# sumatra_files = [f for f in all_masks if not f.startswith(('kalimanatan'))]
# kali_files = [f for f in all_masks if f.startswith(('kalimanatan'))]

# 2. Salin Sumatra (ASLI)
# print(f"Menyalin {len(sumatra_files)} data Sumatra...")
# for f in tqdm(sumatra_files):
#     for folder in FOLDERS:
#         src = os.path.join(BASE_DIR, folder, f)
#         dst = os.path.join(OUTPUT_DIR, folder, f)
#         if os.path.exists(src):
#             shutil.copy2(src, dst)

# 3. Filter Kalimantan (Target: Minimalkan Background)
print(f"Memproses kalimantan dan sumatera (Filter Background < {MAX_BACKGROUND*100}%)...")
count_kali_added = 0

for f in tqdm(all_masks):
    mask_path = os.path.join(BASE_DIR, 'mask', f)
    try:
        ratios = get_pixel_ratios(mask_path)
        
        if ratios[0] < MAX_BACKGROUND:
            for folder in FOLDERS:
                src = os.path.join(BASE_DIR, folder, f)
                dst = os.path.join(OUTPUT_DIR, folder, f)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            count_kali_added += 1
            
    except Exception as e:
        print(f"Error pada {f}: {e}")

# print(f"Total Sumatra: {len(sumatra_files)}")
print(f"Total data Lolos ( kalimantan + sumatera ) (Padat Konten): {count_kali_added}")
# print(f"Total Dataset Baru: {len(sumatra_files) + count_kali_added}")