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
    counts = np.bincount(mask.flatten(), minlength=3)
    return counts / mask.size 

# 1. List File
all_masks = [f for f in os.listdir(os.path.join(BASE_DIR, 'mask')) if f.endswith('.tif')]
sumatra_files = [f for f in all_masks if not f.startswith(('kb_', 'kl_'))]
kali_files = [f for f in all_masks if f.startswith(('kb_', 'kl_'))]

# 2. Salin Sumatra (ASLI)
print(f"Menyalin {len(sumatra_files)} data Sumatra...")
for f in tqdm(sumatra_files):
    for folder in FOLDERS:
        src = os.path.join(BASE_DIR, folder, f)
        dst = os.path.join(OUTPUT_DIR, folder, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)

# 3. Filter Kalimantan (Target: Minimalkan Background)
print(f"Memproses Kalimantan (Filter Background < {MAX_BACKGROUND*100}%)...")
count_kali_added = 0

for f in tqdm(kali_files):
    mask_path = os.path.join(BASE_DIR, 'mask', f)
    try:
        ratios = get_pixel_ratios(mask_path)
        
        # LOGIKA BARU: Jika label 0 (Background) kurang dari 40%
        if ratios[0] < MAX_BACKGROUND:
            for folder in FOLDERS:
                src = os.path.join(BASE_DIR, folder, f)
                dst = os.path.join(OUTPUT_DIR, folder, f)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            count_kali_added += 1
            
    except Exception as e:
        print(f"Error pada {f}: {e}")

print(f"Total Sumatra: {len(sumatra_files)}")
print(f"Total Kalimantan Lolos (Padat Konten): {count_kali_added}")
print(f"Total Dataset Baru: {len(sumatra_files) + count_kali_added}")