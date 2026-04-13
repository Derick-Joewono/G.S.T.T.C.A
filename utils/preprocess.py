import os
import numpy as np
import rasterio
from tqdm import tqdm

def preprocess_to_npy(root_dir):
    # Setup Path
    t1_dir = os.path.join(root_dir, "t1")
    t2_dir = os.path.join(root_dir, "t2")
    mask_dir = os.path.join(root_dir, "mask")
    cache_dir = os.path.join(root_dir, "cache_npy_output")
    
    os.makedirs(cache_dir, exist_ok=True)

    # Ambil daftar file
    files = [f for f in os.listdir(t1_dir) if f.endswith(".tif")]
    
    labels = []
    ids = []

    print(f"🚀 Memulai Preprocessing {len(files)} file...")

    for f in tqdm(files, desc="Processing TIFF to NPZ"):
        npy_path = os.path.join(cache_dir, f.replace(".tif", ".npz"))
        
        try:
            # --- PROSES T1 ---
            with rasterio.open(os.path.join(t1_dir, f)) as src:
                data_t1 = src.read().astype(np.float32)
                mask_nodata_t1 = (data_t1 <= -1) # Tandai area bolong
                
                if np.max(data_t1) > 100:
                    data_t1 /= 10000.0
                
                t1 = np.clip(data_t1, 0, 1) # Amankan rentang 0-1
                t1[mask_nodata_t1] = -1.0   # Kembalikan -1 murni

            # --- PROSES T2 ---
            with rasterio.open(os.path.join(t2_dir, f)) as src:
                data_t2 = src.read().astype(np.float32)
                mask_nodata_t2 = (data_t2 <= -1)
                
                if np.max(data_t2) > 100:
                    data_t2 /= 10000.0
                
                t2 = np.clip(data_t2, 0, 1)
                t2[mask_nodata_t2] = -1.0

            # --- PROSES MASK ---
            with rasterio.open(os.path.join(mask_dir, f)) as src:
                mask = src.read(1).astype(np.int64)

            # Simpan ke NPZ (Compressed agar hemat storage)
            np.savez_compressed(npy_path, t1=t1, t2=t2, mask=mask)

            # --- LABELING OTOMATIS (Untuk Stratified Split) ---
            # Menghitung rasio piksel kelas 1 (Deforest) atau 2 (Forest)
            change_pixels = ((mask == 1) | (mask == 2)).sum()
            change_ratio = change_pixels / mask.size
            
            # Label 1 jika ada perubahan > 1% piksel dalam patch tersebut
            label = 1 if change_ratio > 0.01 else 0

            ids.append(f)
            labels.append(label)

        except Exception as e:
            print(f"⚠️ Error pada file {f}: {e}")
            continue

    # Simpan Metadata untuk DataLoader
    np.save(os.path.join(cache_dir, "ids.npy"), np.array(ids))
    np.save(os.path.join(cache_dir, "labels.npy"), np.array(labels))

    print(f"\n✅ Selesai! Data disimpan di: {cache_dir}")
    print(f"📊 Total Berhasil: {len(ids)} / {len(files)}")

# --- CARA JALANKAN ---
ROOT_PROJECT = r"D:\path\ke\dataset\kamu"
preprocess_to_npy(ROOT_PROJECT)