import numpy as np
import os
import rasterio
from tqdm import tqdm

def filter_nodata_tif_patches(root_dir, threshold):
    """
    Menganalisis file mentah .tif untuk membuang patch yang NoData (-1) > threshold.
    Menghasilkan ids_cleaned.npy dan labels_cleaned.npy dari data mentah.
    """
    # Sesuaikan dengan struktur folder mentah kamu
    t1_dir = os.path.join(root_dir, "t1")
    cache_dir = os.path.join(root_dir, "cache_npy_output") # Tempat simpan .npy baru
    
    # 1. Load file ID dan Label asli (hasil scanning folder awal)
    print("📂 Memuat metadata dataset awal...")
    ids = np.load(os.path.join(cache_dir, "ids.npy"))
    labels = np.load(os.path.join(cache_dir, "labels.npy"))
    
    os.makedirs(cache_dir, exist_ok=True)
    
    clean_ids = []
    clean_labels = []
    discarded_count = 0

    print(f"🔍 Menganalisis {len(ids)} file .tif dengan threshold NoData > {threshold*100}%...")

    # 2. Iterasi setiap patch .tif
    for i, file_id in enumerate(tqdm(ids)):
        # Pastikan file_id memiliki ekstensi .tif
        tif_name = file_id if file_id.endswith(".tif") else f"{file_id}.tif"
        tif_path = os.path.join(t1_dir, tif_name)

        if not os.path.exists(tif_path):
            # Jika tidak ada di t1, coba cek file_id aslinya
            continue

        try:
            # Menggunakan rasterio untuk membaca metadata dan data secara efisien
            with rasterio.open(tif_path) as src:
                # Kita hanya baca band pertama (1) untuk menghemat waktu & RAM
                # Data Sentinel-2 biasanya -1 untuk NoData
                band1 = src.read(1)
                
                total_pixels = band1.size
                nodata_count = np.sum(band1 == -1)
                nodata_ratio = nodata_count / total_pixels

                # 3. Filter Logic
                if nodata_ratio <= threshold:
                    clean_ids.append(file_id)
                    clean_labels.append(labels[i])
                else:
                    discarded_count += 1
                
        except Exception as e:
            print(f"⚠️ Error processing {tif_name}: {e}")
            continue

    # 4. Simpan hasil pembersihan
    new_ids = np.array(clean_ids)
    new_labels = np.array(clean_labels)

    np.save(os.path.join(cache_dir, "ids_cleaned.npy"), new_ids)
    np.save(os.path.join(cache_dir, "labels_cleaned.npy"), new_labels)

    print("\n✅ Pembersihan Data Mentah Selesai!")
    print(f"📊 Total Awal    : {len(ids)}")
    print(f"🗑️ Dibuang (Null) : {discarded_count}")
    print(f"✨ Total Bersih  : {len(new_ids)}")
    print(f"💾 Metadata bersih disimpan di {cache_dir}")
    print("-" * 30)

# --- CARA PAKAI ---
# Pastikan folder 't1' berisi file .tif mentah kamu (Sumatera + Kalimantan)
ROOT_DATASET = r"D:\DATASET_PROJECT" 
filter_nodata_tif_patches(ROOT_DATASET, threshold=0.30)