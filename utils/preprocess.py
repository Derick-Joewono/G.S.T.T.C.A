import os
import numpy as np
import rasterio

# ====================================
# PREPROCESS (TIFF -> NPZ + LABEL NPY)
# ====================================
def preprocess_to_npy(root_dir):

    t1_dir = os.path.join(root_dir, "t1")
    t2_dir = os.path.join(root_dir, "t2")
    mask_dir = os.path.join(root_dir, "mask")

    cache_dir = os.path.join(root_dir, "cache_npy")
    os.makedirs(cache_dir, exist_ok=True)

    files = [f for f in os.listdir(t1_dir) if f.endswith(".tif")]

    labels = []
    ids = []

    for f in files:

        npy_path = os.path.join(cache_dir, f.replace(".tif", ".npz"))

        with rasterio.open(os.path.join(t1_dir, f)) as src:
            t1 = src.read() / 10000.0

        with rasterio.open(os.path.join(t2_dir, f)) as src:
            t2 = src.read() / 10000.0

        with rasterio.open(os.path.join(mask_dir, f)) as src:
            mask = src.read(1)

        t1 = t1.astype(np.float32)
        t2 = t2.astype(np.float32)
        mask = mask.astype(np.int64)

        np.savez_compressed(npy_path, t1=t1, t2=t2, mask=mask)

        # 🔥 label otomatis
        change_ratio = ((mask == 1) | (mask == 2)).sum() / mask.size
        label = 1 if change_ratio > 0.01 else 0

        ids.append(f)
        labels.append(label)

    np.save(os.path.join(cache_dir, "ids.npy"), np.array(ids))
    np.save(os.path.join(cache_dir, "labels.npy"), np.array(labels))