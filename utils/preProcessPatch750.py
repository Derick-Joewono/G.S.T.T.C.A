import os
import numpy as np
from tqdm import tqdm

def compute_patch_stats(cache_dir):
    ids = np.load(os.path.join(cache_dir, "ids.npy"))

    stats = []

    for file_id in tqdm(ids):
        npy_path = os.path.join(cache_dir, file_id.replace(".tif", ".npz"))
        data = np.load(npy_path)
        mask = data["mask"]

        total = mask.size
        c0 = (mask == 0).sum()
        c1 = (mask == 1).sum()
        c2 = (mask == 2).sum()

        change = c1 + c2

        stats.append({
            "id": file_id,
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "change_ratio": change / total
        })

    return stats


def select_750_with_pixel_ratio(stats, target_ratio=0.1, total_patches=750):
    # sort patch dari yang paling banyak change
    stats = sorted(stats, key=lambda x: x["change_ratio"], reverse=True)

    selected = []

    total_c0, total_c1, total_c2 = 0, 0, 0

    for s in stats:
        if len(selected) >= total_patches:
            break

        # coba tambah patch ini
        new_c0 = total_c0 + s["c0"]
        new_c1 = total_c1 + s["c1"]
        new_c2 = total_c2 + s["c2"]

        total_pixels = new_c0 + new_c1 + new_c2
        new_ratio = (new_c1 + new_c2) / total_pixels

        # greedy condition:
        # jangan terlalu jauh dari target (10%)
        if new_ratio <= target_ratio + 0.02:  # tolerance
            selected.append(s["id"])
            total_c0, total_c1, total_c2 = new_c0, new_c1, new_c2

    print("Selected:", len(selected))

    final_pixels = total_c0 + total_c1 + total_c2
    final_ratio = (total_c1 + total_c2) / final_pixels

    print("Final ratio:", final_ratio)

    return selected
