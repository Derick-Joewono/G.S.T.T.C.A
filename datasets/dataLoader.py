import os
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

# ====================================
# DATASET (HANYA FLIP AUGMENTATION)
# ====================================
class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.cache_dir = os.path.join(root_dir, "cache_npy_output")
        self.ids = np.load(os.path.join(self.cache_dir, "ids.npy"))
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]
        npy_path = os.path.join(self.cache_dir, file_id.replace(".tif", ".npz"))

        # Load data (.npz)
        with np.load(npy_path) as data:
            t1 = torch.from_numpy(data["t1"]).float()
            t2 = torch.from_numpy(data["t2"]).float()
            mask = torch.from_numpy(data["mask"]).long()

        # --- LOGIKA AUGMENTASI (HANYA FLIP UNTUK MENJAGA STRUKTUR FITUR) ---
        if self.augment:
            # 1. Horizontal Flip (50% chance)
            if random.random() > 0.5:
                t1 = torch.flip(t1, dims=[2])
                t2 = torch.flip(t2, dims=[2])
                mask = torch.flip(mask, dims=[1])

            # 2. Vertical Flip (50% chance)
            if random.random() > 0.5:
                t1 = torch.flip(t1, dims=[1])
                t2 = torch.flip(t2, dims=[1])
                mask = torch.flip(mask, dims=[0])

        return t1, t2, mask, file_id

# ====================================
# FUNGSI PEMBUAT DATALOADER
# ====================================
def create_dataloaders(root_dir, batch_size=8):
    cache_dir = os.path.join(root_dir, "cache_npy_output")
    ids = np.load(os.path.join(cache_dir, "ids.npy"))
    labels = np.load(os.path.join(cache_dir, "labels.npy"))
    indices = np.arange(len(ids))

    # Split 1: 80% Train, 20% Sisa (temp)
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Split 2: Membagi 20% sisa menjadi 10% Val dan 10% Test
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Instance Dataset
    train_ds_master = ChangeDetectionDataset(root_dir, augment=True)
    eval_ds_master = ChangeDetectionDataset(root_dir, augment=False)

    # Subset
    train_dataset = Subset(train_ds_master, train_idx)
    val_dataset   = Subset(eval_ds_master, val_idx)
    test_dataset  = Subset(eval_ds_master, test_idx)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader