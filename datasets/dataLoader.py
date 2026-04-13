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
        # 1. Pastikan folder sinkron dengan preprocess terbaru
        self.cache_dir = os.path.join(root_dir, "cache_npy") 
        self.ids = np.load(os.path.join(self.cache_dir, "ids.npy"))
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]
        npy_path = os.path.join(self.cache_dir, file_id.replace(".tif", ".npz"))

        try:
            with np.load(npy_path) as data:
                # Ambil data dan pastikan tidak ada NaN yang lolos
                t1 = torch.from_numpy(np.nan_to_num(data["t1"], nan=-1.0)).float()
                t2 = torch.from_numpy(np.nan_to_num(data["t2"], nan=-1.0)).float()
                mask = torch.from_numpy(data["mask"]).long()
        except Exception as e:
            # Jika ada file rusak yang lolos ke sini, print namanya agar kamu bisa hapus nanti
            print(f"❌ Error loading file {file_id}: {e}")
            # Return tensor nol agar loop tidak berhenti (atau handle sesuai kebutuhan)
            return torch.zeros((10, 256, 256)), torch.zeros((10, 256, 256)), torch.zeros((256, 256)), file_id

        # --- AUGMENTASI ---
        if self.augment:
            if random.random() > 0.5:
                t1 = torch.flip(t1, dims=[2])
                t2 = torch.flip(t2, dims=[2])
                mask = torch.flip(mask, dims=[1])
            if random.random() > 0.5:
                t1 = torch.flip(t1, dims=[1])
                t2 = torch.flip(t2, dims=[1])
                mask = torch.flip(mask, dims=[0])

        return t1, t2, mask, file_id

# ====================================
# FUNGSI PEMBUAT DATALOADER
# ====================================
def create_dataloaders(root_dir, batch_size=8):
    cache_dir = os.path.join(root_dir, "cache_npy")
    ids = np.load(os.path.join(cache_dir, "ids.npy"))
    labels = np.load(os.path.join(cache_dir, "labels.npy"))
    indices = np.arange(len(ids))

    # Split 1: 80% Train, 20% Sisa (temp)
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Split 2: Membagi 20% sisa menjadi 10% Val dan 10% Test
    # Ditambahkan _, _ untuk menangkap labels (menghindari ValueError unpack)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Instance Dataset: Pisahkan antara Train (Augment) dan Eval (Original)
    train_ds_master = ChangeDetectionDataset(root_dir, augment=True)
    eval_ds_master = ChangeDetectionDataset(root_dir, augment=False)

    # Gunakan Subset agar index tetap sinkron
    train_dataset = Subset(train_ds_master, train_idx)
    val_dataset   = Subset(eval_ds_master, val_idx)
    test_dataset  = Subset(eval_ds_master, test_idx)

    # Inisialisasi Loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader