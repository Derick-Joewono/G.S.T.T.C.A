import os
import torch
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader, Subset

# ====================================
# DATASET
# ====================================
class ChangeDetectionDataset(Dataset):

    def __init__(self, root_dir):

        self.cache_dir = os.path.join(root_dir, "cache_npy")

        self.ids = np.load(os.path.join(self.cache_dir, "ids.npy"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        file_id = self.ids[idx]
        npy_path = os.path.join(self.cache_dir, file_id.replace(".tif", ".npz"))

        data = np.load(npy_path)

        t1 = torch.from_numpy(data["t1"])
        t2 = torch.from_numpy(data["t2"])
        mask = torch.from_numpy(data["mask"])

        return t1, t2, mask, file_id


# ====================================
# DATALOADER
# ====================================
def create_dataloaders(root_dir, batch_size=8):

    cache_dir = os.path.join(root_dir, "cache_npy")

    ids = np.load(os.path.join(cache_dir, "ids.npy"))
    labels = np.load(os.path.join(cache_dir, "labels.npy"))

    indices = np.arange(len(ids))

    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )

    dataset = ChangeDetectionDataset(root_dir)

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)
    test_dataset  = Subset(dataset, test_idx)

    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
