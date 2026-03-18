import os
import torch
import rasterio
import numpy as np

from torch.utils.data import Dataset, DataLoader, Subset


class ChangeDetectionDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.t1_dir = os.path.join(root_dir, "t1")
        self.t2_dir = os.path.join(root_dir, "t2")
        self.mask_dir = os.path.join(root_dir, "mask")

        self.ids = sorted([
            f for f in os.listdir(self.t1_dir)
            if f.endswith(".tif")
        ])

        for file_id in self.ids:
            assert os.path.exists(os.path.join(self.t2_dir, file_id)), f"Missing in T2: {file_id}"
            assert os.path.exists(os.path.join(self.mask_dir, file_id)), f"Missing in MASK: {file_id}"
        
        assert len(self.ids) == len([
            f for f in os.listdir(self.t2_dir) if f.endswith(".tif")
        ])
        assert len(self.ids) == len([
            f for f in os.listdir(self.mask_dir) if f.endswith(".tif")
        ])

    def __len__(self):
        return len(self.ids)


    def read_image(self, path):

        with rasterio.open(path) as src:
            img = src.read()
        
        img = img / 10000.0
        img = img.astype(np.float32)

        return img


    def read_mask(self, path):

        with rasterio.open(path) as src:
            mask = src.read(1)

        mask = mask.astype(np.int64)

        return mask


    def __getitem__(self, idx):

        file_id = self.ids[idx]

        t1_path = os.path.join(self.t1_dir, file_id)
        t2_path = os.path.join(self.t2_dir, file_id)
        mask_path = os.path.join(self.mask_dir, file_id)

        t1 = self.read_image(t1_path)
        t2 = self.read_image(t2_path)
        mask = self.read_mask(mask_path)

        t1 = torch.from_numpy(t1)
        t2 = torch.from_numpy(t2)
        mask = torch.from_numpy(mask)

        assert t1.shape == t2.shape, f"Shape mismatch: {file_id}"
        assert t1.shape[1:] == mask.shape, f"Mask mismatch: {file_id}"

        if not torch.isfinite(t1).all() or not torch.isfinite(t2).all():
            raise ValueError(f"Invalid value in image: {file_id}")

        return t1, t2, mask, file_id


# ====================================
# SEQUENTIAL SPLIT DATALOADER
# ====================================

def create_dataloaders(root_dir, batch_size=8):

    dataset = ChangeDetectionDataset(root_dir)

    dataset_size = len(dataset)

    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, dataset_size))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
