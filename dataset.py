import os
import torch
import rasterio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CloudDataset(Dataset):
    def __init__(self, data_dir, patch_ids, is_train=True):
        self.data_dir = data_dir
        self.patch_ids = patch_ids
        self.is_train = is_train
        self.train_dir = os.path.join(data_dir, "38-Cloud_training")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]

        # reads image patches
        bands = []
        band_names = ['red', 'green', 'blue', 'nir']
        
        for b in band_names:
            folder_name = f"train_{b}"
            file_name = f"{b}_{patch_id}.TIF"
            file_path = os.path.join(self.train_dir, folder_name, file_name)
            
            with rasterio.open(file_path) as src:
                band_data = src.read(1).astype(np.float32)
                bands.append(band_data)

        # merged all bands as array (4, 384, 384)
        image = np.stack(bands, axis=0)

        # normalization (0-1)
        image = np.clip(image / 65535.0, 0.0, 1.0)
        image_tensor = torch.from_numpy(image)

        # access binary mask
        if self.is_train:
            mask_name = f"gt_{patch_id}.TIF"
            mask_path = os.path.join(self.train_dir, "train_gt", mask_name)

            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
                mask = np.where(mask > 0, 1.0, 0.0)  # convert to 1.0 and 0.0
            
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, 384, 384)
            return image_tensor, mask_tensor
        
        return image_tensor

def get_dataloaders(data_dir, csv_path, batch_size=8, val_size=0.2):
    df = pd.read_csv(csv_path)
    patch_ids = df['name'].values

    # split
    train_ids, val_ids = train_test_split(patch_ids, test_size=val_size, random_state=42)

    train_dataset = CloudDataset(data_dir, train_ids, is_train=True)
    val_dataset = CloudDataset(data_dir, val_ids, is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader
