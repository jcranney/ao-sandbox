import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Optional

if os.environ.get("TORCH_DEVICE"):
    device = int(os.environ["TORCH_DEVICE"])
else:
    device = "cpu"

class CustomImageDataset(Dataset):
    def __init__(self, imgs, modes):
        self.imgs = imgs
        self.modes = modes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.modes[idx]
        return img, label


class Data(BaseModel):
    train_dataloader: Optional[DataLoader] = None
    test_dataloader: Optional[DataLoader] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(
        self, filename: str, *args, split: float = 0.8,
        batch_size: int = 1000, **kwargs
    ):
        super().__init__(*args, **kwargs)
        data = np.load(filename)
        modes = torch.tensor(data["modes"].astype(np.float32), device=device)
        imgs = torch.tensor(data["imgs"].astype(np.float32), device=device)
        
        idx = int(len(modes)*split)
        self.train_dataloader = DataLoader(
            CustomImageDataset(imgs[:idx], modes[:idx]),
            batch_size=batch_size, shuffle=True,
        )
        self.test_dataloader = DataLoader(
            CustomImageDataset(imgs[idx:], modes[idx:]),
            batch_size=batch_size, shuffle=True
        )
