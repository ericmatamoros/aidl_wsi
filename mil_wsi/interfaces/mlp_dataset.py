import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from loguru import logger


class MLPDataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features.values, dtype=torch.float32)  # Convert to tensor
        target = np.array(target.values, dtype=np.int64)  # Convert to float32
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]