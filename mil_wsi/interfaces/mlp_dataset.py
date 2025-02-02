import torch
from torch.utils.data import Dataset, DataLoader


class MLPDataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features.values, dtype=torch.float32)  # Convert to tensor
        self.target = torch.tensor(target.values, dtype=torch.float32)  # Convert to tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]