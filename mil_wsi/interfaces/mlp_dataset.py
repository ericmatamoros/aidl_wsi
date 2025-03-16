"""Custom data loader for MLP methods"""
import torch
import numpy as np
from torch.utils.data import Dataset

from loguru import logger


class MLPDataset(Dataset):
    """
    Custom dataset class for training an MLP model.

    This dataset is designed for supervised learning, where each sample consists 
    of a feature vector and a corresponding target label.

    Args:
        features (pd.DataFrame): DataFrame containing feature values.
        target (pd.Series or pd.DataFrame): Series or DataFrame containing target labels.

    Attributes:
        features (torch.Tensor): Feature tensors of shape (num_samples, num_features).
        target (torch.Tensor): Target labels as tensors of shape (num_samples,).

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the feature vector and target label at index `idx`.

    """
    def __init__(self, features, target):
        self.features = torch.tensor(features.values, dtype=torch.float32)  # Convert to tensor
        target = np.array(target.values, dtype=np.int64)  # Convert to float32
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieves the feature vector and target label at a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Feature tensor of shape (num_features,).
            torch.Tensor: Target label tensor.
        """
        return self.features[idx], self.target[idx]