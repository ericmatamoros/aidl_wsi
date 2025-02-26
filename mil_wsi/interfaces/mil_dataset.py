import torch
from torch.utils.data import Dataset

class MILDataset(Dataset):
    def __init__(self, bags, labels):
        """
        Args:
            bags (list of torch.Tensor): Each bag is a tensor of shape (N, d), where
                                        N is the number of instances in the bag,
                                        and d is the feature dimension.
            labels (torch.Tensor): Labels for each bag (binary or multi-class).
        """
        self.bags = [torch.tensor(bag, dtype=torch.float32) for bag in bags]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """
        Returns the number of bags in the dataset.
        """
        return len(self.bags)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the bag to retrieve.
        
        Returns:
            tuple: (bag, label) where bag is a list of instances and label is the corresponding label.
        """
        bag = self.bags[idx]
        label = self.labels[idx]
        return bag, label