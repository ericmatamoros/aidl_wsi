import torch
from torch.utils.data import Dataset

class MILBagDataset(Dataset):
    """Custom dataset class for Multiple Instance Learning (MIL)"""
    def __init__(self, input_path, files_pt, target):
        self.bags = []
        self.labels = []
        self.filenames = []

        for file in files_pt:
            basename = file.split(".pt")[0]
            data = torch.load(f"{input_path}/pt_files/{file}")  # Shape: (N_instances, D_features)
            
            if data.ndim == 1:  # Ensure 2D structure
                data = data.unsqueeze(0)
            
            label = target.loc[target['filename'] == basename, 'target'].values
            if len(label) == 0:
                continue  # Skip if no label found

            self.bags.append(data)
            self.labels.append(label[0])
            self.filenames.append(basename)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx], self.filenames[idx]  # Return basename

