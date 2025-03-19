"""Custom data loader for MIL (attention and transformer methods)"""
import torch
from torch.utils.data import Dataset

class MILBagDataset(Dataset):
    """
    Custom dataset class for Multiple Instance Learning (MIL).

    This dataset handles loading preprocessed `.pt` files representing bags of instances.
    Each bag consists of multiple feature vectors (instances) associated with a single label.
    It is used for MIL tasks where the model needs to infer a label from multiple instances.

    Args:
        input_path (str): Path to the directory containing the `.pt` files.
        files_pt (list of str): List of `.pt` file names.
        target (pd.DataFrame): DataFrame containing labels with columns 'filename' and 'target'.

    Attributes:
        bags (list of torch.Tensor): List of instance tensors, each of shape (N_instances, D_features).
        labels (list of int/float): List of labels corresponding to each bag.
        filenames (list of str): List of filenames corresponding to each bag.

    Methods:
        __len__(): Returns the number of bags in the dataset.
        __getitem__(idx): Returns the bag, label, and filename at index `idx`.

    """

    def __init__(self, input_path, files_pt, target):
        self.bags = []
        self.labels = []
        self.filenames = []

        for file in files_pt:
            basename = file.split(".pt")[0]
            data = torch.load(f"{input_path}/pt_files/{file}") 
            
            if data.ndim == 1:
                data = data.unsqueeze(0)
            
            label = target.loc[target['filename'] == basename, 'target'].values
            if len(label) == 0:
                continue  # Skip if no label found

            self.bags.append(data)
            self.labels.append(label[0])
            self.filenames.append(basename)

    def __len__(self):
        """
        Returns the number of bags in the dataset.

        Returns:
            int: Number of bags.
        """
        return len(self.bags)

    def __getitem__(self, idx):
        """
        Retrieves the bag, label, and filename at a given index.

        Args:
            idx (int): Index of the bag.

        Returns:
            torch.Tensor: Tensor of instances for the given bag.
            int/float: Label associated with the bag.
            str: Filename of the bag.
        """
        return self.bags[idx], self.labels[idx], self.filenames[idx]

