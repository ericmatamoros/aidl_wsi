import torch
from pathlib import Path
import h5py



def func1(): 

    folder_path = Path('./mil_wsi/results/pt_files/')

    for file_path in folder_path.glob('*.pt'):
        emb_tensor = torch.load(file_path)
        print(emb_tensor.size())

def func2(): 

    folder_path = Path('./mil_wsi/results/h5_files/')
    # Open the .h5 file

    for file_path in folder_path.glob('*.h5'):
        with h5py.File(file_path, 'r') as f:
            print("Keys in the HDF5 file:")
            for coords in f.keys():
                item = f[coords]
                print(item)

