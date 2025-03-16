"""Script to project dimensions into a two-dimensional space through UMAP"""
import os
import argparse
import pandas as pd
import numpy as np
import umap
import torch
import matplotlib.pyplot as plt
from loguru import logger

from mil_wsi.interfaces import MLPDataset

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

def load_data(input_path: str, files_pt: list, target: pd.DataFrame) -> pd.DataFrame:
    """
    Load data from .pt files, compute the mean along dimension 0, and merge with the target.

    Args:
        input_path (str): Path to the results folder.
        files_pt (list): List of .pt filenames.
        target (pd.DataFrame): DataFrame containing the target information.

    Returns:
        pd.DataFrame: DataFrame with features and merged target.
    """
    df_list = []
    for file in files_pt:
        basename = file.split(".pt")[0]
        data = torch.load(f"{input_path}/pt_files/{file}")
        data = torch.mean(data, dim=0).numpy()  # Compute the mean along dimension 0 of the loaded tensor.
        df_dims = pd.DataFrame([data], columns=list(range(len(data))))
        df_dims['filename'] = basename
        df_dims = df_dims.merge(target.drop(columns='slide'), on='filename', how="left")
        df_list.append(df_dims)

    df = pd.concat(df_list).reset_index(drop=True)
    return df

# Argument Parser
parser = argparse.ArgumentParser(description='Feature Projection')
parser.add_argument('--dir_results', type=str, help='Path to folder containing the results')
parser.add_argument('--dir_data', type=str, help='Path containing slides')
parser.add_argument('--dimentions', type=int, default=1024, help='Number of latent spaces')

if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.dir_results
    data_path = args.dir_data
    dimentions = args.dimentions

    files_pt = os.listdir(f"{input_path}/pt_files")

    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)
    # Uncomment the following line to invert target values if needed (positive class = 1)
    target_counts = target['target'].value_counts()
    print("Target Value Distribution:")
    for value, count in target_counts.items():
        print(f"Value {value}: {count} instances")

    # Load data from .pt files and merge with target
    df = load_data(input_path, files_pt, target)

    features = df.iloc[:, 0:dimentions].values 
    targets = df["target"].values

    # Apply UMAP
    logger.info("Generating UMAP")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED)
    embedding = umap_reducer.fit_transform(features)

    # Plot UMAP projection
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap="coolwarm", alpha=0.6)
    plt.colorbar(scatter, label="Target")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP Projection Colored by Target")
    
    plt.savefig(f"{input_path}/feature_projection", dpi=300, bbox_inches="tight")   