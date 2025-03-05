import os
import argparse
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from loguru import logger

from mil_wsi.interfaces import MILBagDataset

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# Argument Parser
parser = argparse.ArgumentParser(description='Feature Projection')
parser.add_argument('--dir_results', type=str, help='Path to folder containing the results')
parser.add_argument('--dir_data', type=str, help='Path containing slides')

if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.dir_results
    data_path = args.dir_data

    files_pt = os.listdir(f"{input_path}/pt_files")

    logger.info("Reading data and generating data loaders")
    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)

    dataset = MILBagDataset(input_path, os.listdir(f"{input_path}/pt_files"), target)
    targets = [dataset[i][1] for i in range(len(dataset))]

    dfs_list = []
    for bags, labels, filenames in dataset:
        df = pd.DataFrame(bags)
        df['target'] = labels
        df['filename'] = filenames
        dfs_list.append(df)
    dfs = pd.concat(dfs_list)
    dfs = dfs.groupby('target').sample(2000, random_state=SEED)

    features = dfs.iloc[:, :-2].values  # All columns except 'target' and 'filename'
    targets = dfs["target"].values

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
    