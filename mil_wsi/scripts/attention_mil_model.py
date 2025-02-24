import os
import argparse
import json
import h5py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from sklearn.model_selection import train_test_split
from mil_wsi.interfaces import compute_metrics, train_attention_mil, predict_attention_mil, MILBagDataset, AttentionMIL
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Argument Parser
parser = argparse.ArgumentParser(description='Attention MIL model with Explainability')
parser.add_argument('--dir_results', type=str, help='Path to folder containing the results')
parser.add_argument('--dir_data', type=str, help='Path containing slides')
parser.add_argument('--dir_model', type=str, help='Path to store the trained models')
parser.add_argument('--dir_metrics', type=str, help='Path to store metrics')
parser.add_argument('--model_name', type=str, default='mil_model', help='Name of the model')
parser.add_argument('--predictions_name', type=str, default='predictions', help='Name for predictions file')
parser.add_argument('--suffix_name', type=str, help='Name suffix for the experiment')
parser.add_argument('--metrics_name', type=str, default='metrics', help='Name for metrics file')
parser.add_argument('--batch_size', type=int, default=1, help='Size of the batch (1 for MIL)')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the MIL network')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--test_size', type=float, default=0.3, help='Test set fraction')
parser.add_argument('--highlight_threshold', type=float, default=0.5, help='Threshold for highlighting patches in WSI')

import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

def visualize_attention(all_attn_weights, all_filenames, predictions, input_path, suffix, threshold=0.5, patch_size=224):
    """
    Save WSI images with highlighted patches based on attention scores.

    Args:
        all_attn_weights (list): List of attention scores for each WSI.
        all_filenames (list): List of WSI filenames corresponding to each attention score.
        predictions (list): Model predictions (0 or 1).
        input_path (str): Path to the WSI images.
        suffix (str): Experiment suffix for saving results.
        threshold (float): Threshold for highlighting patches.
        patch_size (int): Size of each patch extracted from the WSI.
    """
    explainability_dir = f"explainability{suffix}"
    os.makedirs(explainability_dir, exist_ok=True)

    for i, (attn_weights, wsi_name) in enumerate(zip(all_attn_weights, all_filenames)):
        if int(predictions[i]) == 1:  # Only highlight for cancer predictions
            wsi_img_path = os.path.join(f"{input_path}/masks/", f"{wsi_name[0]}.jpg")
            h5_patch_path = os.path.join(f"{input_path}/patches/", f"{wsi_name[0]}.h5")

            if os.path.exists(wsi_img_path) and os.path.exists(h5_patch_path):  # Ensure both WSI and patches exist
                wsi_img = plt.imread(wsi_img_path)  # Load WSI image

                # Load patch coordinates from the .h5 file
                with h5py.File(h5_patch_path, "r") as f:
                    patches = f["coords"][:]

                # Normalize attention scores
                attn_scores = attn_weights[0].flatten()
                attn_scores = (attn_scores - np.min(attn_scores)) / (np.max(attn_scores) - np.min(attn_scores) + 1e-8)

                # Get WSI dimensions
                wsi_h, wsi_w = wsi_img.shape[:2]

                # Create empty heatmap
                heatmap = np.zeros((wsi_h, wsi_w), dtype=np.float32)

                # Overlay attention scores at correct WSI locations
                for (x, y), attn in zip(patches, attn_scores):
                    x, y = int(x), int(y)
                    heatmap[y:y + patch_size, x:x + patch_size] += attn  # Accumulate attention

                # Normalize heatmap to [0, 1]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                # Resize heatmap to match WSI dimensions
                heatmap_resized = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smoothen heatmap
                heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # Blend heatmap with original WSI
                overlay = cv2.addWeighted(wsi_img, 0.6, heatmap_colored, 0.4, 0)

                # Apply threshold mask for high-attention patches
                highlight_mask = heatmap_resized > threshold
                contours, _ = cv2.findContours(highlight_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Green contours

                # Save the explainability image
                plt.figure(figsize=(10, 10))
                plt.imshow(overlay)
                plt.axis("off")
                plt.title(f"WSI {wsi_name} - Highlighted Patches")
                plt.savefig(os.path.join(explainability_dir, f"{wsi_name[0]}.jpg"), bbox_inches="tight", dpi=300)
                plt.close()
            else:
                print(f"Skipping {wsi_name}: WSI or patches file not found.")



# Main MIL Training & Prediction Pipeline
if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.dir_results
    data_path = args.dir_data
    model_path = args.dir_model
    metrics_path = args.dir_metrics
    explainability_path = f"explainability{args.suffix_name}"

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(explainability_path, exist_ok=True)

    model_name = f"{args.model_name}{args.suffix_name}"
    predictions_name = f"{args.predictions_name}{args.suffix_name}"
    metrics_name = f"{args.metrics_name}{args.suffix_name}"

    files_pt = os.listdir(f"{input_path}/pt_files")

    logger.info("Reading data and generating data loaders")
    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)

    # Create dataset
    dataset = MILBagDataset(input_path, files_pt, target)

    # Train-test split
    train_size = int((1 - args.test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = next(iter(train_loader))[0].shape[-1]
    hidden_size = args.hidden_size
    model = AttentionMIL(input_size=input_size, hidden_size=hidden_size, output_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(f"{model_path}{model_name}.pth"):
        logger.info("Training Attention MIL model")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = train_attention_mil(model, train_loader, criterion, optimizer, device, args.epochs)
        torch.save(model.state_dict(), f"{model_path}{model_name}.pth")
    else:
        logger.info("Loading trained model")
        model.load_state_dict(torch.load(f"{model_path}{model_name}.pth"))

    model.eval()

    logger.info("Performing predictions with explainability")
    predictions, attn_weights, bag_ids = predict_attention_mil(model, test_loader, device)
    predictions = predictions.cpu().numpy()

    visualize_attention(attn_weights, bag_ids, predictions, input_path, args.suffix_name, args.highlight_threshold)

    logger.info("Computing classification metrics")
    metrics = compute_metrics(predictions, [int(x.cpu().numpy()) for _, x, _ in test_loader])

    # Convert confusion matrix to structured JSON format
    metrics['confusion_matrix'] = {
        f"Actual_{i}-Predicted_{j}": int(metrics['confusion_matrix'].iloc[i, j])
        for i in range(metrics['confusion_matrix'].shape[0])
        for j in range(metrics['confusion_matrix'].shape[1])
    }

    # Save metrics to JSON file
    with open(f'{metrics_path}/{metrics_name}.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    logger.info(f"Explainability images saved in {explainability_path}")
