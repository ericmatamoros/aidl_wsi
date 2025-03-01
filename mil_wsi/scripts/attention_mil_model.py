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
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from openslide import OpenSlide

from mil_wsi.interfaces import (
    compute_metrics, 
    train_attention_mil, 
    predict_attention_mil, 
    MILBagDataset, 
    AttentionMIL,
    MultiHeadAttention,
    AttentionMILMLP
)

import warnings
warnings.filterwarnings("ignore")

def visualize_attention(all_attn_weights, all_filenames, predictions, data_path, suffix, threshold=0.5, patch_size=224):
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
    print(all_filenames)

    for i, (attn_weights, wsi_name) in enumerate(zip(all_attn_weights, all_filenames)):
        if int(predictions[i]) == 1:  # Only highlight for cancer predictions
            print(attn_weights)
            wsi_img_path = os.path.join(f"{data_path}images/", f"{wsi_name}.svs")
            mask_img_path = os.path.join(f"{data_path}patches/masks/", f"{wsi_name}.jpg")
            h5_patch_path = os.path.join(f"{data_path}patches/patches/", f"{wsi_name}.h5")

            if os.path.exists(mask_img_path) and os.path.exists(h5_patch_path) and os.path.exists(wsi_img_path):  # Ensure WSI, patches and masks exist
                mask_img = plt.imread(mask_img_path)  # Load WSI image

                # Load patch coordinates from the .h5 file
                with h5py.File(h5_patch_path, "r") as f:
                    patches = f["coords"][:]

                # Normalize attention scores
                attn_scores = attn_weights[0].flatten()
                print(attn_scores)
                attn_scores = (attn_scores - np.min(attn_scores)) / (np.max(attn_scores) - np.min(attn_scores) + 1e-8)

                # Get WSI dimensions (px)
                slide = OpenSlide(wsi_img_path)
                WSI_WIDTH, WSI_HEIGHT = slide.dimensions
                # Get mask dimensions (px)
                MASK_HEIGHT, MASK_WIDTH = mask_img.shape[:2]

                # Calculate scaling factors
                SCALE_X = MASK_WIDTH / WSI_WIDTH
                SCALE_Y = MASK_HEIGHT / WSI_HEIGHT

                # Create empty heatmap
                heatmap = np.zeros((MASK_HEIGHT, MASK_WIDTH), dtype=np.float32)

                # Overlay attention scores at correct WSI locations
                for (x, y), attn in zip(patches, attn_scores):
                    assert len(patches) == len(attn_scores), f"Mismatch: {len(patches)} patches vs {len(attn_scores)} attn scores"

                    # Adjust the coordinates and patch size
                    x, y = int(x*SCALE_X), int(y*SCALE_Y)
                    print(f"Coordinates: {x}, {y} with attention {attn}")
                    heatmap[y:y + patch_size, x:x + patch_size] += attn  # Accumulate attention
                
                print(f"For the image: {wsi_name} with dimentions: {MASK_HEIGHT}, {MASK_WIDTH}")
                
                # Normalize heatmap to [0, 1]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                # Resize heatmap to match WSI dimensions
                heatmap_resized = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smoothen heatmap
                heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # Blend heatmap with original WSI
                overlay = cv2.addWeighted(mask_img, 0.6, heatmap_colored, 0.4, 0)

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



# Argument Parser
parser = argparse.ArgumentParser(description='Attention MIL model with K-Fold Cross-Validation')
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
parser.add_argument('--k_folds', type=int, default=4, help='Number of K-fold splits')
parser.add_argument('--highlight_threshold', type=float, default=0.5, help='Threshold for highlighting patches in WSI')

if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.dir_results
    data_path = args.dir_data
    model_path = args.dir_model
    metrics_path = f"{args.dir_metrics}/{args.suffix_name}"
    os.makedirs(metrics_path, exist_ok=True)

    model_name = f"{args.model_name}{args.suffix_name}"
    predictions_name = f"{args.predictions_name}{args.suffix_name}"
    metrics_name = f"{args.metrics_name}{args.suffix_name}"

    files_pt = os.listdir(f"{input_path}/pt_files")

    logger.info("Reading data and generating data loaders")
    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)

    dataset = MILBagDataset(input_path, files_pt, target)
    n_splits = args.k_folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    targets = [dataset[i][1] for i in range(len(dataset))]

    
    all_metrics = []

    logger.info("Starting K-Fold Cross-Validation")
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        logger.info(f"Fold {fold + 1}/{n_splits}")

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        input_size = next(iter(train_loader))[0].shape[-1]

        model = AttentionMILMLP (input_size=input_size, hidden_size=args.hidden_size, attention_class="AttentionMIL")
        #model = AttentionMILMLP (input_size=input_size, hidden_size=args.hidden_size, attention_class="MultiHeadAttention", n_heads = 10)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model, attn_weights = train_attention_mil(model, train_loader,  criterion, optimizer, device, args.epochs)
        
        logger.info("Performing validation predictions")
        predictions, attn_weights, bag_ids = predict_attention_mil(model, val_loader, device)
        predictions = predictions.cpu().numpy().round().astype(int)
        
        fold_metrics = compute_metrics(predictions, [y for _, y, _ in val_dataset])
        all_metrics.append(fold_metrics)
        
        fold_preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': [y for _, y, _ in val_dataset]})
        fold_preds.to_csv(f"{metrics_path}/{predictions_name}_fold{fold + 1}.csv", index=False)
    
    final_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    with open(f"{metrics_path}/{metrics_name}_kfold.json", 'w') as json_file:
        json.dump(final_metrics, json_file, indent=4)

    logger.info("Evaluating on final test set")
    test_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    predictions, attn_weights, bag_ids = predict_attention_mil(model, test_loader, device)
    predictions = predictions.cpu().numpy().round().astype(int)

    visualize_attention(attn_weights, bag_ids, predictions, data_path, args.suffix_name, args.highlight_threshold)

    preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': [y for _, y, _ in test_dataset]})
    preds.to_csv(f"{metrics_path}/{predictions_name}_test.csv", index=False)

    metrics = compute_metrics(predictions, [y for _, y, _ in test_dataset])
    metrics['confusion_matrix'] = {
        f"Actual_{i}-Predicted_{j}": int(metrics['confusion_matrix'].iloc[i, j])
        for i in range(metrics['confusion_matrix'].shape[0])
        for j in range(metrics['confusion_matrix'].shape[1])
    }
    with open(f'{metrics_path}/{metrics_name}_test.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    logger.info("K-Fold Cross-Validation Completed")