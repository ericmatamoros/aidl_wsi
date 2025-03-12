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
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from openslide import OpenSlide

from mil_wsi.interfaces import (
    compute_metrics, 
    train_transformer_model, 
    predict_transformer_model, 
    MILBagDataset, 
    TransformerMIL,
    TransformerMarugoto
)

from ._explainability import visualize_attention

import warnings
warnings.filterwarnings("ignore")

# Argument Parser
parser = argparse.ArgumentParser(description='Transformer MIL model with K-Fold Cross-Validation')
parser.add_argument('--dir_results', type=str, help='Path to folder containing the results')
parser.add_argument('--dir_data', type=str, help='Path containing slides')
parser.add_argument('--dir_model', type=str, help='Path to store the trained models')
parser.add_argument('--dir_metrics', type=str, help='Path to store metrics')
parser.add_argument('--experiment_name', type = str,help='name of the experiment')
parser.add_argument('--model_name', type=str, default='mil_model', help='Name of the model')
parser.add_argument('--predictions_name', type=str, default='predictions', help='Name for predictions file')
parser.add_argument('--metrics_name', type=str, default='metrics', help='Name for metrics file')
parser.add_argument('--batch_size', type=int, default=1, help='Size of the batch (1 for MIL)')
parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--n_heads', type=int, default=4, help='Number of heads of the attention')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--output_size', type=int, default=1, help='Output size')
parser.add_argument('--k_folds', type=int, default=4, help='Number of K-fold splits')
parser.add_argument('--highlight_threshold', type=float, default=0.5, help='Threshold for highlighting patches in WSI')

if __name__ == '__main__':
    args = parser.parse_args()

    input_path = f"{args.dir_results}/{args.experiment_name}/"
    data_path = args.dir_data
    model_path = f"{args.dir_model}/{args.experiment_name}/"
    suffix_name = f"TransformerMIL_bs{args.batch_size}_ep{args.epochs}_ts{args.test_size}_kf{args.k_folds}_lr{args.learning_rate}_heads{args.n_heads}_os{args.output_size}"
    metrics_path = f"{args.dir_metrics}/{args.experiment_name}/{suffix_name}"
    os.makedirs(metrics_path, exist_ok=True)

    model_name = f"{args.model_name}{suffix_name}"
    predictions_name = f"{args.predictions_name}{suffix_name}"
    metrics_name = f"{args.metrics_name}{suffix_name}"

    files_pt = os.listdir(f"{args.dir_results}/pt_files_conch")

    logger.info("Reading data and generating data loaders")
    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)

    dataset = MILBagDataset(input_path, os.listdir(f"{args.dir_results}/pt_files_conch"), target)
    targets = [dataset[i][1] for i in range(len(dataset))]
    
    # Split dataset into train and test
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=args.test_size, stratify=targets, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    n_splits = args.k_folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_targets = [dataset[i][1] for i in train_idx]

    all_metrics = []

    logger.info("Starting K-Fold Cross-Validation")
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        logger.info(f"Fold {fold + 1}/{n_splits}")

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        input_size = next(iter(train_loader))[0].shape[-1] 
        
        #model = TransformerMIL(input_size=input_size, n_heads = args.n_heads, num_classes= args.output_size)
        model = TransformerMarugoto(input_size=input_size, n_heads = args.n_heads, num_classes= args.output_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        model, attn_weights = train_transformer_model(model, train_loader, criterion, optimizer, device, args.epochs)
        
        logger.info("Performing validation predictions")
        predictions, attn_weights, bag_ids = predict_transformer_model(model, val_loader, device)
        predictions = predictions.cpu().numpy().round().astype(int)
        
        fold_metrics = compute_metrics(predictions, [y for _, y, _ in val_dataset])
        all_metrics.append(fold_metrics)
        
        fold_preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': [y for _, y, _ in val_dataset]})
        fold_preds.to_csv(f"{metrics_path}/{predictions_name}_fold{fold + 1}.csv", index=False)
    
    final_metrics = {
        key: {"mean": np.mean([m[key] for m in all_metrics]), "std": np.std([m[key] for m in all_metrics])}
        for key in all_metrics[0].keys()
    }
    with open(f"{metrics_path}/{metrics_name}_kfold.json", 'w') as json_file:
        json.dump(final_metrics, json_file, indent=4)

    logger.info("Evaluating on final test set")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    predictions, attn_weights, bag_ids = predict_transformer_model(model, test_loader, device)
    predictions = predictions.cpu().numpy().round().astype(int)

    visualize_attention(attn_weights, bag_ids, predictions, data_path, suffix_name, args.highlight_threshold)

    preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': [y for _, y, _ in test_dataset]})
    preds.to_csv(f"{metrics_path}/{predictions_name}_test.csv", index=False)

    metrics = compute_metrics(predictions, [y for _, y, _ in test_dataset])
    with open(f'{metrics_path}/{metrics_name}_test.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    logger.info("K-Fold Cross-Validation Completed")