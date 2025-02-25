import os
import argparse
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from sklearn.model_selection import StratifiedKFold
import numpy as np
from mil_wsi.interfaces import compute_metrics, train_mil, predict_mil, MILBagDataset, MIL

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MIL model with K-Fold')
parser.add_argument('--dir_results', type=str, help='path to folder containing the results')
parser.add_argument('--dir_data', type=str, help='path containing slides')
parser.add_argument('--dir_model', type=str, help='path to store the trained models')
parser.add_argument('--dir_metrics', type=str, help='path to store metrics')
parser.add_argument('--model_name', type=str, default='mil_model', help='name of the model')
parser.add_argument('--predictions_name', type=str, default='predictions', help='name for predictions file')
parser.add_argument('--suffix_name', type=str, help='name suffix for the experiment')
parser.add_argument('--metrics_name', type=str, default='metrics', help='name for metrics file')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batch (1 for MIL)')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of the MIL network')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--k_folds', type=int, default=4, help='number of K-fold splits')

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
    
    # Create dataset
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
        model = MIL(input_size=input_size, hidden_size=args.hidden_size, output_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model = train_mil(model, train_loader, criterion, optimizer, device, args.epochs)
        
        logger.info("Performing validation predictions")
        predictions = predict_mil(model, val_loader, device)
        predictions = predictions.cpu().numpy().round().astype(int)

        fold_metrics = compute_metrics(predictions, [y for _, y, _  in val_dataset])
        all_metrics.append(fold_metrics)

        fold_preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': [y for _, y, _  in val_dataset]})
        fold_preds.to_csv(f"{metrics_path}/{predictions_name}_fold{fold + 1}.csv", index=False)

    final_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    with open(f"{metrics_path}/{metrics_name}_kfold.json", 'w') as json_file:
        json.dump(final_metrics, json_file, indent=4)

    logger.info("Evaluating on final test set")
    test_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    predictions = predict_mil(model, test_loader, device)
    predictions = predictions.cpu().numpy().round().astype(int)

    preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': [y for _, y, _  in test_dataset]})
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
