import os
import argparse
import json
import pandas as pd
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

from mil_wsi.interfaces import (
    compute_metrics,
    train_mlp,
    predict_mlp,
    MLPDataset,
    MLP,
    plot_loss
)

import warnings
warnings.filterwarnings("ignore")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP model')
    parser.add_argument('--dir_results', type=str, help='Path to folder containing the results')
    parser.add_argument('--dir_data', type=str, help='Path containing slides')
    parser.add_argument('--dir_model', type=str, help='Path to store the trained models')
    parser.add_argument('--dir_metrics', type=str, help='Path to store metrics')
    parser.add_argument('--model_name', type=str, default='mlp_model', help='Name of the model')
    parser.add_argument('--predictions_name', type=str, default='predictions', help='Name for predictions file')
    parser.add_argument('--metrics_name', type=str, default='metrics', help='Name for metrics file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_size', type=float, default=128, help='Hidden size of the MLP network')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--k_folds', type=int, default=4, help='Number of train-test splits to perform')


    args = parser.parse_args()

    # Define directories and create them if they don't exist
    input_path = args.dir_results
    data_path = args.dir_data
    model_path = args.dir_model
    suffix_name = f"MLP_bs{args.batch_size}_hs{args.hidden_size}_ep{args.epochs}_ts{args.test_size}_kf{args.k_folds}"
    metrics_path = f"{args.dir_metrics}/{suffix_name}"
    loss_graph_path = f"{args.dir_metrics}/{suffix_name}/losses_graphs"

    os.makedirs(input_path, exist_ok=True)
    #os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(loss_graph_path, exist_ok=True)

    model_name = f"{args.model_name}{suffix_name}"
    predictions_name = f"{args.predictions_name}{suffix_name}"
    metrics_name = f"{args.metrics_name}{suffix_name}"

    files_pt = os.listdir(f"{input_path}/pt_files")

    logger.info("Reading data and generating data loaders")
    # Read target CSV file
    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)
    # Uncomment the following line to invert target values if needed (positive class = 1)
    # target['target'] = 1 - target['target']
    target_counts = target['target'].value_counts()
    print("Target Value Distribution:")
    for value, count in target_counts.items():
        print(f"Value {value}: {count} instances")

    # Load data from .pt files and merge with target
    df = load_data(input_path, files_pt, target)

    # Separate features and target
    features = df.drop(columns=['filename', 'target'])
    input_size = features.shape[1]
    print("input size:")
    print(input_size)
    target = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=args.test_size, random_state=4, stratify=target
    )

    # Define the number of folds for cross-validation (change if needed)
    n_splits = args.k_folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Variables to store metrics and losses for each fold
    all_metrics = []
    all_predictions = []
    train_losses_total = []
    val_losses_total = []

    # Convert training data to NumPy arrays for fold indexing. This is necessary because StratifiedKFold's split() method expects NumPy arrays rather than DataFrames.
    X_np = X_train.to_numpy()
    y_np = y_train.to_numpy()

    logger.info("Starting K-Fold Cross-Validation")

    for fold, (train_index, val_index) in enumerate(skf.split(X_np, y_np)):
        logger.info(f"Fold {fold + 1}/{n_splits}")

        # Split data into training and validation sets for this fold
        X_train_fold, X_val_fold = X_np[train_index], X_np[val_index]
        y_train_fold, y_val_fold = y_np[train_index], y_np[val_index]

        # Convert NumPy arrays to Pandas DataFrame and Series
        X_train_fold = pd.DataFrame(X_train_fold, columns=features.columns)
        y_train_fold = pd.Series(y_train_fold, name=target.name)
        X_val_fold = pd.DataFrame(X_val_fold, columns=features.columns)
        y_val_fold = pd.Series(y_val_fold, name=target.name)

        # Create datasets and dataloaders for this fold
        train_dataset = MLPDataset(X_train_fold, y_train_fold)
        val_dataset = MLPDataset(X_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Instantiate the model for this fold
        model = MLP(input_size=input_size, hidden_size=args.hidden_size, output_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set up the loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        n_epochs = args.epochs

        # Train the model for this fold
        model, train_losses, val_losses = train_mlp(
            model, train_loader, val_loader, criterion, optimizer, device, n_epochs
        )

        # Store training and validation losses for this fold
        train_losses_total.append(train_losses)
        val_losses_total.append(val_losses)

        # Evaluate on the validation set for this fold
        model.eval()
        predictions = predict_mlp(model, val_loader, device).cpu().numpy()
        predictions = predictions.round().astype(int)
        all_predictions.append(predictions)

        # Save fold predictions to CSV
        fold_preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': y_val_fold})
        fold_preds.to_csv(f"{metrics_path}/{predictions_name}_fold{fold + 1}.csv", index=False)

        # Calculate and store metrics for this fold
        fold_metrics = compute_metrics(predictions, y_val_fold)
        all_metrics.append(fold_metrics)

    # Average validation metrics across folds
    final_metrics = {
        key: {"mean": np.mean([m[key] for m in all_metrics]), "std": np.std([m[key] for m in all_metrics])}
        for key in all_metrics[0].keys()
    }

    # Save final averaged validation metrics to JSON
    with open(f"{metrics_path}/{metrics_name}_kfold.json", 'w') as json_file:
        json.dump(final_metrics, json_file, indent=4)

    print(f"Saving final averaged validation metrics to: {metrics_path}/{metrics_name}_kfold.json")
    logger.info("K-Fold Cross-Validation Completed")

    # Plot training and validation loss graphs
    plot_loss(train_losses_total, loss_graph_path, suffix_name, "train")
    plot_loss(val_losses_total, loss_graph_path, suffix_name, "val")

    # Evaluation on the test set using the model from the last fold
    logger.info("Evaluating on final test set")
    test_dataset = MLPDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    logger.info("Performing predictions using MLP model")
    predictions = predict_mlp(model, test_loader, device)
    predictions = predictions.cpu().numpy()
    predictions = predictions.round().astype(int)

    # Save test set predictions to CSV
    preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': y_test})
    preds.to_csv(f"{metrics_path}/{predictions_name}_test.csv", index=False)

    logger.info("Computing classification metrics")
    metrics = compute_metrics(predictions, y_test)
    metrics['confusion_matrix'] = {
        f"Actual_{i}-Predicted_{j}": int(metrics['confusion_matrix'].iloc[i, j])
        for i in range(metrics['confusion_matrix'].shape[0])
        for j in range(metrics['confusion_matrix'].shape[1])
    }
    with open(f'{metrics_path}/{metrics_name}_test.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    print(f"Saving predictions to: {metrics_path}/{predictions_name}_test.csv")
    print(f"Saving metrics to: {metrics_path}/{metrics_name}_test.json")
