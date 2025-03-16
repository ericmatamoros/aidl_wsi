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
    parser.add_argument('--experiment_name', type = str,
					help='name of the experiment')
    parser.add_argument('--dir_model', type=str, help='Path to store the trained models')
    parser.add_argument('--dir_metrics', type=str, help='Path to store metrics')
    parser.add_argument('--model_name', type=str, default='mlp_model', help='Name of the model')
    parser.add_argument('--predictions_name', type=str, default='predictions', help='Name for predictions file')
    parser.add_argument('--metrics_name', type=str, default='metrics', help='Name for metrics file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the MLP network')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--k_folds', type=int, default=4, help='Number of train-test splits to perform')
    parser.add_argument('--dimentions', type=int, help='Numer of latent spaces')


    args = parser.parse_args()

    # Define directories and create them if they don't exist
    input_path = f"{args.dir_results}/{args.experiment_name}/"
    data_path = f"{args.dir_data}{args.experiment_name}"
    suffix_name = f"MLP_bs{args.batch_size}_hs{args.hidden_size}_ep{args.epochs}_ts{args.test_size}_kf{args.k_folds}_lr{args.learning_rate}_dim{args.dimentions}"
    model_path = f"{args.dir_model}mlp/{args.experiment_name}/{suffix_name}"
    metrics_path = f"{args.dir_metrics}/{args.experiment_name}/{suffix_name}"
    loss_graph_path = f"{args.dir_metrics}/{args.experiment_name}/{suffix_name}/losses_graphs"

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(loss_graph_path, exist_ok=True)

    model_name = f"{args.model_name}{suffix_name}"
    predictions_name = f"{args.predictions_name}{suffix_name}"
    metrics_name = f"{args.metrics_name}{suffix_name}"

    files_pt = os.listdir(f"{input_path}/pt_files")

    print(data_path)

    logger.info("Reading data and generating data loaders")
    # Read target CSV file
    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)
    # Uncomment the following line to invert target values if needed (positive class = 1)
    # target['target'] = 1 - target['target']
    target_counts = target['target'].value_counts()
    logger.info("Target Value Distribution:")
    for value, count in target_counts.items():
        logger.info(f"Value {value}: {count} instances")

    # Load data from .pt files and merge with target
    df = load_data(input_path, files_pt, target)

    # Separate features and target
    #breakpoint()
    features = df.iloc[:, 0:args.dimentions]
    input_size = features.shape[1]
    target = df['target']
    num_classes = len(np.unique(target))

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

    best_val_metric = float(0)

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
        output_size = 1 if num_classes == 2 else num_classes
        model = MLP(input_size=input_size, hidden_size=args.hidden_size, output_size=output_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set up the loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        n_epochs = args.epochs

        # Train the model for this fold
        model, train_losses, val_losses = train_mlp(
            model, train_loader, val_loader, optimizer, device, n_epochs, num_classes
        )

        # Store training and validation losses for this fold
        train_losses_total.append(train_losses)
        val_losses_total.append(val_losses)

        # Evaluate on the validation set for this fold
        model.eval()
        predictions = predict_mlp(model, val_loader, device, num_classes).cpu().numpy()
        predictions = predictions.round().astype(int)
        all_predictions.append(predictions)

        # Save fold predictions to CSV
        fold_preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': y_val_fold})
        fold_preds.to_csv(f"{metrics_path}/{predictions_name}_fold{fold + 1}.csv", index=False)

        # Calculate and store metrics for this fold
        fold_metrics = compute_metrics(predictions, y_val_fold, num_classes)
        logger.info(fold_metrics)
        all_metrics.append(fold_metrics)

        # Suponiendo que quieres optimizar el F1-score (ajusta según tu criterio)
        val_metric = fold_metrics.get('f1', 0)  # O usa otra métrica relevante
        logger.info(val_metric)
        logger.info(f"Fold {fold + 1} - F1 Score: {val_metric}")

        logger.info(model_path)

        logger.info(f"val_metric: {val_metric} and best_val_metric: {best_val_metric}")
        # GUARDAR EL MEJOR MODELO
        if val_metric > best_val_metric:  # Cambia a "<" si minimizas la pérdida
            best_val_metric = val_metric
            best_model_path = os.path.join(model_path, "best_model_mlp.pth")
            torch.save(model.state_dict(), best_model_path )
            logger.info(f"Nuevo mejor modelo guardado para Fold {fold + 1} con F1 Score: {best_val_metric}")




            

        
    # Average validation metrics across folds
    final_metrics = {
        key: {"mean": np.mean([m[key] for m in all_metrics]), "std": np.std([m[key] for m in all_metrics])}
        for key in all_metrics[0].keys()
    }

    # Save final averaged validation metrics to JSON
    with open(f"{metrics_path}/{metrics_name}_kfold.json", 'w') as json_file:
        json.dump(final_metrics, json_file, indent=4)

    logger.info(f"Saving final averaged validation metrics to: {metrics_path}/{metrics_name}_kfold.json")
    logger.info("K-Fold Cross-Validation Completed")

    # Plot training and validation loss graphs
    plot_loss(train_losses_total, loss_graph_path, suffix_name, "Train")
    plot_loss(val_losses_total, loss_graph_path, suffix_name, "Validation")

    # Evaluation on the test set using the model from the last fold
    logger.info("Evaluating on final test set")
    test_dataset = MLPDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Cargar el mejor modelo antes de la evaluación final
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()
    logger.info("Performing predictions using MLP model")
    predictions = predict_mlp(model, test_loader, device, num_classes)
    predictions = predictions.cpu().numpy()
    predictions = predictions.round().astype(int)

    # Save test set predictions to CSV
    if num_classes == 2:
        predictions = predictions.flatten()  # Asegura que predictions sea 1D
        y_test = np.array(y_test).flatten()  # Asegura que y_test sea 1D
    
    preds = pd.DataFrame({'y_pred': predictions, 'y_true': y_test})
    preds.to_csv(f"{metrics_path}/{predictions_name}_test.csv", index=False)

    logger.info("Computing classification metrics")
    metrics = compute_metrics(predictions, y_test, num_classes)
    metrics['confusion_matrix'] = {
        f"Actual_{i}-Predicted_{j}": int(metrics['confusion_matrix'].iloc[i, j])
        for i in range(metrics['confusion_matrix'].shape[0])
        for j in range(metrics['confusion_matrix'].shape[1])
    }
    with open(f'{metrics_path}/{metrics_name}_test.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    logger.info(f"Saving predictions to: {metrics_path}/{predictions_name}_test.csv")
    logger.info(f"Saving metrics to: {metrics_path}/{metrics_name}_test.json")
