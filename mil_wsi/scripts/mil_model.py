import os
import argparse
import json
import pandas as pd
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from mil_wsi.interfaces import (
    compute_metrics,
    train_mil,
    predict_mil,
    MILDataset,
    MILTransformer
)

import warnings
warnings.filterwarnings("ignore")

def load_mil_data(input_path: str, files_pt: list, target: pd.DataFrame):
    df_list = []
    for file in files_pt:
        basename = file.split(".pt")[0]
        
        data = torch.load(f"{input_path}/pt_files/{file}")
        df_dims = pd.DataFrame([basename], columns=['filename'])
        df_dims['data'] = [data]  # Store full feature set for MIL processing
        df_dims = df_dims.merge(target.drop(columns='slide'), on=['filename'], how="left")
        df_list.append(df_dims)
    
    return pd.concat(df_list)

parser = argparse.ArgumentParser(description='MIL Transformer Model')
parser.add_argument('--dir_results', type=str, help='path to results folder')
parser.add_argument('--dir_data', type=str, help='path containing slides')
parser.add_argument('--dir_model', type=str, help='path to store trained models')
parser.add_argument('--dir_metrics', type=str, help='path to store metrics')
parser.add_argument('--model_name', type=str, default='mil_transformer', help='name of the model')
parser.add_argument('--predictions_name', type=str, default='predictions', help='name for predictions file')
parser.add_argument('--suffix_name', type=str, help='experiment suffix')
parser.add_argument('--metrics_name', type=str, default='metrics', help='name for metrics file')
parser.add_argument('--batch_size', type=int, default=1, help='MIL requires batch size 1 (one bag per batch)')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size for MIL Transformer')
parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--test_size', type=float, default=0.3, help='test dataset proportion')

if __name__ == '__main__':
    args = parser.parse_args()
    
    input_path = args.dir_results
    data_path = args.dir_data
    model_path = args.dir_model
    metrics_path = args.dir_metrics
    
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    
    model_name = f"{args.model_name}{args.suffix_name}"
    predictions_name = f"{args.predictions_name}{args.suffix_name}"
    metrics_name = f"{args.metrics_name}{args.suffix_name}"
    
    files_pt = os.listdir(f"{input_path}/pt_files")
    logger.info("Reading MIL data and generating dataloaders")
    
    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)
    
    df = load_mil_data(input_path, files_pt, target)
    
    # Train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        df[['filename', 'data']], df['target'], test_size=args.test_size, random_state=4, stratify=df['target']
    )
    X_eval, X_test, y_eval, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=4, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = MILDataset(X_train, y_train)
    test_dataset = MILDataset(X_test, y_test)
    val_dataset = MILDataset(X_eval, y_eval)
    
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize MIL Transformer model
    input_size = train_dataset.feature_dim
    model = MILTransformer(input_size=input_size, hidden_size=args.hidden_size, output_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if not os.path.exists(f"{model_path}{model_name}.pth"):
        logger.info("Training MIL Transformer model")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model = train_mil(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)
        
        torch.save(model.state_dict(), f"{model_path}{model_name}.pth")
    else:
        logger.info("Loading trained MIL Transformer model")
        model.load_state_dict(torch.load(f"{model_path}{model_name}.pth"))
    
    model.eval()
    
    logger.info("Performing predictions using MIL Transformer")
    predictions = predict_mil(model, test_loader, device)
    predictions = predictions.cpu().numpy().round().astype(int)
    
    preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': y_test})
    preds.to_csv(f"{metrics_path}{predictions_name}.csv", index=False)
    
    logger.info("Computing classification metrics")
    metrics = compute_metrics(predictions, y_test)
    
    metrics['confusion_matrix'] = metrics['confusion_matrix'].to_dict()
    with open(f'{metrics_path}{metrics_name}.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)
