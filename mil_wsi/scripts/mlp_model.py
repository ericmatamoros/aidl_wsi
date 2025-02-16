import os
import argparse
import json
import pandas as pd
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from mil_wsi.interfaces import (
     compute_metrics,
     train_mlp,
     predict_mlp,
     MLPDataset,
     MLP
)

import warnings
warnings.filterwarnings("ignore")
	

def load_data(input_path: str, files_pt: list, target: pd.DataFrame):

    df_list = []
    for file in files_pt:
        basename = file.split(".pt")[0]

        data = torch.load(f"{input_path}/pt_files/{file}")
        data = torch.mean(data, dim=0).numpy()
        df_dims = pd.DataFrame([data], columns=[i for i in range(len(data))])

        df_dims['filename'] = basename
        df_dims = df_dims.merge(target.drop(columns = 'slide'), on = ['filename'], how = "left")
        df_list.append(df_dims)

    
    df = pd.concat(df_list)

    return df


parser = argparse.ArgumentParser(description='mlp model')
parser.add_argument('--dir_results', type=str,
                    help='path to folder containing the results')
parser.add_argument('--dir_data', type=str,
                    help='path containing slides')
parser.add_argument('--dir_model', type=str,
                    help='path to store the trained models')
parser.add_argument('--dir_metrics', type=str,
                    help='path to store metrics')
parser.add_argument('--model_name', type=str, default='mlp_model',
                    help='name of the model')
parser.add_argument('--predictions_name', type=str, default='predictions',
                    help='name for predictions file')
parser.add_argument('--suffix_name', type=str,
                    help='name suffix for the experiment')
parser.add_argument('--metrics_name', type=str, default='metrics',
                    help='name for metrics file')
parser.add_argument('--batch_size', type=int, default=32,
                    help='size of the batch')
parser.add_argument('--hidden_size', type=float, default=128,
                    help='hidden size of the MLP network')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to train')
parser.add_argument('--test_size', type=float, default=0.3,
                    help='test size')
if __name__ == '__main__':
    args = parser.parse_args()

    # Define list of pt files
    input_path = args.dir_results
    data_path = args.dir_data
    model_path = args.dir_model
    metrics_path = args.dir_metrics
    

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    model_name = f"{args.model_name}{args.suffix_name}"
    predictions_name = f"{args.predictions_name}{args.suffix_name}"
    metrics_name = f"{args.metrics_name}{args.suffix_name}"

    files_pt = os.listdir(f"{input_path}/pt_files")

    logger.info("Reading data and generating data loaders")
    # Read target
    target = pd.read_csv(f"{data_path}/target.csv")
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)

    # Read data
    df = load_data(input_path, files_pt, target)

    # Features and target
    features = df.drop(columns=['filename', 'target'])
    input_size = features.shape[1]
    target = df['target']

    # Train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, target, test_size=args.test_size, random_state=4, stratify=target
    )

    # Balance the training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Then, split the temporary set into evaluation and test sets (keeping their natural imbalance)
    X_eval, X_test, y_eval, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=4, stratify=y_temp
    )

    # Create Dataset objects for training and testing
    train_dataset = MLPDataset(X_train, y_train)
    test_dataset = MLPDataset(X_test, y_test)
    val_dataset = MLPDataset(X_eval, y_eval)

    # Create DataLoader objects for batching
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    n_epochs = args.epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #MLP Loader    
    model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if os.path.exists(f"{model_path}{model_name}") is False:
        logger.info("Training MLP model")
        # Configuración de la función de pérdida y el optimizador
        criterion = nn.BCEWithLogitsLoss()  # Para clasificación binaria
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the MLP model
        model = train_mlp(model, train_loader, val_loader, criterion, optimizer, device, n_epochs)

        # Store model state
        torch.save(model.state_dict(), f"{model_path}{model_name}.pth")
    else:
        logger.info("Loading trained MLP model")
        model.load_state_dict(torch.load(f"{model_path}{model_name}.pth"))
    
    model.eval()

    logger.info("Performing predictions using MLP model")
    predictions = predict_mlp(model, test_loader, device)
    predictions = predictions.cpu().numpy()
    predictions = predictions.round().astype(int)

    preds = pd.DataFrame({'y_pred': predictions.ravel(), 'y_true': y_test})
    preds.to_csv(f"{metrics_path}{predictions_name}.csv", index= False)


    logger.info("Computing classification metrics")
    metrics = compute_metrics(predictions, y_test)

    metrics['confusion_matrix'] = metrics['confusion_matrix'].to_dict()
    with open(f'{metrics_path}{metrics_name}.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)




