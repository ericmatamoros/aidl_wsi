"""Code implementation of MLP model"""

import os
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._focal_loss import FocalLoss

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for classification or regression.

    This MLP consists of:
    - An input layer that connects to a hidden layer with ReLU activation.
    - A final output layer without an activation function (activation should be handled externally).

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output neurons (e.g., 1 for binary classification, 
                           or number of classes for multi-class classification).

    Attributes:
        fc1 (nn.Linear): Fully connected layer from input to hidden layer.
        fc2 (nn.Linear): Fully connected layer from hidden to output layer.
        relu (nn.ReLU): ReLU activation function.

    Methods:
        forward(x): Computes the forward pass.

    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def train_mlp(model, train_loader, val_loader, optimizer, device, epochs, num_classes=2, save_path="best_model.pth"):
    """
    Trains a Multi-Layer Perceptron (MLP) model for binary or multiclass classification.

    This function optimizes the model using a specified loss function, saves the best-performing 
    model based on validation loss, and returns the final trained model along with loss histories.

    Args:
        model (torch.nn.Module): The MLP model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        device (torch.device): Device for computation (e.g., "cuda" or "cpu").
        epochs (int): Number of training epochs.
        num_classes (int, optional): Number of output classes (2 for binary classification, 
                                     >2 for multiclass classification). Defaults to 2.
        save_path (str, optional): File path to save the best model. Defaults to "best_model.pth".

    Returns:
        torch.nn.Module: The best-trained model based on validation loss.
        list: Training loss history for each epoch.
        list: Validation loss history for each epoch.
    """
    # Choose loss function based on classification type
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #breakpoint()
            if num_classes == 2:
                #loss = criterion(outputs.view(-1), labels.float())  # Binary
                focal_loss = FocalLoss()
                loss = focal_loss(outputs.view(-1), labels.float())
            else:
                loss = criterion(outputs, labels.long())  # Multiclass

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                if num_classes == 2:
                    loss = criterion(outputs.view(-1), labels.float())  # Binary .squeeze()
                else:
                    loss = criterion(outputs, labels.long())  # Multiclass

                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved at epoch {best_epoch} with val loss {avg_val_loss:.4f}")

        logger.info(
            f'Epoch [{epoch+1}/{epochs}], '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Val Loss: {avg_val_loss:.4f}'
        )

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    logger.info(f"Loaded best model from epoch {best_epoch} with val loss {best_val_loss:.4f}")
    
    if os.path.exists(save_path):
        os.remove(save_path)
        logger.info(f"Removed checkpoint file: {save_path}")

    return model, train_losses, val_losses




def predict_mlp(model, test_loader, device: torch.device, num_classes: int, threshold=0.5):
    """
    Performs inference using a trained MLP model for binary or multiclass classification.

    Args:
        model (torch.nn.Module): Trained MLP model.
        test_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run inference on (e.g., "cuda" or "cpu").
        num_classes (int): Number of output classes (2 for binary classification, >2 for multiclass classification).
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        torch.Tensor: Predicted class labels for all test samples.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if num_classes == 2:
                probs = torch.sigmoid(outputs)  # Binary classification
                #threshold = probs.mean().item() 
                preds = (probs > threshold).float()
            else:
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)  # Multiclass classification
            
            all_preds.append(preds)

    return torch.cat(all_preds, dim=0)