
from loguru import logger
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Capa de entrada a capa oculta
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Capa oculta a capa de salida
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Función de activación relu
        self.relu = nn.ReLU()
        # funcion de activacion sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
            # Propagación hacia adelante
            x = self.relu(self.fc1(x))  # Aplicamos ReLU después de la primera capa
            x = self.sigmoid(self.fc2(x))  # Salida de la segunda capa
            return x
    

def train_mlp(model, train_loader, val_loader, criterion, optimizer, device: torch.device, epochs: int, penalty_coefficient=0.1):
    for epoch in range(epochs):
        # Training phase
        model.train()  # set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Use the penalized loss during training
            loss = penalized_loss_binary(outputs, labels, criterion, penalty_coefficient=penalty_coefficient)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # Evaluation phase
        model.eval()  # set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # disable gradient computation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # During evaluation, you may choose to use the base loss only
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f'Epoch [{epoch+1}/{epochs}], '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Val Loss: {avg_val_loss:.4f}'
        )

    return model

def predict_mlp(model, test_loader, device: torch.device, threshold=0.5):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            # Use a custom threshold instead of just rounding
            preds = (probs > threshold).float()
            all_preds.append(preds)
    return torch.cat(all_preds, dim=0)


def penalized_loss_binary(outputs, targets, criterion, penalty_coefficient=0.8):
    """
    Computes the standard loss (e.g., BCEWithLogitsLoss) and adds a penalty
    term if the average predicted probability deviates from 0.5.
    
    Args:
        outputs: Raw model outputs (logits) of shape (batch_size,).
        targets: Ground truth labels of shape (batch_size,).
        criterion: The base loss function (e.g., BCEWithLogitsLoss).
        penalty_coefficient: How strongly to penalize imbalanced predictions.
        
    Returns:
        Combined loss value.
    """
    # Compute the primary loss (e.g., binary cross entropy)
    loss = criterion(outputs.squeeze(), targets.float())
    
    # Convert logits to probabilities
    probs = torch.sigmoid(outputs)
    # Calculate the average predicted probability for the positive class
    avg_prob = probs.mean()
    
    # Penalize if avg_prob deviates from 0.5
    penalty = penalty_coefficient * (avg_prob - 0.5)**2
    
    return loss + penalty