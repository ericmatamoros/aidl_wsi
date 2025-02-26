from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.
        alpha: Balancing factor for positive and negative classes.
        gamma: Focusing parameter.
        reduction: 'mean' or 'sum' (default is 'mean').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: Model outputs (before sigmoid).
        targets: Ground-truth labels (0 or 1).
        """
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        probs = torch.clamp(probs, min=1e-6, max=1-1e-6)  # Prevent log(0) errors
        
        # Compute focal loss components
        focal_weight = self.alpha * (1 - probs) ** self.gamma * targets + (1 - self.alpha) * probs ** self.gamma * (1 - targets)
        loss = -focal_weight * (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        return loss.mean() if self.reduction == 'mean' else loss.sum()


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
            x = self.fc2(x)  # Salida de la segunda capa
            return x
    

def train_mlp(model, train_loader, val_loader, criterion, optimizer, device: torch.device, epochs: int, use_focal_loss=True):
    """
    Train an MLP model using either standard loss or Focal Loss.

    Arguments:
    model: PyTorch MLP model
    train_loader: DataLoader for training
    val_loader: DataLoader for validation
    criterion: Loss function (e.g., BCEWithLogitsLoss or FocalLoss)
    optimizer: PyTorch optimizer
    device: CUDA or CPU device
    epochs: Number of training epochs
    use_focal_loss: Whether to use Focal Loss for training
    
    Returns:
    Trained model
    """

    train_losses = []
    val_losses = []
    focal_loss = FocalLoss() if use_focal_loss else None

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Apply focal loss if specified
            if use_focal_loss:
                loss = focal_loss(outputs.squeeze(), labels.float())
            else:
                loss = criterion(outputs.squeeze(), labels.float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logger.info(
            f'Epoch [{epoch+1}/{epochs}], '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Val Loss: {avg_val_loss:.4f}'
        )

    return model, train_losses, val_losses


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