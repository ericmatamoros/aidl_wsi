"""Implementation of Attention & MultiHead-Attention MIL classes and functions"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class AttentionMIL(nn.Module):
    """
    Implements a gated attention-based Multiple Instance Learning (MIL) mechanism.

    This attention mechanism computes instance-level attention scores 
    and aggregates them to form a bag-level representation.

    Args:
        input_size (int): Dimension of the input feature vectors.
        hidden_size (int): Dimension of the hidden layer in the attention mechanism.

    Methods:
        forward(x): Computes attention scores for instances and 
                    aggregates them into a bag representation.

    """
    def __init__(self, input_size, hidden_size):
        super(AttentionMIL, self).__init__()
        self.V = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(input_size, hidden_size, bias=False) 
        self.w = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, x):
        """
        Forward pass through the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_instances, input_size).

        Returns:
            torch.Tensor: Bag representation after weighted aggregation, 
                          shape (batch_size, input_size).
            torch.Tensor: Attention weights for each instance, shape (batch_size, num_instances).
        """
        batch_size, N_instances, _ = x.shape
        Vh = torch.tanh(self.V(x))
        Uh = torch.sigmoid(self.U(x))
        gated_output = Vh * Uh
        attn_logits = self.w(gated_output).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        bag_representation = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        return bag_representation, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention for Multiple Instance Learning (MIL).

    This model applies multiple AttentionMIL heads in parallel and averages 
    their attention weights to form a bag-level representation.

    Args:
        input_size (int): Dimension of the input feature vectors.
        hidden_size (int): Dimension of the hidden layer in each attention head.
        n_heads (int): Number of attention heads.

    Methods:
        forward(x): Computes multiple attention scores and aggregates 
                    them into a final bag representation.

    """
    def __init__(self, input_size, hidden_size, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            AttentionMIL(input_size, hidden_size) for _ in range(n_heads)
        ])
    
    def forward(self, x):
        """
        Forward pass through multiple attention heads.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_instances, input_size).

        Returns:
            torch.Tensor: Bag representation after weighted aggregation, 
                          shape (batch_size, input_size).
            torch.Tensor: Averaged attention weights across heads, 
                          shape (batch_size, num_instances).
        """
        head_outputs = [head(x) for head in self.heads]
        attn_weights = torch.cat([output[1].unsqueeze(0) for output in head_outputs], dim=0)
        attn_weights = torch.mean(attn_weights, dim=0)
        bag_representation = head_outputs[0][0]
        return bag_representation, attn_weights

class AttentionMILMLP(nn.Module):
    """
    A Multiple Instance Learning (MIL) model with an attention mechanism 
    and a Multi-Layer Perceptron (MLP) classifier.

    This model applies an attention mechanism to obtain a bag-level 
    representation and then passes it through an MLP classifier to 
    generate predictions.

    Args:
        input_size (int): Dimension of the input feature vectors.
        hidden_size (int): Dimension of the hidden layer in both attention 
                           and classifier MLP.
        attention_class (str): The type of attention mechanism to use. 
                               Options: "AttentionMIL", "MultiHeadAttention".
        n_heads (int): Number of attention heads (used only if MultiHeadAttention is selected).
        output_size (int): Number of output classes.

    Raises:
        Exception: If an invalid attention mechanism is specified.

    Methods:
        forward(x): Passes the input through the attention mechanism 
                    and classifier, returning predictions and attention weights.

    """
    def __init__(self, input_size, hidden_size, attention_class, n_heads, output_size):
        super(AttentionMILMLP, self).__init__()
        
        if attention_class == "AttentionMIL":
            self.attention_mil = AttentionMIL(input_size, hidden_size)
        elif attention_class == "MultiHeadAttention":
            self.attention_mil = MultiHeadAttention(input_size, hidden_size, n_heads)
        else:
            raise Exception("Not a valid attention mechanism")
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass through the attention mechanism and MLP classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_instances, input_size).

        Returns:
            torch.Tensor: Model predictions for each bag.
            torch.Tensor: Attention weights assigned to instances within each bag.
        """
        bag_representation, attn_weights = self.attention_mil(x)
        output = self.classifier(bag_representation)
        return output, attn_weights

def train_attention_mil(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        epochs, 
        num_classes, 
        save_path="best_model.pth"):
    """
    Trains a multiple instance learning (MIL) model with attention, using either binary 
    cross-entropy or cross-entropy loss, and saves the best-performing model based on 
    validation loss.

    Args:
        model (torch.nn.Module): The MIL model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        device (torch.device): Device for computation (e.g., "cuda" or "cpu").
        epochs (int): Number of training epochs.
        num_classes (int): Number of output classes (binary classification if num_classes == 2).
        save_path (str, optional): File path to save the best model. Defaults to "best_model.pth".

    Returns:
        torch.nn.Module: The best-trained model based on validation loss.
        list: Training losses for each epoch.
        list: Validation losses for each epoch.
    """
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

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            
            if num_classes == 2:
                loss = criterion(outputs.view(-1), labels.float())
            else:
                loss = criterion(outputs, labels.long())
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                
                if num_classes == 2:
                    loss = criterion(outputs.view(-1), labels.float())
                else:
                    loss = criterion(outputs, labels.long())
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
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

    model.load_state_dict(torch.load(save_path, map_location=device))
    logger.info(f"Loaded best model from epoch {best_epoch} with val loss {best_val_loss:.4f}")
    if os.path.exists(save_path):
        os.remove(save_path)
        logger.info(f"Removed checkpoint file: {save_path}")

    return model, train_losses, val_losses


def predict_attention_mil(model, test_loader, device,  num_classes: int, threshold=0.5):
    """
    Performs inference on a multiple instance learning (MIL) model with attention, 
    extracting predictions, attention weights, and bag IDs.

    Args:
        model (torch.nn.Module): The trained MIL model.
        test_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run the inference on (e.g., "cuda" or "cpu").
        num_classes (int): Number of output classes (binary classification if num_classes == 2).
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        torch.Tensor: Predictions for each bag.
        list: Attention weights for each bag (as NumPy arrays).
        list: Bag IDs corresponding to each sample.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_attn_weights = []
    all_bag_ids = []

    with torch.no_grad():
        for i, (inputs, _, basename) in enumerate(test_loader):
            inputs = inputs.to(device)

            # Extract predictions & attention scores
            outputs, attn_weights = model(inputs)
            outputs = outputs.squeeze(1)

            if num_classes == 2:
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
            else:
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            # Store predictions
            all_preds.extend(preds)

            # Store attention weights (converted to numpy)
            all_attn_weights.append(attn_weights.cpu().numpy())

            # Store bag IDs for explainability
            all_bag_ids.append(basename[0])

    return torch.tensor(all_preds, dtype=torch.float32), all_attn_weights, all_bag_ids