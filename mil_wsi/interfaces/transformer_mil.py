"""Code implementation of transformer MIL"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention  # Make sure to install this package!

from loguru import logger


class TransLayer(nn.Module):
    """
    Transformer-based attention layer using NystromAttention for computational efficiency.

    This layer applies Layer Normalization followed by Nystrom-based self-attention. 
    The output is residual-connected to the input. If the model is in evaluation mode,
    the attention weights are also returned.

    Args:
        norm_layer (nn.Module, optional): Normalization layer to use. Defaults to nn.LayerNorm.
        dim (int, optional): Feature dimension. Defaults to 512.
        n_heads (int, optional): Number of attention heads. Defaults to 8.

    Methods:
        forward(x): Applies attention mechanism and returns updated feature representation.

    """
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, n_heads=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim // 8,
            heads = n_heads,
            num_landmarks = dim // 2,    # number of landmarks
            pinv_iterations = 6,         # number of Moore-Penrose iterations for approximating pinverse
            residual = True,             # extra residual connection on the value
            dropout = 0.1
        )

    def forward(self, x):
        """
        Forward pass through the Transformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Updated feature representation with residual connection.
            torch.Tensor (optional): Attention weights if the model is in evaluation mode.
        """
        attn_result = None
        if not self.training:
            output, attn_result = self.attn(self.norm(x), return_attn=True)  # Extract attention weights. Only return attention weights if not training.
        else:
            output = self.attn(self.norm(x))  # Extract attention weights. Only return attention weights if not training.

        x = x + output
        return x, attn_result
    
class PPEG(nn.Module):
    """
    Positional Prior Encoding Generator (PPEG).

    This module enhances feature representations by introducing local positional 
    priors using depth-wise convolutional layers with different kernel sizes. 

    The input feature tokens are reshaped into a 2D spatial grid and processed 
    through three different convolutional layers (7x7, 5x5, 3x3). The resulting 
    feature maps are summed with the original features to incorporate positional 
    information before being flattened back into sequence format.

    Args:
        dim (int): Feature dimension of the input tensor.

    Methods:
        forward(x, H, W): Applies convolution-based positional encoding.

    """

    def __init__(self, dim):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3//2, groups=dim)

    def forward(self, x, H, W):
        """
        Forward pass of the PPEG module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            H (int): Height of the feature map.
            W (int): Width of the feature map.

        Returns:
            torch.Tensor: Output tensor with positional priors applied, 
                          shape (batch_size, seq_len, dim).
        """
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        # Reshape features into a 2D grid
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # Prepend the class token
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning (MIL) model.

    TransMIL processes a set of feature embeddings using a transformer architecture, 
    incorporating positional encoding and multi-head self-attention layers to learn 
    instance-level relationships for classification tasks.

    Args:
        n_classes (int): Number of output classes.
        n_heads (int): Number of attention heads in the transformer layers.
        in_dim (int): Input feature dimension.

    Attributes:
        pos_layer (PPEG): Positional encoding module using convolutional layers.
        _fc1 (nn.Sequential): Initial feature transformation layer (FC + ReLU).
        cls_token (nn.Parameter): Learnable class token for global representation.
        layer1 (TransLayer): First transformer layer.
        layer2 (TransLayer): Second transformer layer.
        norm (nn.LayerNorm): Layer normalization before classification.
        _fc2 (nn.Linear): Final classification layer.

    Methods:
        forward(**kwargs): Performs forward pass and returns classification results 
                           along with optional attention weights.
    """
    def __init__(self, n_classes, n_heads, in_dim):
        super(TransMIL, self).__init__()
        self.in_dim = in_dim
        self.pos_layer = PPEG(dim=in_dim)
        # Now use `in_dim` instead of hardcoding 1024.
        self._fc1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=in_dim, n_heads=n_heads)
        self.layer2 = TransLayer(dim=in_dim, n_heads=n_heads)
        self.norm = nn.LayerNorm(in_dim)
        self._fc2 = nn.Linear(in_dim, self.n_classes)

    def forward(self, **kwargs):
        """
        Forward pass of the TransMIL model.

        Args:
            kwargs['data'] (torch.Tensor): Input tensor of shape (batch_size, num_instances, in_dim).

        Returns:
            dict: A dictionary containing:
                - 'logits' (torch.Tensor): Raw model predictions of shape (batch_size, n_classes).
                - 'Y_prob' (torch.Tensor): Softmax probabilities of shape (batch_size, n_classes).
                - 'Y_hat' (torch.Tensor): Predicted class labels of shape (batch_size,).
                - 'attn_weights' (torch.Tensor or None): Attention weights from the second transformer 
                  layer (if in evaluation mode) of shape (batch_size, heads, H, H).
        """
        # Expect kwargs['data'] to have shape [B, n, in_dim]
        h = kwargs['data'].float()  # [B, n, in_dim]
        h = self._fc1(h)            # [B, n, 512]

        # Pad so that n_patches can form a 2D grid.
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)  # [B, 1+N, 512]

        h, attn_weights1 = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h, attn_weights2 = self.layer2(h)

        # Use the class token for prediction.
        h = self.norm(h)[:, 0]
        logits = self._fc2(h)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}

        results_dict['attn_weights'] = None
        if not self.training:  # Only store attention weights during inference
            # Slice attention weights to exclude class token and padding
            attn_weights2 = attn_weights2[:, :, 1:H+1, 1:H+1]  # Shape: [B, heads, H, H]
            results_dict['attn_weights'] = attn_weights2  # Return only the second layer's attention weights, since they are generally more high-level and expressive

        return results_dict

class TransformerMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning (MIL) model.

    This model leverages the `TransMIL` architecture for MIL classification tasks.
    It processes feature embeddings using a transformer-based self-attention mechanism 
    and extracts instance-level relationships for bag-level classification.

    Args:
        input_size (int): Dimensionality of the input feature vectors.
        n_heads (int): Number of attention heads in the transformer layers.
        output_size (int): Number of output classes.

    Attributes:
        attention_mil (TransMIL): Transformer-based MIL model for feature processing.
        classifier (nn.Identity): Placeholder for classifier (not explicitly used here).

    Methods:
        forward(x): Performs forward pass and returns predictions along with attention weights.

    """
    def __init__(self, input_size, n_heads, output_size):
        super().__init__()  # Ensure the parent class is initialized properly

        # Define submodules after calling super()
        self.attention_mil = TransMIL(n_classes=num_classes, n_heads=n_heads,in_dim=input_size)
        self.classifier = nn.Identity()

    def forward(self, x):
        """
        Forward pass of TransformerMIL.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_instances, input_size).

        Returns:
            torch.Tensor: Predicted logits of shape (batch_size, output_size).
            torch.Tensor (optional): Attention weights from the second transformer layer 
                                     (if in evaluation mode), shape (batch_size, heads, H, H).
        """
        output_dict = self.attention_mil(data=x)
        logits = output_dict['logits']

        if not self.training:
            attn_weights = output_dict['attn_weights']  # Extract attention weights
        else:
            attn_weights = None
        return logits, attn_weights  # Return predictions and attention weights from the second layer. We could also return both layers' weights and average them for example.


def train_transformer_model(model, train_loader, criterion, optimizer, device, epochs):
    """
    Trains a transformer-based Multiple Instance Learning (MIL) model.

    This function trains the model using binary cross-entropy loss for 
    binary classification tasks, optimizing using the provided optimizer.

    Args:
        model (torch.nn.Module): Transformer-based MIL model.
        train_loader (torch.utils.data.DataLoader): DataLoader containing training data.
        criterion (torch.nn.Module): Loss function (e.g., `nn.BCEWithLogitsLoss` for binary classification).
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        device (torch.device): Device to run training on (e.g., "cuda" or "cpu").
        epochs (int): Number of training epochs.

    Returns:
        torch.nn.Module: Trained model.
        torch.Tensor (optional): Attention weights from the last batch.
    """
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for bags, labels, _ in train_loader:
            bags, labels = bags.to(device), labels.to(device).float()  # Convert labels to float

            optimizer.zero_grad()
            outputs, attn_weights = model(bags)  # Extract predictions & (if available) attention scores
            outputs = outputs.squeeze(1)  # Ensure shape matches labels
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model, attn_weights


def predict_transformer_model(model, test_loader, device):
    """
    Performs inference using a trained transformer-based Multiple Instance Learning (MIL) model.

    This function evaluates the model on a test dataset, obtaining predictions, 
    attention weights (if available), and associated bag IDs.

    Args:
        model (torch.nn.Module): Trained Transformer-based MIL model.
        test_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run inference on (e.g., "cuda" or "cpu").

    Returns:
        torch.Tensor: Predictions for each bag (binary classification, values 0 or 1).
        list (of numpy arrays or None): Attention weights for each bag if available, otherwise None.
        list (of str): Bag IDs corresponding to each sample.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_attn_weights = []
    all_bag_ids = []

    with torch.no_grad():
        for i, (bags, _, basename) in enumerate(test_loader):
            bags = bags.to(device)

            outputs, attn_weights = model(bags)
            outputs = outputs.squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_attn_weights.append(attn_weights if attn_weights is None else attn_weights.cpu().numpy())
            all_bag_ids.append(basename[0])

    return torch.tensor(all_preds, dtype=torch.float32), all_attn_weights, all_bag_ids
