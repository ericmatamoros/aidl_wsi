"""Implementation of the focal loss function."""
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Implements the Focal Loss for binary classification.

        Focal Loss is designed to address class imbalance by down-weighting 
        easy examples and focusing on hard examples. It modifies the standard 
        binary cross-entropy loss by adding a modulating factor (1 - p_t)^gamma 
        to the loss, where p_t is the probability of the correct class.

        Args:
            alpha (float, optional): Balancing factor for positive and negative classes (default: 0.25).
            gamma (float, optional): Focusing parameter that adjusts the weight given to hard samples (default: 2.0).
            reduction (str, optional): Specifies the reduction mode ('mean' or 'sum'). Defaults to 'mean'.

        Methods:
            forward(logits, targets): Computes the focal loss.

        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Computes the focal loss.

        Args:
            logits (torch.Tensor): Model outputs (before applying sigmoid), shape (batch_size,).
            targets (torch.Tensor): Ground-truth binary labels (0 or 1), shape (batch_size,).

        Returns:
            torch.Tensor: Computed focal loss value.

        """
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        probs = torch.clamp(probs, min=1e-6, max=1-1e-6)  # Prevent log(0) errors
        
        # Compute focal loss components
        focal_weight = self.alpha * (1 - probs) ** self.gamma * targets + (1 - self.alpha) * probs ** self.gamma * (1 - targets)
        loss = -focal_weight * (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        return loss.mean() if self.reduction == 'mean' else loss.sum()