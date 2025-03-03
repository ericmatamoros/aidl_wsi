import torch
import torch.nn as nn

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