from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-6, max=1-1e-6)
        focal_weight = self.alpha * (1 - probs) ** self.gamma * targets + (1 - self.alpha) * probs ** self.gamma * (1 - targets)
        loss = -focal_weight * (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class MILTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim):
        super(MILTransformer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.encoder(x)  # Apply transformer layers
        x = x.mean(dim=1)  # Aggregate instances in the bag (mean pooling)
        x = self.fc(x)  # Final classification layer
        return x


def train_mil(model, train_loader, val_loader, criterion, optimizer, device, epochs, use_focal_loss=True):
    train_losses, val_losses = [], []
    focal_loss = FocalLoss() if use_focal_loss else None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = focal_loss(outputs.squeeze(), labels.float()) if use_focal_loss else criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
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
        
        logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return model, train_losses, val_losses

def predict_mil(model, test_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            all_preds.append(preds)
    return torch.cat(all_preds, dim=0)
