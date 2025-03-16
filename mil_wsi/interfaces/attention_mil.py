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
    def __init__(self, input_size, hidden_size):
        super(AttentionMIL, self).__init__()
        self.V = nn.Linear(input_size, hidden_size, bias=False)  # Replaces V * h_i^T
        self.U = nn.Linear(input_size, hidden_size, bias=False)  # Replaces U * h_i^T
        self.w = nn.Linear(hidden_size, 1, bias=False)  # Replaces w^T * (...)
    
    def forward(self, x):
        batch_size, N_instances, _ = x.shape  # Shape: (batch_size, N, M)
        Vh = torch.tanh(self.V(x))  # Shape: (batch_size, N, hidden_size)
        Uh = torch.sigmoid(self.U(x))  # Shape: (batch_size, N, hidden_size)
        gated_output = Vh * Uh  # Shape: (batch_size, N, hidden_size)
        attn_logits = self.w(gated_output).squeeze(-1)  # Shape: (batch_size, N)
        attn_weights = torch.softmax(attn_logits, dim=1)  # Normalize
        bag_representation = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # Shape: (batch_size, input_size)
        return bag_representation, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            AttentionMIL(input_size, hidden_size) for _ in range(n_heads)
        ])
    
    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        attn_weights = torch.cat([output[1].unsqueeze(0) for output in head_outputs], dim=0)
        attn_weights = torch.mean(attn_weights, dim=0)
        bag_representation = head_outputs[0][0]
        return bag_representation, attn_weights

class AttentionMILMLP(nn.Module):
    def __init__(self, input_size, hidden_size, attention_class, n_heads, output_size):
        super(AttentionMILMLP, self).__init__()
        
        if attention_class == "AttentionMIL":
            self.attention_mil = AttentionMIL(input_size, hidden_size)  # Instancia de atención
        elif attention_class == "MultiHeadAttention":
            self.attention_mil = MultiHeadAttention(input_size, hidden_size, n_heads)  # Instancia de atención
        else:
            raise Exception("Not a valid attention mechanism")
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        bag_representation, attn_weights = self.attention_mil(x)
        output = self.classifier(bag_representation)  # Pasa por MLP externa
        return output, attn_weights

def train_attention_mil(model, train_loader, val_loader, optimizer, device, epochs, num_classes, save_path="best_model.pth"):
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