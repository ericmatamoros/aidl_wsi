import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerMILMLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads=8, output_size=1):
        super(TransformerMILMLP, self).__init__()

        # Ensure input size is compatible
        assert input_size % n_heads == 0, f"input_size ({input_size}) must be divisible by num_heads ({n_heads})"

        # Transformer-based MIL attention
        self.attention_mil = TransformerMIL(input_size, hidden_size, num_heads=n_heads)  # Ensure input_size matches d_model

        # Classifier (Ensure correct input to MLP)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Correct: Ensure correct input to MLP
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # Ensure output is (batch_size, 1)
        )


    def forward(self, x):
        bag_representation, attn_weights = self.attention_mil(x)  # Shape: (batch_size, input_size)
        output = self.classifier(bag_representation)  # Shape: (batch_size, 1)

        output = output.squeeze(1) if output.dim() == 2 else output

        return output, attn_weights  # Ensure output is (batch_size,)



class TransformerMIL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=8, dropout=0.1):
        super(TransformerMIL, self).__init__()

        # Ensure `input_size` is divisible by `num_heads`
        assert input_size % num_heads == 0, f"input_size ({input_size}) must be divisible by num_heads ({num_heads})"

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,  # Ensure we use input_size as `d_model`
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,  # Standard FFN expansion
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attention = nn.Linear(input_size, 1)

    def forward(self, x):
        batch_size, N_instances, _ = x.shape

        # Apply transformer
        transformed_x = self.transformer(x)  # Ensure x has `input_size` dimensions

        # Compute attention scores
        attn_logits = self.attention(transformed_x).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)

        # Weighted sum of instances
        bag_representation = torch.sum(attn_weights.unsqueeze(-1) * transformed_x, dim=1)

        return bag_representation, attn_weights


# Training function adapted for TransformerMIL
def train_transformer_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for bags, labels, _ in train_loader:
            bags, labels = bags.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs, attn_weights = model(bags)
            #outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model, attn_weights

# Prediction function adapted for TransformerMIL
def predict_transformer_model(model, test_loader, device):
    model.eval()
    model.to(device)

    all_preds = []
    all_attn_weights = []
    all_bag_ids = []

    with torch.no_grad():
        for i, (bags, _, basename) in enumerate(test_loader):
            bags = bags.to(device)

            outputs, attn_weights = model(bags)
            #outputs = outputs.squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_attn_weights.append(attn_weights.cpu().numpy())
            all_bag_ids.append(basename[0])

    return torch.tensor(all_preds, dtype=torch.float32), all_attn_weights, all_bag_ids
