import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(AttentionMIL, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classification layer
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, N_instances, _ = x.shape

        # Extract features
        features = self.feature_extractor(x)

        # Compute attention weights
        attn_weights = self.attention(features)  # Shape: (batch_size, N_instances, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize

        # Compute weighted bag representation
        bag_representation = torch.sum(attn_weights * features, dim=1)

        # Classification output
        output = self.classifier(bag_representation)

        return output, attn_weights  # Return both logits & attention weights



def train_attention_mil(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for bags, labels, _  in train_loader:
            bags, labels = bags.to(device), labels.to(device).float()  # Convert labels to float

            optimizer.zero_grad()
            outputs, attn_weights = model(bags)  # Extract both predictions & attention scores
            outputs = outputs.squeeze(1)  # Ensure shape matches labels
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model

def predict_attention_mil(model, test_loader, device):
    model.eval()
    model.to(device)

    all_preds = []
    all_attn_weights = []
    all_bag_ids = []

    with torch.no_grad():
        for i, (bags, _, basename) in enumerate(test_loader):
            bags = bags.to(device)

            # Extract predictions & attention scores
            outputs, attn_weights = model(bags)
            outputs = outputs.squeeze(1)

            # Convert to probabilities & binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Store predictions
            all_preds.extend(preds.cpu().numpy())

            # Store attention weights (converted to numpy)
            all_attn_weights.append(attn_weights.cpu().numpy())

            # Store bag IDs for explainability
            all_bag_ids.append(basename)

    return torch.tensor(all_preds, dtype=torch.float32), all_attn_weights, all_bag_ids
