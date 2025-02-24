import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(MIL, self).__init__()
        
        # Feature extraction MLP
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification Layer
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch_size, N_instances, D_features)
        """
        batch_size, N_instances, _ = x.shape
        
        # Extract Features
        features = self.feature_extractor(x)  # Shape: (batch_size, N_instances, hidden_size)

        # Compute Attention Weights
        attn_weights = self.attention(features)  # Shape: (batch_size, N_instances, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize across instances

        # Aggregate Features Using Attention
        bag_representation = torch.sum(attn_weights * features, dim=1)  # Shape: (batch_size, hidden_size)

        # Classification
        output = self.classifier(bag_representation)  # Shape: (batch_size, 1)
        
        return output



def train_mil(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for bags, labels in train_loader:
            bags, labels = bags.to(device), labels.to(device).float()  # Convert labels to float

            optimizer.zero_grad()
            outputs = model(bags).squeeze(1)  # Ensure shape matches labels
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



def predict_mil(model, test_loader, device):
    model.eval()
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for bags, _ in test_loader:
            bags = bags.to(device)
            outputs = model(bags).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).float()  # Convert to float tensor
            all_preds.extend(preds.cpu().numpy())

    return torch.tensor(all_preds, dtype=torch.float32)  # Ensure correct dtype

