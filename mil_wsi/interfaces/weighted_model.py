import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class WeightedModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(WeightedModel, self).__init__()

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classification layer
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, x):
        batch_size, N_instances, _ = x.shape

        # Compute attention weights
        attn_weights = self.attention(x)  # Shape: (batch_size, N_instances, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize

        # Compute weighted bag representation
        bag_representation = torch.sum(attn_weights * x, dim=1)

        # Classification output
        output = self.classifier(bag_representation)

        return output, attn_weights  # Return both logits & attention weights



def train_weighted_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bags, labels, _ in train_loader:
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

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bags, labels, _ in val_loader:
                bags, labels = bags.to(device), labels.to(device).float()
                outputs, _ = model(bags)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Load the best model state at the end
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model based on validation loss.")

    return model, train_losses, val_losses



def predict_weighted_model(model, test_loader, device):
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
