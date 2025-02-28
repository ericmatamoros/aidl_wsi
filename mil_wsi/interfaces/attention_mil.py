import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(AttentionMIL, self).__init__()
        
        # Linear layers to replace matmul operations
        self.V = nn.Linear(input_size, hidden_size, bias=False)  # Replaces V * h_i^T
        self.U = nn.Linear(input_size, hidden_size, bias=False)  # Replaces U * h_i^T
        self.w = nn.Linear(hidden_size, 1, bias=False)  # Replaces w^T * (...)

        # Definicion MLP
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Capa oculta a capa de salida
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Función de activación relu
        self.relu = nn.ReLU()
        # funcion de activacion sigmoid
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, x):
        batch_size, N_instances, _ = x.shape  # Shape: (batch_size, N, M)
        
        # Apply linear transformations
        Vh = torch.tanh(self.V(x))  # Shape: (batch_size, N, hidden_size)
        Uh = torch.sigmoid(self.U(x))  # Shape: (batch_size, N, hidden_size)
        
        # Element-wise multiplication (Gated Mechanism)
        gated_output = Vh * Uh  # Shape: (batch_size, N, hidden_size)
        
        # Compute attention weights
        attn_logits = self.w(gated_output).squeeze(-1)  # Shape: (batch_size, N)
        attn_weights = torch.softmax(attn_logits, dim=1)  # Normalize
        
        # Compute weighted sum
        bag_representation = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # Shape: (batch_size, input_size)

        # Propagación hacia adelante
        bag_representation = self.relu(self.fc1(bag_representation))  # Aplicamos ReLU después de la primera capa
        bag_representation = self.fc2(bag_representation)  # Salida de la segunda capa
        
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

    return model, attn_weights

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
