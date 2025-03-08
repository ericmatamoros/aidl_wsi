import torch
import torch.nn as nn
import torch.nn.functional as F

# Original AttentionMIL module
class AttentionMIL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(AttentionMIL, self).__init__()
        
        # Linear layers to replace matmul operations
        self.V = nn.Linear(input_size, hidden_size, bias=False)  # Replaces V * h_i^T
        self.U = nn.Linear(input_size, hidden_size, bias=False)  # Replaces U * h_i^T
        self.w = nn.Linear(hidden_size, 1, bias=False)  # Replaces w^T * (...)

    def forward(self, x):
        # x shape: (batch_size, N_instances, input_size)
        Vh = torch.tanh(self.V(x))  # (batch_size, N, hidden_size)
        Uh = torch.sigmoid(self.U(x))  # (batch_size, N, hidden_size)
        gated_output = Vh * Uh  # Element-wise multiplication
        
        attn_logits = self.w(gated_output).squeeze(-1)  # (batch_size, N)
        attn_weights = torch.softmax(attn_logits, dim=1)
        bag_representation = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # (batch_size, input_size)
        
        return bag_representation, attn_weights

# Original MultiHeadAttention using AttentionMIL modules
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            AttentionMIL(input_size, hidden_size) for _ in range(n_heads)
        ])
    
    def forward(self, x):
        # Collect outputs from each head
        head_outputs = [head(x) for head in self.heads]
        # Concatenate attention weights from each head and average them
        attn_weights = torch.cat([output[1].unsqueeze(0) for output in head_outputs], dim=0)
        attn_weights = torch.mean(attn_weights, dim=0)
        # Here we take the bag representation from the first head (alternatively, you can aggregate differently)
        bag_representation = head_outputs[0][0]
        return bag_representation, attn_weights

# New Transformer-based MIL module
class TransformerMIL(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, num_layers=1, dropout=0.1):
        super(TransformerMIL, self).__init__()
        # Project input features to the transformer model dimension (hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        # Create a transformer encoder layer and stack num_layers of them
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Instead of attention weights, we pool over the sequence dimension (instances)
    
    def forward(self, x):
        # x: (batch, N, input_size)
        x_proj = self.input_proj(x)  # (batch, N, hidden_size)
        # Transformer expects (sequence length, batch, hidden_size)
        x_proj = x_proj.transpose(0, 1)  # (N, batch, hidden_size)
        x_encoded = self.transformer_encoder(x_proj)  # (N, batch, hidden_size)
        x_encoded = x_encoded.transpose(0, 1)  # (batch, N, hidden_size)
        # Simple pooling over instances (e.g., mean pooling)
        bag_representation = x_encoded.mean(dim=1)  # (batch, hidden_size)
        # For the transformer version, we are not directly returning attention weights
        attn_weights = None
        return bag_representation, attn_weights

# Main model class: choose from different attention mechanisms (or transformer)
class MILModels(nn.Module):
    def __init__(self, input_size, hidden_size, attention_class, n_heads=8, output_size=1):
        super(MILModels, self).__init__()
        
        # Depending on the attention_class, instantiate the corresponding module.
        if attention_class == "AttentionMIL":
            self.attention_mil = AttentionMIL(input_size, hidden_size)
            classifier_input_size = input_size  # bag_representation is computed as weighted sum of x
        elif attention_class == "MultiHeadAttention":
            self.attention_mil = MultiHeadAttention(input_size, hidden_size, n_heads)
            classifier_input_size = input_size
        elif attention_class == "Transformer":
            self.attention_mil = TransformerMIL(input_size, hidden_size, n_heads)
            classifier_input_size = hidden_size  # transformer returns representation of dimension hidden_size
        else:
            raise ValueError("Unsupported attention class. Choose from: 'AttentionMIL', 'MultiHeadAttention', 'Transformer'")
        
        # Classifier network that maps the bag representation to the final output
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        bag_representation, attn_weights = self.attention_mil(x)
        output = self.classifier(bag_representation)
        return output, attn_weights

# Training function (unchanged)
def train_transformer_model(model, train_loader, criterion, optimizer, device, epochs):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model, attn_weights

# Prediction function (unchanged)
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
            outputs = outputs.squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_attn_weights.append(attn_weights if attn_weights is None else attn_weights.cpu().numpy())
            all_bag_ids.append(basename[0])

    return torch.tensor(all_preds, dtype=torch.float32), all_attn_weights, all_bag_ids
