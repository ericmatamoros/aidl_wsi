import torch
from einops import rearrange
from torch import nn

# --- Transformer building blocks (as provided) ---
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=None, dropout=0.1):
        super().__init__()
        if dim_head is None:
            dim_head = dim // heads
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x shape: (batch, n, dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=8, use_ff=True, use_norm=True):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim // heads)
        self.use_ff = use_ff
        self.use_norm = use_norm
        if self.use_ff:
            self.ff = FeedForward(dim=dim)
    def forward(self, x):
        if self.use_norm:
            x = x + self.attn(self.norm(x))
        else:
            x = x + self.attn(x)
        if self.use_ff:
            x = x + self.ff(x)
        return x

# --- New TransformerMIL module using the above transformer blocks ---
class TransformerMIL(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=2, heads=8, num_classes=1, dropout=0.1):
        """
        Args:
            input_size: Dimension of each patch's feature vector.
            hidden_size: Internal transformer dimension.
            num_layers: Number of TransformerLayer blocks.
            heads: Number of attention heads.
            num_classes: Number of output classes (1 for binary classification).
            dropout: Dropout probability.
        """
        super().__init__()
        # Project input features into the transformer embedding space.
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Stack transformer layers.
        self.layers = nn.ModuleList([
            TransformerLayer(dim=hidden_size, heads=heads, use_ff=False, use_norm=True)
            for _ in range(num_layers)
        ])
        # Final classification head.
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, _=None):
        """
        Args:
            x: Tensor of shape (batch, n_patches, input_size).
        Returns:
            logits: (batch, num_classes)
        """
        h = self.fc1(x)  # (batch, n_patches, hidden_size)
        for layer in self.layers:
            h = layer(h)
        # Aggregate patch representations via mean pooling.
        h = h.mean(dim=1)  # (batch, hidden_size)
        logits = self.fc2(h)  # (batch, num_classes)
        return logits

# --- MILModels that integrates different attention modules ---
# (We assume AttentionMIL and MultiHeadAttention are defined elsewhere.)
class MILModels(nn.Module):
    def __init__(self, input_size, hidden_size, attention_class, n_heads=8, output_size=1,
                 num_layers=2, dropout=0.1):
        super(MILModels, self).__init__()
        if attention_class == "AttentionMIL":
            self.attention_mil = AttentionMIL(input_size, hidden_size)
            classifier_input_size = input_size  # bag_representation from AttentionMIL.
        elif attention_class == "MultiHeadAttention":
            self.attention_mil = MultiHeadAttention(input_size, hidden_size, n_heads)
            classifier_input_size = input_size
        elif attention_class == "Transformer":
            # Here we use our new TransformerMIL module.
            self.attention_mil = TransformerMIL(input_size, hidden_size,
                                                 num_layers=num_layers, heads=n_heads,
                                                 num_classes=output_size, dropout=dropout)
            classifier_input_size = hidden_size  # Not used since TransformerMIL outputs logits.
        else:
            raise ValueError("Unsupported attention class. Choose from: 'AttentionMIL', 'MultiHeadAttention', 'Transformer'")
        
        # For non-transformer modules, we use a separate classifier.
        if attention_class != "Transformer":
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        else:
            self.classifier = nn.Identity()  # Already integrated in TransformerMIL.

    def forward(self, x):
        if isinstance(self.attention_mil, TransformerMIL):
            # TransformerMIL already produces logits.
            logits = self.attention_mil(x)
            return logits, None  # No explicit attention weights.
        else:
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
