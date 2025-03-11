import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from nystrom_attention import NystromAttention  # Make sure to install this package!

from loguru import logger


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, n_heads=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim // 8,
            heads = n_heads,
            num_landmarks = dim // 2,    # number of landmarks
            pinv_iterations = 6,         # number of Moore-Penrose iterations for approximating pinverse
            residual = True,             # extra residual connection on the value
            dropout = 0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
    
    # def forward(self, x):
    #     attn_result = self.attn(self.norm(x), return_attn=not self.training)  # Extract attention weights. Only return attention weights if not training.

    #     if self.training:
    #         attn_output = attn_result
    #         attn_weights = None
    #     else:
    #         attn_output, attn_weights = attn_result

    #     return x, attn_weights if not self.training else x  # Return both transformed features and attention weights


class PPEG(nn.Module):
    def __init__(self, dim):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        # Reshape features into a 2D grid
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # Prepend the class token
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes, n_heads, in_dim):
        super(TransMIL, self).__init__()
        self.in_dim = in_dim
        self.pos_layer = PPEG(dim=in_dim)
        # Now use `in_dim` instead of hardcoding 1024.
        self._fc1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=in_dim, n_heads=n_heads)
        self.layer2 = TransLayer(dim=in_dim, n_heads=n_heads)
        self.norm = nn.LayerNorm(in_dim)
        self._fc2 = nn.Linear(in_dim, self.n_classes)

    def forward(self, **kwargs):
        # Expect kwargs['data'] to have shape [B, n, in_dim]
        h = kwargs['data'].float()  # [B, n, in_dim]
        h = self._fc1(h)            # [B, n, 512]

        # Pad so that n_patches can form a 2D grid.
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)  # [B, 1+N, 512]

        # h = self.layer1(h)
        # h = self.pos_layer(h, _H, _W)
        # h = self.layer2(h)
        # Collect attention weights from both transformer layers
        h, attn_weights1 = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h, attn_weights2 = self.layer2(h)

        # Use the class token for prediction.
        h = self.norm(h)[:, 0]
        logits = self._fc2(h)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        results_dict = {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat
        }
        if not self.training:  # Only store attention weights during inference
            #results_dict['attn_weights'] = (attn_weights1, attn_weights2)  # Return both attention layers' weights
            #results_dict['attn_weights'] = attn_weights2  # Return only the second layer's attention weights
            results_dict['attn_weights'] = None  # Return only the second layer's attention weights
        return results_dict

class TransformerMIL(nn.Module):
    def __init__(self, input_size, n_heads, output_size):
        super().__init__()  # Ensure the parent class is initialized properly

        # Define submodules after calling super()
        self.attention_mil = TransMIL(n_classes=output_size, n_heads=n_heads,in_dim=input_size)
        self.classifier = nn.Identity()

    # def forward(self, x):
    #     logits = self.attention_mil(data=x)['logits']
    #     return logits, None
    def forward(self, x):
        output_dict = self.attention_mil(data=x)
        logits = output_dict['logits']

        if not self.training:
            attn_weights = output_dict['attn_weights']  # Extract attention weights
        else:
            attn_weights = None
        #return logits, attn_weights  # Return predictions and attention weights from the second layer. We could also return both layers' weights and average them for example.
        return logits, None  # Return predictions and attention weights from the second layer. We could also return both layers' weights and average them for example.


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
            #with torch.autocast(device_type="cuda"):
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
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model, None

# Prediction function (unchanged)
def predict_transformer_model(model, test_loader, device, test):
    model.eval()
    model.to(device)

    all_preds = []
    all_attn_weights = []
    all_bag_ids = []
    save_path = "./mil_wsi/results/attn_weights"


    with torch.no_grad():
        for i, (bags, _, basename) in enumerate(test_loader):
            bags = bags.to(device)

            #with torch.autocast(device_type="cuda"):
            outputs, attn_weights = model(bags)
            outputs = outputs.squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_bag_ids.append(basename[0])

            #all_attn_weights.append(attn_weights if attn_weights is None else attn_weights.cpu().numpy())
            #all_attn_weights.append(attn_weights if attn_weights is None else [w.cpu().numpy() for w in attn_weights])
            #all_attn_weights.append(attn_weights)


            # Save attention weights to disk in batches
            if attn_weights is not None and test and False:
                batch_filename = os.path.join(save_path, f"attn_weights_batch_{i}.pt")
                torch.save(attn_weights.cpu(), batch_filename)
                del attn_weights

    return torch.tensor(all_preds, dtype=torch.float32), all_attn_weights, all_bag_ids
    #return torch.tensor(all_preds), all_attn_weights, all_bag_ids #provem sense forçar el float32, ja que estem fent autocast per reduir memoria
