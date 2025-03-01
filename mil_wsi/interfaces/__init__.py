"""
Interfaces
"""

from .metrics import compute_metrics
from .mlp_model import (
    train_mlp,
    predict_mlp,
    MLP
)
from .mlp_dataset import (
    MLPDataset
)
from .plot_loss import (
    plot_loss
)

from .mil_dataset import (
    MILBagDataset
)
from .mil_model import (
    MIL,
    train_mil,
    predict_mil
)

from .attention_mil import(
    AttentionMIL,
    MultiHeadAttention,
    AttentionMILMLP,
    train_attention_mil,
    predict_attention_mil,
)

__all__: list[str] = [
    "compute_metrics",
    "train_mlp",
    "predict_mlp",
    "MLP",
    "MILBagDataset",
    "MIL",
    "train_mil", 
    "predict_mil",
    "AttentionMIL",
    "train_attention_mil",
    "predict_attention_mil",
    "MLPDataset"
    "plot_loss"
]
