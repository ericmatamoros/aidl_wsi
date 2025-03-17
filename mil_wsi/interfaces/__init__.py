"""
Interfaces
"""

from ._metrics import compute_metrics
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

from .attention_mil import(
    AttentionMIL,
    MultiHeadAttention,
    AttentionMILMLP,
    train_attention_mil,
    predict_attention_mil,
)

from .transformer_mil import (
    TransformerMIL,
    train_transformer_model,
    predict_transformer_model
)

from .transformer_marugoto import (
    TransformerMarugoto,
    train_transformer_model,
    predict_transformer_model
)

__all__: list[str] = [
    "compute_metrics",
    "train_mlp",
    "predict_mlp",
    "MLP",
    "MILBagDataset",
    "MLPDataset"
    "plot_loss",
    "AttentionMIL",
    "MultiHeadAttention",
    "AttentionMILMLP",
    "train_attention_mil",
    "predict_attention_mil",
    "TransformerMIL",
    "TransformerMarugoto",
    "train_transformer_model",
    "predict_transformer_model"
]
