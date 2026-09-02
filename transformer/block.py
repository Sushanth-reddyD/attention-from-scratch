import torch
import torch.nn as nn

from multihead.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attn_weights
