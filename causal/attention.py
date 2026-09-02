import torch
import torch.nn as nn

from multihead.attention import MultiHeadAttention


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)


def create_padding_mask(
    lengths: torch.Tensor, max_len: int, device: torch.device | None = None
) -> torch.Tensor:
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    return (arange < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2).float()


def combine_masks(
    causal_mask: torch.Tensor, padding_mask: torch.Tensor
) -> torch.Tensor:
    return causal_mask * padding_mask


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, device=x.device)

        if padding_mask is not None:
            mask = combine_masks(causal_mask, padding_mask)
        else:
            mask = causal_mask

        return self.mha(x, x, x, mask=mask)
