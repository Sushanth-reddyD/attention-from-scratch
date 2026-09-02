import torch
import torch.nn as nn

from multihead.attention import MultiHeadAttention


def create_encoder_padding_mask(
    lengths: torch.Tensor, max_len: int, device: torch.device | None = None
) -> torch.Tensor:
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    return (arange < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1).float()


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(
        self,
        decoder_x: torch.Tensor,
        encoder_x: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mha(decoder_x, encoder_x, encoder_x, mask=encoder_mask)
