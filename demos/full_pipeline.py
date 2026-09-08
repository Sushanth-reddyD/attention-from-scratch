"""End-to-end demo: all 6 attention pieces working together."""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singlehead.attention import SingleHeadAttention
from multihead.attention import MultiHeadAttention
from causal.attention import CausalMultiHeadAttention, create_causal_mask
from crossattention.attention import CrossAttention
from positional.encoding import PositionalEncoding
from transformer.block import TransformerBlock

torch.manual_seed(42)

# --- Config ---
batch_size = 2
seq_len = 10
d_model = 64
num_heads = 4

print("=" * 60)
print("Attention From Scratch — Full Pipeline Demo")
print("=" * 60)
print(f"batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")
print()

# --- Step 1: Random input embeddings ---
x = torch.randn(batch_size, seq_len, d_model)
print(f"1. Input embeddings:          {x.shape}")

# --- Step 2: Positional encoding ---
pe = PositionalEncoding(d_model, max_len=100, dropout=0.0)
x_pos = pe(x)
print(f"2. After positional encoding:  {x_pos.shape}")

# --- Step 3: Single-head attention ---
sha = SingleHeadAttention(d_model, d_k=d_model, d_v=d_model)
sha_out, sha_weights = sha(x_pos, x_pos, x_pos)
print(f"3. Single-head attention out:  {sha_out.shape}  weights: {sha_weights.shape}")

# --- Step 4: Multi-head attention ---
mha = MultiHeadAttention(d_model, num_heads)
mha_out, mha_weights = mha(x_pos, x_pos, x_pos)
print(f"4. Multi-head attention out:   {mha_out.shape}  weights: {mha_weights.shape}")

# --- Step 5: Causal multi-head attention ---
cmha = CausalMultiHeadAttention(d_model, num_heads)
cmha_out, cmha_weights = cmha(x_pos)
print(f"5. Causal MHA out:             {cmha_out.shape}  weights: {cmha_weights.shape}")

# --- Step 6: Cross-attention (decoder attending to encoder) ---
encoder_out = torch.randn(batch_size, 8, d_model)  # encoder seq_len=8
cross = CrossAttention(d_model, num_heads)
cross_out, cross_weights = cross(x_pos, encoder_out)
print(f"6. Cross-attention out:        {cross_out.shape}  weights: {cross_weights.shape}")

# --- Step 7: Full transformer block (MHA + FFN + residual + layernorm) ---
causal_mask = create_causal_mask(seq_len)
block = TransformerBlock(d_model, num_heads, dropout=0.0)
block_out, block_weights = block(x_pos, mask=causal_mask)
print(f"7. Transformer block out:      {block_out.shape}  weights: {block_weights.shape}")

# --- Step 8: Stack 3 transformer blocks ---
blocks = [TransformerBlock(d_model, num_heads, dropout=0.0) for _ in range(3)]
h = x_pos
for i, b in enumerate(blocks):
    h, w = b(h, mask=causal_mask)
print(f"8. After 3 stacked blocks:     {h.shape}")

print()
print("All 6 pieces working together successfully.")
