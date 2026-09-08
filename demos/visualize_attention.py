"""Visualize attention weight heatmaps from causal multi-head attention.

Generates a figure with one subplot per attention head showing the
attention pattern, and saves it to demos/attention_weights.png.
"""

import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal.attention import CausalMultiHeadAttention

torch.manual_seed(42)

# --- Config ---
BATCH_SIZE = 1
SEQ_LEN = 16
D_MODEL = 64
NUM_HEADS = 4


def main():
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    model = CausalMultiHeadAttention(D_MODEL, NUM_HEADS)
    with torch.no_grad():
        _, attn_weights = model(x)

    # attn_weights: (1, num_heads, seq_len, seq_len)
    weights = attn_weights.squeeze(0).numpy()  # (num_heads, seq_len, seq_len)

    fig, axes = plt.subplots(1, NUM_HEADS, figsize=(4 * NUM_HEADS, 4))
    if NUM_HEADS == 1:
        axes = [axes]

    for head_idx, ax in enumerate(axes):
        im = ax.imshow(weights[head_idx], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Head {head_idx}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_xticks(range(0, SEQ_LEN, 4))
        ax.set_yticks(range(0, SEQ_LEN, 4))

    fig.suptitle("Causal Multi-Head Attention Weights", fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Attention weight")
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention_weights.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved attention heatmap to {save_path}")
    print(f"Figure shows {NUM_HEADS} heads, each {SEQ_LEN}x{SEQ_LEN}")
    print("Lower triangle pattern = causal mask (no future token attention)")


if __name__ == "__main__":
    main()
