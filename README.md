# Attention From Scratch

Attention mechanisms implemented from scratch in PyTorch — no `nn.MultiheadAttention`, no wrappers. Every matrix multiply is explicit.

Built for understanding how attention actually works, piece by piece.

## Roadmap

| # | Piece | What it covers |
|---|-------|----------------|
| 1 | **Single-Head Attention** | QKV math, softmax, √d_k scaling, attention weights |
| 2 | **Multi-Head Attention** | Parallel heads, split/concat, independent subspaces |
| 3 | **Causal (Masked) Attention** | Autoregressive mask, preventing future token leakage |
| 4 | **Cross-Attention** | Encoder-decoder attention, Q from decoder, KV from encoder |
| 5 | **Positional Encoding** | Sinusoidal positional encodings (Vaswani et al.) |
| 6 | **Full Transformer Block** | Attention + FFN + LayerNorm + residual connections |

Each piece builds on the last. Every piece has its own module, tests, and demo.

## Structure

```
attention-from-scratch/
├── singlehead/            # Piece 1: Single-head attention
├── multihead/             # Piece 2: Multi-head attention
├── causal/                # Piece 3: Causal masking
├── crossattention/        # Piece 4: Cross-attention
├── positional/            # Piece 5: Positional encoding
├── transformer/           # Piece 6: Full transformer block
├── tests/                 # All tests
├── demos/                 # Runnable demos per piece
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest tests/ -v
```
