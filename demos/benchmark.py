"""Benchmark attention forward pass at different sequence lengths.

Shows O(n^2) scaling of self-attention: doubling seq_len ~4x the time.
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multihead.attention import MultiHeadAttention
from causal.attention import CausalMultiHeadAttention, create_causal_mask
from transformer.block import TransformerBlock

torch.manual_seed(42)

BATCH_SIZE = 2
D_MODEL = 64
NUM_HEADS = 4
SEQ_LENGTHS = [32, 64, 128, 256, 512, 1024]
WARMUP_RUNS = 3
TIMED_RUNS = 10


def benchmark(name: str, fn, seq_lengths: list[int]) -> dict[int, float]:
    """Time fn(seq_len) for each seq_len. Returns {seq_len: avg_ms}."""
    results = {}
    for seq_len in seq_lengths:
        # warmup
        for _ in range(WARMUP_RUNS):
            fn(seq_len)

        # timed
        times = []
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            fn(seq_len)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms
        results[seq_len] = sum(times) / len(times)
    return results


def run_mha(seq_len: int):
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    with torch.no_grad():
        mha(x, x, x)


def run_causal_mha(seq_len: int):
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
    cmha = CausalMultiHeadAttention(D_MODEL, NUM_HEADS)
    with torch.no_grad():
        cmha(x)


def run_transformer_block(seq_len: int):
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
    mask = create_causal_mask(seq_len)
    block = TransformerBlock(D_MODEL, NUM_HEADS, dropout=0.0)
    with torch.no_grad():
        block(x, mask=mask)


def print_table(name: str, results: dict[int, float]):
    print(f"\n{name}")
    print("-" * 45)
    print(f"{'seq_len':>10}  {'time (ms)':>10}  {'ratio vs prev':>14}")
    print("-" * 45)
    prev = None
    for seq_len in SEQ_LENGTHS:
        ms = results[seq_len]
        if prev is not None:
            ratio = ms / prev
            print(f"{seq_len:>10}  {ms:>10.3f}  {ratio:>13.2f}x")
        else:
            print(f"{seq_len:>10}  {ms:>10.3f}  {'—':>14}")
        prev = ms
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("Attention Benchmark — O(n^2) Scaling Demo")
    print(f"batch={BATCH_SIZE}, d_model={D_MODEL}, heads={NUM_HEADS}")
    print(f"warmup={WARMUP_RUNS}, timed_runs={TIMED_RUNS}")
    print("=" * 55)

    for name, fn in [
        ("Multi-Head Attention", run_mha),
        ("Causal Multi-Head Attention", run_causal_mha),
        ("Transformer Block", run_transformer_block),
    ]:
        results = benchmark(name, fn, SEQ_LENGTHS)
        print_table(name, results)

    print("Expect ~4x time when doubling seq_len (O(n^2) attention).")
    print("Actual ratios vary due to overhead, memory, and small sizes.")
