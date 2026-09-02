import torch
import torch.nn as nn
import pytest
from transformer.block import FeedForward, TransformerBlock


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(d_model=8, d_ff=32)
        x = torch.randn(2, 5, 8)
        out = ffn(x)
        assert out.shape == (2, 5, 8)

    def test_relu_kills_negatives(self):
        ffn = FeedForward(d_model=4, d_ff=16, dropout=0.0)
        ffn.eval()
        x = torch.randn(1, 3, 4)
        hidden = ffn.relu(ffn.linear1(x))
        assert (hidden >= 0).all()

    def test_parameter_count(self):
        d_model, d_ff = 8, 32
        ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
        total = sum(p.numel() for p in ffn.parameters())
        expected = d_model * d_ff + d_ff + d_ff * d_model + d_model
        assert total == expected

    def test_gradients_flow(self):
        ffn = FeedForward(d_model=8, d_ff=32)
        x = torch.randn(1, 3, 8, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_position_wise(self):
        ffn = FeedForward(d_model=8, d_ff=32, dropout=0.0)
        ffn.eval()
        x = torch.randn(1, 4, 8)
        out_full = ffn(x)
        out_single = ffn(x[:, 2:3, :])
        assert torch.allclose(out_full[:, 2], out_single[:, 0], atol=1e-6)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=16, num_heads=4)
        x = torch.randn(2, 5, 16)
        out, weights = block(x)
        assert out.shape == (2, 5, 16)
        assert weights.shape == (2, 4, 5, 5)

    def test_default_d_ff_is_4x(self):
        block = TransformerBlock(d_model=8, num_heads=2)
        assert block.ffn.linear1.out_features == 32

    def test_custom_d_ff(self):
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=64)
        assert block.ffn.linear1.out_features == 64

    def test_residual_connection_not_zero(self):
        block = TransformerBlock(d_model=8, num_heads=2, dropout=0.0)
        block.eval()
        x = torch.randn(1, 3, 8)
        out, _ = block(x)
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-3)

    def test_output_is_normalized(self):
        block = TransformerBlock(d_model=16, num_heads=4, dropout=0.0)
        block.eval()
        x = torch.randn(1, 5, 16)
        out, _ = block(x)
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-1)

    def test_residual_preserves_input_influence(self):
        torch.manual_seed(0)
        block = TransformerBlock(d_model=8, num_heads=2, dropout=0.0)
        block.eval()
        x1 = torch.randn(1, 3, 8)
        x2 = x1.clone()
        x2[0, 1] += 10.0
        out1, _ = block(x1)
        out2, _ = block(x2)
        diff = (out1[0, 1] - out2[0, 1]).abs().mean()
        assert diff > 0.1

    def test_with_mask(self):
        block = TransformerBlock(d_model=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        mask = torch.tril(torch.ones(1, 1, 4, 4))
        _, weights = block(x, mask=mask)
        upper = weights[0, :, :, :].triu(diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)

    def test_gradients_flow(self):
        block = TransformerBlock(d_model=8, num_heads=2)
        x = torch.randn(1, 3, 8, requires_grad=True)
        out, _ = block(x)
        out.sum().backward()
        assert x.grad is not None
        for param in block.parameters():
            assert param.grad is not None

    def test_batch_independence(self):
        block = TransformerBlock(d_model=8, num_heads=2, dropout=0.0)
        block.eval()
        x = torch.randn(2, 4, 8)
        out_batched, _ = block(x)
        out_0, _ = block(x[0:1])
        out_1, _ = block(x[1:2])
        assert torch.allclose(out_batched[0], out_0[0], atol=1e-5)
        assert torch.allclose(out_batched[1], out_1[0], atol=1e-5)

    def test_stacking_blocks(self):
        blocks = nn.ModuleList([
            TransformerBlock(d_model=8, num_heads=2, dropout=0.0)
            for _ in range(3)
        ])
        x = torch.randn(1, 4, 8)
        for block in blocks:
            x, _ = block(x)
        assert x.shape == (1, 4, 8)

    def test_parameter_count(self):
        d_model, num_heads, d_ff = 16, 4, 64
        block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        total = sum(p.numel() for p in block.parameters())
        attn_params = 4 * d_model * d_model
        ffn_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
        norm_params = 2 * (d_model + d_model)
        assert total == attn_params + ffn_params + norm_params

    def test_weights_returned(self):
        block = TransformerBlock(d_model=8, num_heads=2)
        x = torch.randn(1, 3, 8)
        _, weights = block(x)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
