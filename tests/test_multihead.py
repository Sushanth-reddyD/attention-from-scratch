import torch
import pytest
from multihead.attention import MultiHeadAttention


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=16, num_heads=4)
        x = torch.randn(2, 5, 16)
        output, weights = mha(x, x, x)
        assert output.shape == (2, 5, 16)
        assert weights.shape == (2, 4, 5, 5)

    def test_weights_sum_to_one_per_head(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        _, weights = mha(x, x, x)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=7, num_heads=2)

    def test_single_head_equivalent(self):
        mha = MultiHeadAttention(d_model=8, num_heads=1)
        x = torch.randn(1, 3, 8)
        output, weights = mha(x, x, x)
        assert output.shape == (1, 3, 8)
        assert weights.shape == (1, 1, 3, 3)

    def test_cross_attention_shape(self):
        mha = MultiHeadAttention(d_model=16, num_heads=4)
        query = torch.randn(1, 3, 16)
        kv = torch.randn(1, 7, 16)
        output, weights = mha(query, kv, kv)
        assert output.shape == (1, 3, 16)
        assert weights.shape == (1, 4, 3, 7)

    def test_mask_broadcast_3d(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        mask = torch.tril(torch.ones(1, 4, 4))
        _, weights = mha(x, x, x, mask=mask)
        upper = weights[0, :, :, :].triu(diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)

    def test_mask_4d(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        mask = torch.tril(torch.ones(1, 1, 4, 4))
        _, weights = mha(x, x, x, mask=mask)
        upper = weights[0, :, :, :].triu(diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)

    def test_gradients_flow(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 3, 8, requires_grad=True)
        output, _ = mha(x, x, x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        for param in mha.parameters():
            assert param.grad is not None

    def test_parameter_count(self):
        d_model, num_heads = 16, 4
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        total = sum(p.numel() for p in mha.parameters())
        expected = 4 * d_model * d_model
        assert total == expected

    def test_batch_independence(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        mha.eval()
        x = torch.randn(2, 4, 8)
        out_batched, _ = mha(x, x, x)
        out_0, _ = mha(x[0:1], x[0:1], x[0:1])
        out_1, _ = mha(x[1:2], x[1:2], x[1:2])
        assert torch.allclose(out_batched[0], out_0[0], atol=1e-5)
        assert torch.allclose(out_batched[1], out_1[0], atol=1e-5)

    def test_heads_see_different_subspaces(self):
        torch.manual_seed(42)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        _, weights = mha(x, x, x)
        head_0 = weights[0, 0]
        head_1 = weights[0, 1]
        assert not torch.allclose(head_0, head_1, atol=1e-3)

    def test_different_head_counts_same_d_model(self):
        for num_heads in [1, 2, 4, 8]:
            mha = MultiHeadAttention(d_model=8, num_heads=num_heads)
            x = torch.randn(1, 3, 8)
            output, weights = mha(x, x, x)
            assert output.shape == (1, 3, 8)
            assert weights.shape == (1, num_heads, 3, 3)
