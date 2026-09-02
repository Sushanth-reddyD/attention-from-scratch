import torch
import math
import pytest
from singlehead.attention import scaled_dot_product_attention, SingleHeadAttention


class TestScaledDotProductAttention:
    def test_output_shape(self):
        Q = torch.randn(2, 5, 8)
        K = torch.randn(2, 5, 8)
        V = torch.randn(2, 5, 8)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (2, 5, 8)
        assert weights.shape == (2, 5, 5)

    def test_weights_sum_to_one(self):
        Q = torch.randn(1, 4, 6)
        K = torch.randn(1, 4, 6)
        V = torch.randn(1, 4, 6)
        _, weights = scaled_dot_product_attention(Q, K, V)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_scaling_factor(self):
        torch.manual_seed(42)
        d_k = 64
        Q = torch.randn(1, 10, d_k)
        K = torch.randn(1, 10, d_k)
        raw_scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled_scores = raw_scores / math.sqrt(d_k)
        expected_weights = torch.softmax(scaled_scores, dim=-1)
        V = torch.randn(1, 10, d_k)
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert torch.allclose(weights, expected_weights, atol=1e-6)

    def test_identical_queries_same_output(self):
        Q = torch.ones(1, 3, 4)
        K = torch.randn(1, 3, 4)
        V = torch.randn(1, 3, 4)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert torch.allclose(output[:, 0], output[:, 1], atol=1e-6)
        assert torch.allclose(output[:, 1], output[:, 2], atol=1e-6)

    def test_one_hot_attention(self):
        d_k = 4
        Q = torch.tensor([[[100.0, 0.0, 0.0, 0.0]]])
        K = torch.tensor([[[100.0, 0.0, 0.0, 0.0],
                           [0.0, 100.0, 0.0, 0.0],
                           [0.0, 0.0, 100.0, 0.0]]])
        V = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]]])
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert torch.allclose(weights[0, 0, 0], torch.tensor(1.0), atol=1e-3)
        assert torch.allclose(output[0, 0], V[0, 0], atol=1e-3)

    def test_mask_zeros_out_positions(self):
        Q = torch.randn(1, 3, 4)
        K = torch.randn(1, 3, 4)
        V = torch.randn(1, 3, 4)
        mask = torch.tensor([[[1, 0, 0],
                              [1, 1, 0],
                              [1, 1, 1]]])
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert torch.allclose(weights[0, 0, 1], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(weights[0, 0, 2], torch.tensor(0.0), atol=1e-6)
        assert weights[0, 0, 0] > 0.99

    def test_mask_weights_still_sum_to_one(self):
        Q = torch.randn(1, 4, 6)
        K = torch.randn(1, 4, 6)
        V = torch.randn(1, 4, 6)
        mask = torch.tril(torch.ones(1, 4, 4))
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_different_seq_lengths_qk_vs_v(self):
        Q = torch.randn(1, 3, 8)
        K = torch.randn(1, 5, 8)
        V = torch.randn(1, 5, 8)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (1, 3, 8)
        assert weights.shape == (1, 3, 5)

    def test_batch_independence(self):
        Q = torch.randn(2, 3, 4)
        K = torch.randn(2, 3, 4)
        V = torch.randn(2, 3, 4)
        output_batched, _ = scaled_dot_product_attention(Q, K, V)
        output_0, _ = scaled_dot_product_attention(
            Q[0:1], K[0:1], V[0:1]
        )
        output_1, _ = scaled_dot_product_attention(
            Q[1:2], K[1:2], V[1:2]
        )
        assert torch.allclose(output_batched[0], output_0[0], atol=1e-6)
        assert torch.allclose(output_batched[1], output_1[0], atol=1e-6)

    def test_no_batch_dim(self):
        Q = torch.randn(3, 4)
        K = torch.randn(3, 4)
        V = torch.randn(3, 4)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (3, 4)
        assert weights.shape == (3, 3)


class TestSingleHeadAttention:
    def test_output_shape(self):
        attn = SingleHeadAttention(d_model=8, d_k=4, d_v=4)
        x = torch.randn(2, 5, 8)
        output, weights = attn(x, x, x)
        assert output.shape == (2, 5, 8)
        assert weights.shape == (2, 5, 5)

    def test_self_attention_mode(self):
        attn = SingleHeadAttention(d_model=16, d_k=8, d_v=8)
        x = torch.randn(1, 10, 16)
        output, weights = attn(x, x, x)
        assert output.shape == (1, 10, 16)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_cross_attention_mode(self):
        attn = SingleHeadAttention(d_model=8, d_k=4, d_v=4)
        query = torch.randn(1, 3, 8)
        kv = torch.randn(1, 7, 8)
        output, weights = attn(query, kv, kv)
        assert output.shape == (1, 3, 8)
        assert weights.shape == (1, 3, 7)

    def test_gradients_flow(self):
        attn = SingleHeadAttention(d_model=8, d_k=4, d_v=4)
        x = torch.randn(1, 3, 8, requires_grad=True)
        output, _ = attn(x, x, x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        for param in attn.parameters():
            assert param.grad is not None

    def test_with_mask(self):
        attn = SingleHeadAttention(d_model=8, d_k=4, d_v=4)
        x = torch.randn(1, 4, 8)
        mask = torch.tril(torch.ones(1, 4, 4))
        _, weights = attn(x, x, x, mask=mask)
        upper = weights[0].triu(diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)

    def test_parameter_count(self):
        d_model, d_k, d_v = 8, 4, 6
        attn = SingleHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v)
        total = sum(p.numel() for p in attn.parameters())
        expected = d_model * d_k + d_model * d_k + d_model * d_v + d_v * d_model
        assert total == expected

    def test_different_dk_dv(self):
        attn = SingleHeadAttention(d_model=16, d_k=8, d_v=12)
        x = torch.randn(1, 5, 16)
        output, weights = attn(x, x, x)
        assert output.shape == (1, 5, 16)
        assert weights.shape == (1, 5, 5)
