import torch
import pytest
from causal.attention import (
    create_causal_mask,
    create_padding_mask,
    combine_masks,
    CausalMultiHeadAttention,
)


class TestCreateCausalMask:
    def test_shape(self):
        mask = create_causal_mask(5)
        assert mask.shape == (1, 1, 5, 5)

    def test_lower_triangular(self):
        mask = create_causal_mask(4)
        expected = torch.tensor(
            [[[[1, 0, 0, 0],
               [1, 1, 0, 0],
               [1, 1, 1, 0],
               [1, 1, 1, 1]]]], dtype=torch.float
        )
        assert torch.equal(mask, expected)

    def test_single_token(self):
        mask = create_causal_mask(1)
        assert torch.equal(mask, torch.ones(1, 1, 1, 1))

    def test_upper_triangle_is_zero(self):
        mask = create_causal_mask(6)
        upper = mask[0, 0].triu(diagonal=1)
        assert torch.equal(upper, torch.zeros(6, 6))


class TestCreatePaddingMask:
    def test_shape(self):
        lengths = torch.tensor([3, 5])
        mask = create_padding_mask(lengths, max_len=5)
        assert mask.shape == (2, 1, 1, 5)

    def test_masks_pad_positions(self):
        lengths = torch.tensor([2])
        mask = create_padding_mask(lengths, max_len=4)
        expected = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.float)
        assert torch.equal(mask, expected)

    def test_full_length_no_masking(self):
        lengths = torch.tensor([4])
        mask = create_padding_mask(lengths, max_len=4)
        assert torch.equal(mask, torch.ones(1, 1, 1, 4))

    def test_batch(self):
        lengths = torch.tensor([2, 4, 1])
        mask = create_padding_mask(lengths, max_len=4)
        assert mask[0, 0, 0, 1] == 1.0
        assert mask[0, 0, 0, 2] == 0.0
        assert mask[1, 0, 0, 3] == 1.0
        assert mask[2, 0, 0, 1] == 0.0


class TestCombineMasks:
    def test_causal_and_padding(self):
        causal = create_causal_mask(4)
        lengths = torch.tensor([2])
        padding = create_padding_mask(lengths, max_len=4)
        combined = combine_masks(causal, padding)
        expected = torch.tensor(
            [[[[1, 0, 0, 0],
               [1, 1, 0, 0],
               [1, 1, 0, 0],
               [1, 1, 0, 0]]]], dtype=torch.float
        )
        assert torch.equal(combined, expected)

    def test_no_padding_equals_causal(self):
        causal = create_causal_mask(3)
        lengths = torch.tensor([3])
        padding = create_padding_mask(lengths, max_len=3)
        combined = combine_masks(causal, padding)
        assert torch.equal(combined, causal)


class TestCausalMultiHeadAttention:
    def test_output_shape(self):
        attn = CausalMultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(2, 5, 8)
        output, weights = attn(x)
        assert output.shape == (2, 5, 8)
        assert weights.shape == (2, 2, 5, 5)

    def test_future_tokens_get_zero_weight(self):
        attn = CausalMultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        _, weights = attn(x)
        for head in range(2):
            upper = weights[0, head].triu(diagonal=1)
            assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)

    def test_first_token_attends_only_to_self(self):
        attn = CausalMultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        _, weights = attn(x)
        for head in range(2):
            assert torch.allclose(
                weights[0, head, 0, 0], torch.tensor(1.0), atol=1e-6
            )

    def test_weights_sum_to_one(self):
        attn = CausalMultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 6, 8)
        _, weights = attn(x)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_changing_future_tokens_no_effect(self):
        torch.manual_seed(0)
        attn = CausalMultiHeadAttention(d_model=8, num_heads=2)
        attn.eval()

        x1 = torch.randn(1, 4, 8)
        x2 = x1.clone()
        x2[0, 2:] = torch.randn(2, 8)

        out1, _ = attn(x1)
        out2, _ = attn(x2)

        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)
        assert torch.allclose(out1[0, 1], out2[0, 1], atol=1e-5)

    def test_with_padding_mask(self):
        attn = CausalMultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        lengths = torch.tensor([2])
        padding_mask = create_padding_mask(lengths, max_len=4)
        _, weights = attn(x, padding_mask=padding_mask)
        for head in range(2):
            assert torch.allclose(
                weights[0, head, :, 2], torch.tensor(0.0), atol=1e-6
            )
            assert torch.allclose(
                weights[0, head, :, 3], torch.tensor(0.0), atol=1e-6
            )

    def test_gradients_flow(self):
        attn = CausalMultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 3, 8, requires_grad=True)
        output, _ = attn(x)
        output.sum().backward()
        assert x.grad is not None
        for param in attn.parameters():
            assert param.grad is not None

    def test_single_token(self):
        attn = CausalMultiHeadAttention(d_model=8, num_heads=2)
        x = torch.randn(1, 1, 8)
        output, weights = attn(x)
        assert output.shape == (1, 1, 8)
        assert torch.allclose(
            weights, torch.ones(1, 2, 1, 1), atol=1e-6
        )
