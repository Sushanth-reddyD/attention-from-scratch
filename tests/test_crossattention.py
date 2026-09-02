import torch
import pytest
from crossattention.attention import CrossAttention, create_encoder_padding_mask


class TestCreateEncoderPaddingMask:
    def test_shape(self):
        lengths = torch.tensor([3, 5])
        mask = create_encoder_padding_mask(lengths, max_len=5)
        assert mask.shape == (2, 1, 1, 5)

    def test_masks_pad_positions(self):
        lengths = torch.tensor([3])
        mask = create_encoder_padding_mask(lengths, max_len=5)
        expected = torch.tensor([[[[1, 1, 1, 0, 0]]]], dtype=torch.float)
        assert torch.equal(mask, expected)

    def test_full_length(self):
        lengths = torch.tensor([4])
        mask = create_encoder_padding_mask(lengths, max_len=4)
        assert torch.equal(mask, torch.ones(1, 1, 1, 4))

    def test_batch(self):
        lengths = torch.tensor([1, 4, 2])
        mask = create_encoder_padding_mask(lengths, max_len=4)
        assert mask[0, 0, 0, 0] == 1.0
        assert mask[0, 0, 0, 1] == 0.0
        assert mask[1, 0, 0, 3] == 1.0
        assert mask[2, 0, 0, 2] == 0.0


class TestCrossAttention:
    def test_output_shape(self):
        ca = CrossAttention(d_model=8, num_heads=2)
        decoder_x = torch.randn(1, 3, 8)
        encoder_x = torch.randn(1, 7, 8)
        output, weights = ca(decoder_x, encoder_x)
        assert output.shape == (1, 3, 8)
        assert weights.shape == (1, 2, 3, 7)

    def test_output_matches_decoder_seq_len(self):
        ca = CrossAttention(d_model=16, num_heads=4)
        for dec_len in [1, 5, 10]:
            decoder_x = torch.randn(1, dec_len, 16)
            encoder_x = torch.randn(1, 8, 16)
            output, weights = ca(decoder_x, encoder_x)
            assert output.shape == (1, dec_len, 16)
            assert weights.shape[2] == dec_len
            assert weights.shape[3] == 8

    def test_weights_sum_to_one(self):
        ca = CrossAttention(d_model=8, num_heads=2)
        decoder_x = torch.randn(1, 3, 8)
        encoder_x = torch.randn(1, 5, 8)
        _, weights = ca(decoder_x, encoder_x)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_encoder_mask_zeros_out_padded(self):
        ca = CrossAttention(d_model=8, num_heads=2)
        decoder_x = torch.randn(1, 3, 8)
        encoder_x = torch.randn(1, 5, 8)
        lengths = torch.tensor([2])
        mask = create_encoder_padding_mask(lengths, max_len=5)
        _, weights = ca(decoder_x, encoder_x, encoder_mask=mask)
        for head in range(2):
            for pos in range(2, 5):
                assert torch.allclose(
                    weights[0, head, :, pos],
                    torch.zeros(3),
                    atol=1e-6,
                )

    def test_encoder_change_does_affect_output(self):
        torch.manual_seed(0)
        ca = CrossAttention(d_model=8, num_heads=2)
        ca.eval()
        decoder_x = torch.randn(1, 3, 8)
        encoder_a = torch.randn(1, 5, 8)
        encoder_b = torch.randn(1, 5, 8)
        out_a, _ = ca(decoder_x, encoder_a)
        out_b, _ = ca(decoder_x, encoder_b)
        assert not torch.allclose(out_a, out_b, atol=1e-3)

    def test_decoder_change_does_not_affect_other_via_encoder(self):
        torch.manual_seed(0)
        ca = CrossAttention(d_model=8, num_heads=2)
        ca.eval()
        encoder_x = torch.randn(1, 5, 8)
        dec1 = torch.randn(1, 3, 8)
        dec2 = dec1.clone()
        dec2[0, 2] = torch.randn(8)
        out1, _ = ca(dec1, encoder_x)
        out2, _ = ca(dec2, encoder_x)
        assert not torch.allclose(out1[0, 2], out2[0, 2], atol=1e-3)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)

    def test_gradients_flow_to_both_inputs(self):
        ca = CrossAttention(d_model=8, num_heads=2)
        decoder_x = torch.randn(1, 3, 8, requires_grad=True)
        encoder_x = torch.randn(1, 5, 8, requires_grad=True)
        output, _ = ca(decoder_x, encoder_x)
        output.sum().backward()
        assert decoder_x.grad is not None
        assert encoder_x.grad is not None
        for param in ca.parameters():
            assert param.grad is not None

    def test_batch_independence(self):
        ca = CrossAttention(d_model=8, num_heads=2)
        ca.eval()
        decoder_x = torch.randn(2, 3, 8)
        encoder_x = torch.randn(2, 5, 8)
        out_batched, _ = ca(decoder_x, encoder_x)
        out_0, _ = ca(decoder_x[0:1], encoder_x[0:1])
        out_1, _ = ca(decoder_x[1:2], encoder_x[1:2])
        assert torch.allclose(out_batched[0], out_0[0], atol=1e-5)
        assert torch.allclose(out_batched[1], out_1[0], atol=1e-5)

    def test_same_input_reduces_to_self_attention(self):
        ca = CrossAttention(d_model=8, num_heads=2)
        ca.eval()
        x = torch.randn(1, 4, 8)
        output, weights = ca(x, x)
        assert output.shape == (1, 4, 8)
        assert weights.shape == (1, 2, 4, 4)

    def test_single_decoder_token(self):
        ca = CrossAttention(d_model=8, num_heads=2)
        decoder_x = torch.randn(1, 1, 8)
        encoder_x = torch.randn(1, 6, 8)
        output, weights = ca(decoder_x, encoder_x)
        assert output.shape == (1, 1, 8)
        assert weights.shape == (1, 2, 1, 6)

    def test_parameter_count_same_as_multihead(self):
        d_model = 16
        ca = CrossAttention(d_model=d_model, num_heads=4)
        total = sum(p.numel() for p in ca.parameters())
        expected = 4 * d_model * d_model
        assert total == expected
