import torch
import math
import pytest
from positional.encoding import sinusoidal_encoding, PositionalEncoding


class TestSinusoidalEncoding:
    def test_shape(self):
        pe = sinusoidal_encoding(100, 64)
        assert pe.shape == (100, 64)

    def test_position_zero(self):
        pe = sinusoidal_encoding(5, 8)
        assert torch.allclose(pe[0, 0::2], torch.zeros(4), atol=1e-6)
        assert torch.allclose(pe[0, 1::2], torch.ones(4), atol=1e-6)

    def test_even_dims_are_sin_odd_are_cos(self):
        pe = sinusoidal_encoding(10, 4)
        pos = 3
        div_0 = math.exp(0 * -(math.log(10000.0) / 4))
        div_1 = math.exp(2 * -(math.log(10000.0) / 4))
        assert torch.allclose(pe[pos, 0], torch.tensor(math.sin(pos * div_0)), atol=1e-5)
        assert torch.allclose(pe[pos, 1], torch.tensor(math.cos(pos * div_0)), atol=1e-5)
        assert torch.allclose(pe[pos, 2], torch.tensor(math.sin(pos * div_1)), atol=1e-5)
        assert torch.allclose(pe[pos, 3], torch.tensor(math.cos(pos * div_1)), atol=1e-5)

    def test_each_position_is_unique(self):
        pe = sinusoidal_encoding(100, 32)
        for i in range(100):
            for j in range(i + 1, 100):
                assert not torch.allclose(pe[i], pe[j], atol=1e-4)

    def test_values_bounded(self):
        pe = sinusoidal_encoding(500, 128)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_high_freq_dims_change_fast(self):
        pe = sinusoidal_encoding(10, 64)
        diff_dim0 = (pe[1:, 0] - pe[:-1, 0]).abs().mean()
        diff_dim_last = (pe[1:, -2] - pe[:-1, -2]).abs().mean()
        assert diff_dim0 > diff_dim_last

    def test_deterministic(self):
        pe1 = sinusoidal_encoding(50, 16)
        pe2 = sinusoidal_encoding(50, 16)
        assert torch.equal(pe1, pe2)

    def test_relative_position_linear_property(self):
        d_model = 8
        pe = sinusoidal_encoding(100, d_model)
        k = 5
        for dim_pair in range(d_model // 2):
            sin_col = 2 * dim_pair
            cos_col = 2 * dim_pair + 1
            for pos in range(50):
                sin_p = pe[pos, sin_col]
                cos_p = pe[pos, cos_col]
                sin_pk = pe[pos + k, sin_col]
                cos_pk = pe[pos + k, cos_col]
                sin_k = pe[k, sin_col]
                cos_k = pe[k, cos_col]
                expected_sin = sin_p * cos_k + cos_p * sin_k
                expected_cos = cos_p * cos_k - sin_p * sin_k
                assert torch.allclose(sin_pk, expected_sin, atol=1e-5)
                assert torch.allclose(cos_pk, expected_cos, atol=1e-5)


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=16, max_len=100, dropout=0.0)
        x = torch.randn(2, 10, 16)
        out = pe(x)
        assert out.shape == (2, 10, 16)

    def test_adds_to_input(self):
        pe_mod = PositionalEncoding(d_model=8, max_len=50, dropout=0.0)
        x = torch.zeros(1, 5, 8)
        out = pe_mod(x)
        expected = sinusoidal_encoding(50, 8)[:5]
        assert torch.allclose(out[0], expected, atol=1e-6)

    def test_different_positions_get_different_encoding(self):
        pe_mod = PositionalEncoding(d_model=16, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 16)
        out = pe_mod(x)
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-4)

    def test_same_word_different_position(self):
        pe_mod = PositionalEncoding(d_model=8, max_len=50, dropout=0.0)
        word = torch.randn(1, 1, 8)
        x = word.expand(1, 5, 8).clone()
        out = pe_mod(x)
        assert not torch.allclose(out[0, 0], out[0, 3], atol=1e-4)

    def test_shorter_than_max_len(self):
        pe_mod = PositionalEncoding(d_model=8, max_len=1000, dropout=0.0)
        x = torch.randn(1, 5, 8)
        out = pe_mod(x)
        assert out.shape == (1, 5, 8)

    def test_dropout_applied(self):
        pe_mod = PositionalEncoding(d_model=16, max_len=50, dropout=0.5)
        pe_mod.train()
        x = torch.ones(1, 10, 16)
        out = pe_mod(x)
        assert (out == 0.0).any()

    def test_no_dropout_in_eval(self):
        pe_mod = PositionalEncoding(d_model=8, max_len=50, dropout=0.5)
        pe_mod.eval()
        x = torch.ones(1, 5, 8)
        out = pe_mod(x)
        assert not (out == 0.0).any()

    def test_pe_is_buffer_not_parameter(self):
        pe_mod = PositionalEncoding(d_model=8)
        assert len(list(pe_mod.parameters())) == 0
        assert "pe" in dict(pe_mod.named_buffers())

    def test_batch_gets_same_encoding(self):
        pe_mod = PositionalEncoding(d_model=8, max_len=50, dropout=0.0)
        x = torch.zeros(3, 5, 8)
        out = pe_mod(x)
        assert torch.allclose(out[0], out[1], atol=1e-6)
        assert torch.allclose(out[1], out[2], atol=1e-6)
