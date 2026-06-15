import torch
import pytest
from loom.amplifier import VCA


class TestVCA:
    def setup_method(self):
        self.vca = VCA()

    def test_output_shape(self):
        signal = torch.randn(4, 1000)
        envelope = torch.ones(4, 1000)
        gain = torch.full((4,), 0.5)
        out = self.vca(signal, envelope, gain)
        assert out.shape == (4, 1000)

    def test_zero_gain_is_silence(self):
        signal = torch.randn(1, 1000)
        envelope = torch.ones(1, 1000)
        gain = torch.tensor([0.0])
        out = self.vca(signal, envelope, gain)
        assert out.abs().max().item() < 0.01

    def test_envelope_shapes_output(self):
        """Applying a half-amplitude envelope should halve the signal."""
        signal = torch.ones(1, 1000)
        envelope = torch.full((1, 1000), 0.5)
        gain = torch.tensor([1.0])  # 0dB
        out = self.vca(signal, envelope, gain)
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=0.01)
