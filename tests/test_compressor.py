import torch
import pytest
from loom.effects.compressor import Compressor
from loom.core import N_SAMPLES


class TestCompressor:
    def setup_method(self):
        self.comp = Compressor()

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES)
        out = self.comp(
            signal,
            threshold=torch.full((4,), 0.5),
            ratio=torch.full((4,), 0.5),
            attack=torch.full((4,), 0.5),
            release=torch.full((4,), 0.5),
            makeup=torch.full((4,), 0.0),
            mix=torch.full((4,), 1.0),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.comp(
            signal,
            threshold=torch.tensor([0.5]),
            ratio=torch.tensor([0.5]),
            attack=torch.tensor([0.5]),
            release=torch.tensor([0.5]),
            makeup=torch.tensor([0.0]),
            mix=torch.tensor([0.0]),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_compresses_loud_signal(self):
        """Loud signal should have lower RMS after compression."""
        signal = torch.randn(1, N_SAMPLES) * 2.0
        out = self.comp(
            signal,
            threshold=torch.tensor([0.3]),
            ratio=torch.tensor([0.8]),
            attack=torch.tensor([0.3]),
            release=torch.tensor([0.5]),
            makeup=torch.tensor([0.0]),
            mix=torch.tensor([1.0]),
        )
        assert out.pow(2).mean().sqrt() < signal.pow(2).mean().sqrt()

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.comp(
            signal,
            threshold=torch.tensor([0.01]),
            ratio=torch.tensor([0.99]),
            attack=torch.tensor([0.01]),
            release=torch.tensor([0.99]),
            makeup=torch.tensor([0.99]),
            mix=torch.tensor([1.0]),
        )
        assert not torch.isnan(out).any()
