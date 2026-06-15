import torch
import pytest
from loom.effects.compressor import Compressor
from loom.core import N_SAMPLES, DEVICE


class TestCompressor:
    def setup_method(self):
        self.comp = Compressor().to(DEVICE)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES, device=DEVICE)
        out = self.comp(
            signal,
            threshold=torch.full((4,), 0.5, device=DEVICE),
            ratio=torch.full((4,), 0.5, device=DEVICE),
            attack=torch.full((4,), 0.5, device=DEVICE),
            release=torch.full((4,), 0.5, device=DEVICE),
            makeup=torch.full((4,), 0.0, device=DEVICE),
            mix=torch.full((4,), 1.0, device=DEVICE),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        out = self.comp(
            signal,
            threshold=torch.tensor([0.5], device=DEVICE),
            ratio=torch.tensor([0.5], device=DEVICE),
            attack=torch.tensor([0.5], device=DEVICE),
            release=torch.tensor([0.5], device=DEVICE),
            makeup=torch.tensor([0.0], device=DEVICE),
            mix=torch.tensor([0.0], device=DEVICE),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_compresses_loud_signal(self):
        """Loud signal should have lower RMS after compression."""
        signal = torch.randn(1, N_SAMPLES, device=DEVICE) * 2.0
        out = self.comp(
            signal,
            threshold=torch.tensor([0.3], device=DEVICE),
            ratio=torch.tensor([0.8], device=DEVICE),
            attack=torch.tensor([0.3], device=DEVICE),
            release=torch.tensor([0.5], device=DEVICE),
            makeup=torch.tensor([0.0], device=DEVICE),
            mix=torch.tensor([1.0], device=DEVICE),
        )
        assert out.pow(2).mean().sqrt() < signal.pow(2).mean().sqrt()

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        out = self.comp(
            signal,
            threshold=torch.tensor([0.01], device=DEVICE),
            ratio=torch.tensor([0.99], device=DEVICE),
            attack=torch.tensor([0.01], device=DEVICE),
            release=torch.tensor([0.99], device=DEVICE),
            makeup=torch.tensor([0.99], device=DEVICE),
            mix=torch.tensor([1.0], device=DEVICE),
        )
        assert not torch.isnan(out).any()
