import torch
import pytest
from loom.effects.distortion import Distortion
from loom.core import DEVICE


class TestDistortion:
    def setup_method(self):
        self.dist = Distortion().to(DEVICE)

    def test_output_shape(self):
        signal = torch.randn(4, 1000, device=DEVICE)
        amount = torch.full((4,), 0.5, device=DEVICE)
        mix = torch.full((4,), 0.5, device=DEVICE)
        out = self.dist(signal, amount, mix)
        assert out.shape == (4, 1000)

    def test_bypass_when_zero_mix(self):
        """Zero mix should pass signal through unchanged."""
        signal = torch.randn(1, 1000, device=DEVICE)
        amount = torch.tensor([0.8], device=DEVICE)
        mix = torch.tensor([0.0], device=DEVICE)
        out = self.dist(signal, amount, mix)
        assert torch.allclose(out, signal, atol=1e-6)

    def test_adds_harmonics(self):
        """Distortion should add harmonic content."""
        t = torch.linspace(0, 1, 44100, device=DEVICE).unsqueeze(0)
        signal = torch.sin(2 * 3.14159 * 440 * t)

        amount = torch.tensor([0.9], device=DEVICE)
        mix = torch.tensor([1.0], device=DEVICE)
        out = self.dist(signal, amount, mix)

        fft_orig = torch.abs(torch.fft.rfft(signal[0]))
        fft_dist = torch.abs(torch.fft.rfft(out[0]))

        fundamental_idx = 440
        harmonic_energy_orig = fft_orig[fundamental_idx * 2:].sum()
        harmonic_energy_dist = fft_dist[fundamental_idx * 2:].sum()
        assert harmonic_energy_dist > harmonic_energy_orig * 2

    def test_output_bounded(self):
        """Output should stay in reasonable range due to tanh."""
        signal = torch.randn(1, 1000, device=DEVICE) * 5.0
        amount = torch.tensor([1.0], device=DEVICE)
        mix = torch.tensor([1.0], device=DEVICE)
        out = self.dist(signal, amount, mix)
        assert out.abs().max().item() <= 1.01

    def test_per_sample_drive(self):
        """amount can be (batch, n_samples) for time-varying drive."""
        signal = torch.sin(torch.linspace(0, 100, 1000, device=DEVICE)).unsqueeze(0)
        amount = torch.linspace(0.0, 1.0, 1000, device=DEVICE).unsqueeze(0)
        mix = torch.tensor([1.0], device=DEVICE)
        out = self.dist(signal, amount, mix)
        assert out.shape == (1, 1000)
        fft_start = torch.abs(torch.fft.rfft(out[0, :500]))
        fft_end = torch.abs(torch.fft.rfft(out[0, 500:]))
        assert fft_end.sum() > fft_start.sum()
