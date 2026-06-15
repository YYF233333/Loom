import torch
import pytest
from loom.effects.chorus import Chorus
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestChorus:
    def setup_method(self):
        self.chorus = Chorus(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES, device=DEVICE)
        out = self.chorus(
            signal,
            rate=torch.full((4,), 0.5, device=DEVICE),
            depth=torch.full((4,), 0.5, device=DEVICE),
            mix=torch.full((4,), 0.5, device=DEVICE),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        out = self.chorus(
            signal,
            rate=torch.tensor([0.5], device=DEVICE),
            depth=torch.tensor([0.5], device=DEVICE),
            mix=torch.tensor([0.0], device=DEVICE),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_spectral_spreading(self):
        """Chorus should widen the spectrum around the fundamental."""
        t = torch.arange(N_SAMPLES, dtype=torch.float32, device=DEVICE) / SAMPLE_RATE
        signal = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

        out = self.chorus(
            signal,
            rate=torch.tensor([0.5], device=DEVICE),
            depth=torch.tensor([0.8], device=DEVICE),
            mix=torch.tensor([1.0], device=DEVICE),
        )

        fft_orig = torch.abs(torch.fft.rfft(signal[0]))
        fft_chorus = torch.abs(torch.fft.rfft(out[0]))
        peak = torch.argmax(fft_orig[1:]) + 1
        sideband_orig = fft_orig[peak - 20 : peak + 20].sum() - fft_orig[peak]
        sideband_chorus = fft_chorus[peak - 20 : peak + 20].sum() - fft_chorus[peak]
        assert sideband_chorus > sideband_orig

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        out = self.chorus(
            signal,
            rate=torch.tensor([0.99], device=DEVICE),
            depth=torch.tensor([0.99], device=DEVICE),
            mix=torch.tensor([1.0], device=DEVICE),
        )
        assert not torch.isnan(out).any()
