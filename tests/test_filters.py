import torch
import pytest
from loom.filters import BiquadFilter
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestBiquadFilter:
    def setup_method(self):
        self.filt = BiquadFilter(sample_rate=SAMPLE_RATE).to(DEVICE)

    def test_output_shape(self):
        batch = 4
        signal = torch.randn(batch, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((batch,), 0.5, device=DEVICE)
        q = torch.full((batch,), 0.5, device=DEVICE)
        filter_type = torch.zeros(batch, 3, device=DEVICE)
        filter_type[:, 0] = 1.0
        out = self.filt(signal, cutoff, q, filter_type)
        assert out.shape == (batch, N_SAMPLES)

    def test_lowpass_attenuates_highs(self):
        """LP filter at low cutoff should attenuate energy above 5kHz."""
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.tensor([0.3], device=DEVICE)
        q = torch.tensor([0.5], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 0] = 1.0

        filtered = self.filt(noise, cutoff, q, filter_type)
        fft_orig = torch.abs(torch.fft.rfft(noise[0]))
        fft_filt = torch.abs(torch.fft.rfft(filtered[0]))

        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE, device=DEVICE)
        high_mask = freqs > 5000
        high_energy_orig = fft_orig[high_mask].mean()
        high_energy_filt = fft_filt[high_mask].mean()
        assert high_energy_filt < high_energy_orig * 0.5

    def test_highpass_attenuates_lows(self):
        """HP filter should attenuate energy below cutoff."""
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.tensor([0.7], device=DEVICE)
        q = torch.tensor([0.5], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 1] = 1.0

        filtered = self.filt(noise, cutoff, q, filter_type)
        fft_orig = torch.abs(torch.fft.rfft(noise[0]))
        fft_filt = torch.abs(torch.fft.rfft(filtered[0]))

        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE, device=DEVICE)
        low_mask = freqs < 500
        low_energy_orig = fft_orig[low_mask].mean()
        low_energy_filt = fft_filt[low_mask].mean()
        assert low_energy_filt < low_energy_orig * 0.5

    def test_no_nan(self):
        """Should not produce NaN for extreme parameters."""
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.tensor([0.01], device=DEVICE)
        q = torch.tensor([0.99], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 0] = 1.0
        out = self.filt(signal, cutoff, q, filter_type)
        assert not torch.isnan(out).any()
