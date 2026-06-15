import torch
import pytest
from loom.svfilter import SVFilter
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestSVFilter:
    def setup_method(self):
        self.filt = SVFilter(sample_rate=SAMPLE_RATE).to(DEVICE)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((4, N_SAMPLES), 0.5, device=DEVICE)
        q = torch.full((4,), 0.5, device=DEVICE)
        filter_type = torch.zeros(4, 3, device=DEVICE)
        filter_type[:, 0] = 1.0
        out = self.filt(signal, cutoff, q, filter_type)
        assert out.shape == (4, N_SAMPLES)

    def test_static_lowpass_attenuates_highs(self):
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((1, N_SAMPLES), 0.3, device=DEVICE)
        q = torch.tensor([0.5], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 0] = 1.0
        filtered = self.filt(noise, cutoff, q, filter_type)
        fft_orig = torch.abs(torch.fft.rfft(noise[0]))
        fft_filt = torch.abs(torch.fft.rfft(filtered[0]))
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE).to(DEVICE)
        high_mask = freqs > 5000
        assert fft_filt[high_mask].mean() < fft_orig[high_mask].mean() * 0.5

    def test_time_varying_cutoff_produces_sweep(self):
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.linspace(0.1, 0.9, N_SAMPLES, device=DEVICE).unsqueeze(0)
        q = torch.tensor([0.5], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 0] = 1.0
        filtered = self.filt(noise, cutoff, q, filter_type)
        half = N_SAMPLES // 2
        first_half_high = torch.abs(torch.fft.rfft(filtered[0, :half])).mean()
        second_half_high = torch.abs(torch.fft.rfft(filtered[0, half:])).mean()
        assert second_half_high > first_half_high

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((1, N_SAMPLES), 0.01, device=DEVICE)
        q = torch.tensor([0.99], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 0] = 1.0
        out = self.filt(signal, cutoff, q, filter_type)
        assert not torch.isnan(out).any()
