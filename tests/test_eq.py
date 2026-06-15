import torch
import pytest
from loom.effects.eq import EQ
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestEQ:
    def setup_method(self):
        self.eq = EQ(sample_rate=SAMPLE_RATE)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES)
        out = self.eq(
            signal,
            low_gain=torch.full((4,), 0.5),
            mid_gain=torch.full((4,), 0.5),
            high_gain=torch.full((4,), 0.5),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_flat_eq_is_passthrough(self):
        """All gains at 0.5 (= 0dB) should be near passthrough."""
        torch.manual_seed(42)
        signal = torch.randn(1, N_SAMPLES)
        out = self.eq(
            signal,
            low_gain=torch.tensor([0.5]),
            mid_gain=torch.tensor([0.5]),
            high_gain=torch.tensor([0.5]),
        )
        assert torch.allclose(out, signal, atol=0.05)

    def test_low_boost_increases_low_energy(self):
        """Boosting low gain should increase energy below 200Hz."""
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES)
        out = self.eq(
            noise,
            low_gain=torch.tensor([1.0]),
            mid_gain=torch.tensor([0.5]),
            high_gain=torch.tensor([0.5]),
        )
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE)
        low_mask = freqs < 200
        orig_low = torch.abs(torch.fft.rfft(noise[0]))[low_mask].pow(2).sum()
        eq_low = torch.abs(torch.fft.rfft(out[0]))[low_mask].pow(2).sum()
        assert eq_low > orig_low * 1.5

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.eq(
            signal,
            low_gain=torch.tensor([0.0]),
            mid_gain=torch.tensor([1.0]),
            high_gain=torch.tensor([0.0]),
        )
        assert not torch.isnan(out).any()
