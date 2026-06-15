import torch
import pytest
from loom.effects.delay import Delay
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestDelay:
    def setup_method(self):
        self.delay = Delay(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES, device=DEVICE)
        out = self.delay(
            signal,
            time=torch.full((4,), 0.5, device=DEVICE),
            feedback=torch.full((4,), 0.3, device=DEVICE),
            mix=torch.full((4,), 0.5, device=DEVICE),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        out = self.delay(
            signal,
            time=torch.tensor([0.5], device=DEVICE),
            feedback=torch.tensor([0.3], device=DEVICE),
            mix=torch.tensor([0.0], device=DEVICE),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_echo_at_delay_time(self):
        """Impulse should produce echo at the delay offset."""
        signal = torch.zeros(1, N_SAMPLES, device=DEVICE)
        signal[0, 1000] = 1.0

        out = self.delay(
            signal,
            time=torch.tensor([0.5], device=DEVICE),
            feedback=torch.tensor([0.5], device=DEVICE),
            mix=torch.tensor([1.0], device=DEVICE),
        )
        # Check there's energy in a region after the impulse
        # time=0.5 with log mapping [10ms,500ms] ~ 70ms ~ 3100 samples
        delayed_region = out[0, 2000:8000]
        assert delayed_region.abs().max().item() > 0.01

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        out = self.delay(
            signal,
            time=torch.tensor([0.99], device=DEVICE),
            feedback=torch.tensor([0.89], device=DEVICE),
            mix=torch.tensor([1.0], device=DEVICE),
        )
        assert not torch.isnan(out).any()
