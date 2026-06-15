import torch
import pytest
from loom.effects.reverb import Reverb
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestReverb:
    def setup_method(self):
        self.reverb = Reverb(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES, device=DEVICE)
        out = self.reverb(
            signal,
            room_size=torch.full((4,), 0.5, device=DEVICE),
            decay=torch.full((4,), 0.5, device=DEVICE),
            damping=torch.full((4,), 0.3, device=DEVICE),
            mix=torch.full((4,), 0.5, device=DEVICE),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        out = self.reverb(
            signal,
            room_size=torch.tensor([0.5], device=DEVICE),
            decay=torch.tensor([0.5], device=DEVICE),
            damping=torch.tensor([0.3], device=DEVICE),
            mix=torch.tensor([0.0], device=DEVICE),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_reverb_tail(self):
        """Reverb output should have energy in the tail that input doesn't."""
        signal = torch.zeros(1, N_SAMPLES, device=DEVICE)
        signal[0, :4410] = torch.randn(4410, device=DEVICE)  # 0.1s burst then silence

        out = self.reverb(
            signal,
            room_size=torch.tensor([0.5], device=DEVICE),
            decay=torch.tensor([0.7], device=DEVICE),
            damping=torch.tensor([0.3], device=DEVICE),
            mix=torch.tensor([1.0], device=DEVICE),
        )
        tail_energy = out[0, N_SAMPLES // 2 :].pow(2).mean()
        assert tail_energy.item() > 1e-6

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        out = self.reverb(
            signal,
            room_size=torch.tensor([0.99], device=DEVICE),
            decay=torch.tensor([0.99], device=DEVICE),
            damping=torch.tensor([0.99], device=DEVICE),
            mix=torch.tensor([1.0], device=DEVICE),
        )
        assert not torch.isnan(out).any()
