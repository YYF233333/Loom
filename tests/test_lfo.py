import torch
import pytest
from loom.lfo import LFO
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestLFO:
    def setup_method(self):
        self.lfo = LFO(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def test_output_shape(self):
        batch = 4
        out = self.lfo(
            rate=torch.full((batch,), 0.5, device=DEVICE),
            depth=torch.full((batch,), 0.5, device=DEVICE),
            waveform=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch, device=DEVICE),
            phase=torch.full((batch,), 0.0, device=DEVICE),
        )
        assert out.shape == (batch, N_SAMPLES)

    def test_zero_depth_is_zero(self):
        out = self.lfo(
            rate=torch.tensor([0.5], device=DEVICE),
            depth=torch.tensor([0.0], device=DEVICE),
            waveform=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
            phase=torch.tensor([0.0], device=DEVICE),
        )
        assert out.abs().max().item() < 1e-6

    def test_rate_affects_frequency(self):
        waveform = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)
        depth = torch.tensor([1.0], device=DEVICE)
        phase = torch.tensor([0.0], device=DEVICE)

        out_slow = self.lfo(torch.tensor([0.0], device=DEVICE), depth, waveform, phase)
        out_fast = self.lfo(torch.tensor([1.0], device=DEVICE), depth, waveform, phase)

        slow_crossings = ((out_slow[0, :-1] * out_slow[0, 1:]) < 0).sum()
        fast_crossings = ((out_fast[0, :-1] * out_fast[0, 1:]) < 0).sum()
        assert fast_crossings > slow_crossings

    def test_no_nan(self):
        out = self.lfo(
            rate=torch.tensor([0.99], device=DEVICE),
            depth=torch.tensor([0.99], device=DEVICE),
            waveform=torch.tensor([[0.25, 0.25, 0.25, 0.25]], device=DEVICE),
            phase=torch.tensor([0.99], device=DEVICE),
        )
        assert not torch.isnan(out).any()
