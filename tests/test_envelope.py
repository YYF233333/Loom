import torch
import pytest
from loom.envelope import ADSR
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestADSR:
    def setup_method(self):
        self.adsr = ADSR(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)

    def test_output_shape(self):
        batch = 4
        attack = torch.full((batch,), 0.3)
        decay = torch.full((batch,), 0.3)
        sustain = torch.full((batch,), 0.7)
        release = torch.full((batch,), 0.3)
        env = self.adsr(attack, decay, sustain, release)
        assert env.shape == (batch, N_SAMPLES)

    def test_range(self):
        """Envelope should be in [0, 1]."""
        attack = torch.tensor([0.3])
        decay = torch.tensor([0.3])
        sustain = torch.tensor([0.7])
        release = torch.tensor([0.3])
        env = self.adsr(attack, decay, sustain, release)
        assert env.min().item() >= -0.01
        assert env.max().item() <= 1.01

    def test_peak_at_attack_end(self):
        """Envelope should reach ~1.0 at the end of the attack phase."""
        attack = torch.tensor([0.3])
        decay = torch.tensor([0.5])
        sustain = torch.tensor([0.5])
        release = torch.tensor([0.3])
        env = self.adsr(attack, decay, sustain, release)
        assert env.max().item() > 0.95

    def test_sustain_level(self):
        """With long sustain, envelope should settle near sustain level."""
        attack = torch.tensor([0.1])
        decay = torch.tensor([0.2])
        sustain = torch.tensor([0.6])
        release = torch.tensor([0.1])
        env = self.adsr(attack, decay, sustain, release)
        mid = env[0, N_SAMPLES // 2].item()
        assert abs(mid - 0.6) < 0.15

    def test_zero_attack(self):
        """Zero attack should not produce NaN."""
        attack = torch.tensor([0.0])
        decay = torch.tensor([0.3])
        sustain = torch.tensor([0.5])
        release = torch.tensor([0.3])
        env = self.adsr(attack, decay, sustain, release)
        assert not torch.isnan(env).any()
